import math
from typing import List, Tuple, Optional, Type, Callable, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from timm.models.layers import DropPath, Mlp
from huggingface_hub import PyTorchModelHubMixin
from monai.networks.blocks.dynunet_block import (
    get_output_padding,
    get_padding,
)
from py4cast.models.vision.utils import features_last_to_second

#HalfUNet imports
from collections import OrderedDict
from functools import reduce

#Unetrpp imports
from py4cast.models.vision.unetrpp import TransformerBlock

def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d Hiera, you could probably just implement this for n=4. (no promises)
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(
    x: torch.Tensor, conv: nn.Module, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)
    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


def undo_windowing(
    x: torch.Tensor, shape: List[int], mu_shape: List[int]
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
            [],
        )
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
        B, _, C = x.shape

        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension

            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = (
                [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            )
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


class Reroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
        stage_ends: List[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(
        self, x: torch.Tensor, block_idx: int, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum(
                    [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))],
                    [],
                )
                + [L - 1]
            )
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # If masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # If not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x


@dataclass_json
@dataclass
class HieraSettings:
    embed_dim: int = 96  # initial embed dim
    num_heads: int = 1  # initial number of heads
    stages: Tuple[int, ...] = (2, 2, 2, 2)  # number of layers per stage
    q_pool: int = 3  # number of q_pool stages
    q_stride: Tuple[int, ...] = (2, 2)  # len of the stride
    mask_unit_size: Tuple[int, ...] = (8, 8)  # must divide q_stride ** (#stages-1)
    # mask_unit_attn: which stages use mask unit attention?
    mask_unit_attn: Tuple[bool, ...] = (True, True, False, False)
    dim_mul: float = 2.0
    head_mul: float = 2.0
    patch_kernel: Tuple[int, ...] = (7, 7)
    patch_stride: Tuple[int, ...] = (4, 4)
    patch_padding: Tuple[int, ...] = (3, 3)
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    norm_layer: Union[str, nn.Module] = "LayerNorm"
    head_dropout: float = 0.0
    head_init_scale: float = 0.001
    sep_pos_embed: bool = False
    decoder: str = "hiera"  # hiera, unetrpp, halfunet


class Hiera(nn.Module, PyTorchModelHubMixin):
    onnx_supported = False
    input_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    output_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    settings_kls = HieraSettings

    # @has_config
    def __init__(
        self,
        num_input_features: int = 21,  # RGB=3, titan=21
        num_output_features: int = 21,  # number of classes
        settings=HieraSettings,
        input_shape: Tuple[int, ...] = (64, 64),  # nb_pixel x nb_pixel
    ) -> None:
        super().__init__()

        if (input_shape[0] % 32 != 0) or (input_shape[1] % 32 != 0):
            raise Exception("input shapes need to be multiples of 32")
        self.copy_embed_dim = settings.embed_dim
        """
        ENCODER PART
        """

        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(settings.norm_layer, str):
            settings.norm_layer = partial(getattr(nn, settings.norm_layer), eps=1e-6)

        depth = sum(settings.stages)  # sum of layers per stage
        self.patch_stride = settings.patch_stride
        self.tokens_spatial_shape = [
            i // s for (i, s) in list(zip(input_shape, settings.patch_stride))
        ]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(settings.mask_unit_size)
        flat_q_stride = math.prod(settings.q_stride)

        assert settings.q_pool < len(settings.stages)
        self.q_pool, self.q_stride = settings.q_pool, settings.q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, settings.mask_unit_size
        self.mask_spatial_shape = [
            i // s for (i, s) in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [
            sum(settings.stages[:i]) - 1 for i in range(1, len(settings.stages) + 1)
        ]

        self.patch_embed = PatchEmbed(
            num_input_features,
            settings.embed_dim,
            settings.patch_kernel,
            settings.patch_stride,
            settings.patch_padding,
        )

        self.sep_pos_embed = settings.sep_pos_embed
        if settings.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    settings.embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape, settings.embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_tokens, settings.embed_dim)
            )

        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_shape,
            settings.patch_stride,
            [settings.q_stride] * len(self.stage_ends[:-1]),
        )
        self.reroll = Reroll(
            input_shape,
            settings.patch_stride,
            [settings.q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            settings.q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[: settings.q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, settings.drop_path_rate, depth)]

        self.bottleneck_shape = [
            int(
                (input_shape[0] * input_shape[1])
                // (16 * 4 ** (len(settings.stages) - 1))
            ),
            int(settings.embed_dim * 2 ** (len(settings.stages) - 1)),
        ]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = settings.embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = settings.mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(settings.embed_dim * settings.dim_mul)
                settings.num_heads = int(settings.num_heads * settings.head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=settings.embed_dim,
                dim_out=dim_out,
                heads=settings.num_heads,
                mlp_ratio=settings.mlp_ratio,
                drop_path=dpr[i],
                norm_layer=settings.norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            settings.embed_dim = dim_out
            self.blocks.append(block)
        self.norm = settings.norm_layer(settings.embed_dim)
        self.head = Head(
            settings.embed_dim, num_output_features, dropout_rate=settings.head_dropout
        )

        # Initialize everything
        if settings.sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(
            settings.head_init_scale
        )  # useful only if classification
        self.head.projection.bias.data.mul_(settings.head_init_scale)

        """
        ENCODER PART
        """

        if settings.decoder == "hiera":
            self.decoder = HieraDecoder(
                output_channels = num_output_features,
                input_shape = input_shape,                
                bottleneck_shape = self.bottleneck_shape, # (Patched image, embedding)
                settings = UnetrppDecoderSettings,
            )
        elif settings.decoder == "unetrpp":
            self.decoder = UnetrppDecoder(
                output_channels = num_output_features,
                input_shape = input_shape,                
                bottleneck_shape = self.bottleneck_shape, # (Patched image, embedding)
                settings = UnetrppDecoderSettings,
            )

        elif settings.decoder == "halfunet":
            self.decoder = HalfUNetDecoder(
                output_channels=num_output_features,
                input_shape=input_shape,
                bottleneck_shape = self.bottleneck_shape, # (Patched image, embedding)
                settings=HalfUnetDecoderSettings(),
            )
        else:
            raise Exception(f"unknwon decoder: {settings.decoder}")

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def get_pos_embed(self) -> torch.Tensor:
        if self.sep_pos_embed:
            return self.pos_embed_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """

        # print("x in hiera forward before last to second", x.shape)
        x = features_last_to_second(x)

        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]
        intermediates = []
        # print("x in hiera forward after last to second", x.shape)
        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        # print("x in hiera forward after patch embed", x.shape)

        x = x + self.get_pos_embed()
        # print("x in hiera forward after get pos embed", x.shape)

        x = self.unroll(x)
        # print("x in hiera forward after unroll", x.shape)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )
        # print("x in hiera forward after mask is not none", x.shape)
        skip_list = []
        for i, blk in enumerate(self.blocks):
            if i in self.stage_ends:
                skip_list.append(x)
            x = blk(x)
            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))
            # print("x in hiera forward after blk=",i, x.shape)

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order
        if return_intermediates:
            return x, intermediates

        # print("x in hiera forward FINAL", x.shape)
        return self.decoder(skip_list, x)


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


class MaskedAutoencoderHiera(Hiera):
    """Masked Autoencoder with Hiera backbone"""

    def __init__(
        self,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        **kwdargs,
    ):
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwdargs,
        )

        del self.norm, self.head
        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]
        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pixel_label_2d(
        self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(
        self, input_vid: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, :: self.patch_stride[0], :, :]

        size = self.pred_stride
        label = input_vid.unfold(3, size, size).unfold(4, size, size)
        label = label.permute(
            0, 2, 3, 4, 5, 6, 1
        )  # Different from 2d, mistake during training lol
        label = label.flatten(1, 3).flatten(2)
        label = label[mask]

        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, return_intermediates=True)
        # Resolution unchanged after q_pool stages, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        return x, mask

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_stride) == 2:
            label = self.get_pixel_label_2d(x, mask)
        elif len(self.q_stride) == 3:
            label = self.get_pixel_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.6,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask = self.forward_encoder(x, mask_ratio, mask=mask)
        pred, pred_mask = self.forward_decoder(
            latent, mask
        )  # pred_mask is mask at resolution of *prediction*

        # Toggle mask, to generate labels for *masked* tokens
        return (*self.forward_loss(x, pred, ~pred_mask), mask)


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // self.heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input should be of shape [batch, tokens, channels]."""
        B, N, _ = x.shape

        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[
            [torch.Tensor], torch.Tensor
        ] = lambda x: x,  # x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


# ------------------------------------------------------------------------------#

class _a:
    a=1


@dataclass_json
@dataclass
class HieraDecoderSettings:
    linear_upsampling: bool = True
    kernel_size: Union[Sequence[int], int] = 3
    upsample_kernel_size: Union[Sequence[int], int] = 4
    linear_upsampling: bool = True

class HieraDecoder(nn.Module):
    def __init__(
        self,
        input_shape: Union[Sequence[int], int],
        output_channels: int,
        bottleneck_shape: Union[Sequence[int], int], # (Patched image, embedding)
        settings = HieraSettings,
    ) -> None:
        super().__init__()

        input_embed = bottleneck_shape[1]
        self.stage_blocks = nn.ModuleList()
        self.transp_conv = nn.ModuleList()
        self.input_shape = input_shape
        padding = get_padding(settings.upsample_kernel_size, settings.upsample_kernel_size)

        # UPSAMPLING + CONV
        for i in range(len(settings.stages) - 1):
            output_embed = int(input_embed // 2)
            if settings.linear_upsampling:
                self.transp_conv.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=settings.upsample_kernel_size),
                        nn.Conv1d(
                            input_embed,
                            output_embed,
                            kernel_size=settings.kernel_size,
                            padding=1,
                        ),
                    )
                )
            else:
                self.transp_conv.append(
                    nn.ConvTranspose1d(
                        input_embed,
                        output_embed,
                        kernel_size=settings.upsample_kernel_size,
                        stride=settings.upsample_kernel_size,
                        padding=padding,
                        output_padding=get_output_padding(
                            settings.upsample_kernel_size, settings.upsample_kernel_size, padding
                        ),
                        dilation=1,
                    )
                )
            input_embed = output_embed

        self.transp_conv.append(
            nn.Sequential(
                nn.Upsample(scale_factor=settings.upsample_kernel_size * settings.upsample_kernel_size),
                nn.Conv1d(input_embed, output_channels, kernel_size=settings.kernel_size, padding=1),
            )
        )

        # ATTENTION BLOCK
        input_embed = bottleneck_shape[1]
        for i in range(len(settings.stages)):
            settings.num_heads = int(settings.num_heads * settings.head_mul)
            block = HieraBlock(
                dim=input_embed,
                dim_out=input_embed,
                heads=settings.num_heads,
                mlp_ratio=4.0,
                drop_path=0.0,
                norm_layer=nn.LayerNorm,
                q_stride=1,
                window_size=0,
                use_mask_unit_attn=False,
            )
            input_embed = int(input_embed // 2)
            self.stage_blocks.append(block)

    def forward(self, skip_list, x):
        for i in range(len(skip_list)):
            k = len(skip_list) - i - 1
            x = skip_list[k] + x
            x = self.stage_blocks[i](x)
            x = torch.permute(x, (0, 2, 1))
            x = self.transp_conv[i](x)
            x = torch.permute(x, (0, 2, 1))
        B, _, C = x.shape
        x = x.reshape(B, self.input_shape[0], self.input_shape[1], C)
        return x


# ------------------------------------------------------------------------------#

class _b:
    a=1

@dataclass_json
@dataclass(slots=True)
class HalfUNetDecoderSettings:
    embed_dim: int = 96  # initial embed dim
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        bias: bool = False,
        kernel_size=3,
        padding="same",
        dilation=1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.sepconv = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=out_channels // 2,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x2 = self.sepconv(x)
        x = torch.cat([x, x2], dim=1)
        x = self.bn(x)
        return self.relu(x)


class HalfUNetDecoder(nn.Module):
    settings_kls = HalfUnetDecoderSettings
    onnx_supported = True
    input_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    output_dims: Tuple[str, ...] = ("batch", "height", "width", "features")

    def __init__(
        self,
        input_shape: Union[Sequence[int], int],
        output_channels: int,
        bottleneck_shape: Union[Sequence[int], int], # [Patched image, embedding]
        settings = HalfUnetDecoderSettings,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape

        self.up2 = nn.Upsample(scale_factor=4)
        self.down2 = nn.Upsample(scale_factor=1 / 2)

        self.up3 = nn.Upsample(scale_factor=16)
        self.down3 = nn.Upsample(scale_factor=1 / 4)

        self.up4 = nn.Upsample(scale_factor=64)
        self.down4 = nn.Upsample(scale_factor=1 / 8)

        self.decoder = self._block(
            settings.embed_dim,
            settings.embed_dim,
            name="decoder",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
            grid_shape=input_shape,
        )

        self.outconv = nn.Conv2d(
            in_channels=settings.embed_dim,
            out_channels=output_channels,
            kernel_size=1,
            bias=settings.bias,
        )

        self.activation = getattr(nn, settings.last_activation)()

    def forward(self, skip_list, x):
        skip_list[0] = torch.permute(skip_list[0], (0, 2, 1))

        skip_list[1] = self.down2(skip_list[1])
        skip_list[1] = torch.permute(skip_list[1], (0, 2, 1))
        skip_list[1] = self.up2(skip_list[1])

        skip_list[2] = self.down3(skip_list[2])
        skip_list[2] = torch.permute(skip_list[2], (0, 2, 1))
        skip_list[2] = self.up3(skip_list[2])

        skip_list[3] = self.down4(skip_list[3])
        skip_list[3] = torch.permute(skip_list[3], (0, 2, 1))
        skip_list[3] = self.up4(skip_list[3])

        summed = reduce(
            torch.Tensor.add_,
            [skip_list[0], skip_list[1], skip_list[2], skip_list[3]],
            torch.zeros_like(skip_list[0]),
        )
        summed = self.up3(summed)  # *16
        B, C, _ = summed.shape
        summed = summed.reshape(B, C, self.input_shape[0], self.input_shape[1])
        dec = self.decoder(summed)
        conv = self.outconv(dec)
        act = self.activation(conv)
        act = torch.permute(act, (0, 2, 3, 1))
        return act

    @staticmethod
    def _block(
        in_channels,
        features,
        name,
        bias=False,
        use_ghost: bool = False,
        dilation: int = 1,
        padding="same",
        grid_shape: Tuple[int, int] = None,
    ):
        if use_ghost:
            layers = nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "ghost1",
                            GhostModule(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                        (
                            name + "ghost2",
                            GhostModule(
                                in_channels=features,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                    ]
                )
            )
        else:
            layers = nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "conv1",
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                        (name + "norm1", nn.BatchNorm2d(num_features=features)),
                        (name + "relu1", nn.ReLU(inplace=True)),
                        (
                            name + "conv2",
                            nn.Conv2d(
                                in_channels=features,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                        (name + "norm2", nn.BatchNorm2d(num_features=features)),
                        (name + "relu2", nn.ReLU(inplace=True)),
                    ]
                )
            )
        return layers


# ------------------------------------------------------------------------------#
class _c:
    a=1

@dataclass_json
@dataclass
class UnetrppDecoderSettings:
    num_heads: int = 1  # initial number of heads
    stages: Tuple[int, ...] = (2, 3, 16, 3)
    head_mul: float = 2.0
    linear_upsampling: bool = True
    kernel_size: Union[Sequence[int], int] = 3
    upsample_kernel_size: Union[Sequence[int], int] = 4
    proj_size = 64
    use_scaled_dot_product_CA = "torch"


class UnetrppDecoder(nn.Module):  # "=" UnetrUpBlock
    def __init__(
        self,
        input_shape: Union[Sequence[int], int],
        output_channels: int,
        bottleneck_shape: Union[Sequence[int], int], # (Patched image, embedding)
        settings = UnetrppDecoderSettings,
    ) -> None:
        super().__init__()
        
        input_patch = bottleneck_shape[0]
        input_embed = bottleneck_shape[1]
        self.stage_blocks = nn.ModuleList()
        self.transp_conv = nn.ModuleList()
        self.input_shape = input_shape
        padding = get_padding(settings.upsample_kernel_size, settings.upsample_kernel_size)


        # UPSAMPLING + CONV
        for i in range(len(settings.stages) - 1):
            output_embed = int(input_embed // 2)
            if settings.linear_upsampling:
                self.transp_conv.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=settings.upsample_kernel_size),
                        nn.Conv1d(
                            input_embed,
                            output_embed,
                            kernel_size=settings.kernel_size,
                            padding=1,
                        ),
                    )
                )
            else:
                self.transp_conv.append(
                    nn.ConvTranspose1d(
                        input_embed,
                        output_embed,
                        kernel_size=settings.upsample_kernel_size,
                        stride=settings.upsample_kernel_size,
                        padding=padding,
                        output_padding=get_output_padding(
                            settings.upsample_kernel_size, settings.upsample_kernel_size, padding
                        ),
                        dilation=1,
                    )
                )
            input_embed = output_embed

        self.transp_conv.append(
            nn.Sequential(
                nn.Upsample(scale_factor=settings.upsample_kernel_size * settings.upsample_kernel_size),
                nn.Conv1d(input_embed, output_channels, kernel_size=settings.kernel_size, padding=1),
            )
        )

        # ATTENTION BLOCK
        input_embed = bottleneck_shape[1]
        for i in range(len(settings.stages)):
            settings.num_heads = int(settings.num_heads * settings.head_mul)
            block = TransformerBlock(
                input_size=input_patch,
                hidden_size=input_embed,
                num_heads=settings.num_heads,
                dropout_rate=0.1,
                pos_embed=True,
                proj_size=settings.proj_size,
                attention_code=settings.use_scaled_dot_product_CA,
            )
            input_embed = int(input_embed // 2)
            input_patch = int(input_patch * 4)
            self.stage_blocks.append(block)

    def forward(self, skip_list, x):
        for i in range(len(skip_list)):
            k = len(skip_list) - i - 1
            height, width = self.input_shape
            #skip connection
            x = skip_list[k] + x
            #attention block
            x = torch.permute(x, (0, 2, 1))
            B, C, _ = x.shape
            x = x.reshape(B, C, int(height/(4*2**k)), int(width/(4*2**k)))
            x = self.stage_blocks[i](x)
            #conv
            B, C, H, W = x.shape
            x = x.reshape(B, C, H*W)
            x = self.transp_conv[i](x)
            x = torch.permute(x, (0, 2, 1))
        B, _, C = x.shape
        x = x.reshape(B, C, self.input_shape[0], self.input_shape[1])
        x = torch.permute(x, (0, 2, 3, 1))
        return x
