
from monai.networks.blocks.dynunet_block import (
    UnetOutBlock,
    UnetResBlock,
    get_conv_layer,
    get_output_padding,
    get_padding,
)
import math
from typing import List, Tuple, Optional, Type, Callable, Dict, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from timm.models.layers import DropPath, Mlp
from py4cast.models.vision.utils import features_last_to_second, features_second_to_last


@dataclass_json
@dataclass
class HieraSettings:
    embed_dim: int = 96  # initial embed dim
    num_heads: int = 1  # initial number of heads
    stages: Tuple[int, ...] = (2, 3, 16, 3)
    q_pool: int = 3  # number of q_pool stages
    q_stride: Tuple[int, ...] = (2, 2)
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
    decoder: str = "hiera"



class Hiera(nn.Module, PyTorchModelHubMixin):

    onnx_supported = False
    input_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    output_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    settings_kls = HieraSettings

    @has_config
    def __init__(
        self,
        num_input_features: int = 3, #RGB
        num_output_features: int = 1000, #number of classes
        settings = HieraSettings,
        input_shape: Tuple[int, ...] = (224, 224),  #nb_pixel x nb_pixel
    ) -> None:
        super().__init__()

        """
        ENCODER PART
        """

        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(settings.norm_layer, str):
            settings.norm_layer = partial(getattr(nn, settings.norm_layer), eps=1e-6)

        depth = sum(settings.stages) #number of layers
        self.patch_stride = settings.patch_stride
        self.tokens_spatial_shape = [i // s for (i, s) in list(zip(input_shape, settings.patch_stride))]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(settings.mask_unit_size)
        flat_q_stride = math.prod(settings.q_stride)

        assert settings.q_pool < len(settings.stages)
        self.q_pool, self.q_stride = settings.q_pool, settings.q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, settings.mask_unit_size
        self.mask_spatial_shape = [
            i // s for (i, s) in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(settings.stages[:i]) - 1 for i in range(1, len(settings.stages) + 1)]

        self.patch_embed = PatchEmbed(
            num_input_features, settings.embed_dim, settings.patch_kernel, settings.patch_stride, settings.patch_padding
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
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, settings.embed_dim))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_shape, settings.patch_stride, [settings.q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            input_shape,
            settings.patch_stride,
            [settings.q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            settings.q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:settings.q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, settings.drop_path_rate, depth)]

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
        self.head = Head(settings.embed_dim, num_output_features, dropout_rate=settings.head_dropout)

        # Initialize everything
        if settings.sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(settings.head_init_scale)
        self.head.projection.bias.data.mul_(settings.head_init_scale)

        """
        DECODER PART
        """

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



        # Number of pixels after stem layer
        no_pixels = (input_shape[0] * input_shape[1]) // (
            settings.downsampling_rate**2
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=settings.spatial_dims,
            in_channels=settings.hidden_size,
            out_channels=settings.hidden_size // 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=settings.norm_name,
            out_size=no_pixels // 16,
            linear_upsampling=settings.linear_upsampling,
            proj_size=settings.proj_size,
            use_scaled_dot_product_CA=settings.attention_code,
        )


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

        #print("x in hiera forward before last to second", x.shape)
        x = features_last_to_second(x)


        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]
        intermediates = []
        print("x in hiera forward after last to second", x.shape)
        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        print("x in hiera forward after patch embed", x.shape)

        x = x + self.get_pos_embed()
        print("x in hiera forward after get pos embed", x.shape)

        x = self.unroll(x)
        print("x in hiera forward after unroll", x.shape)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )
        print("x in hiera forward after mask is not none", x.shape)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))
            print("x in hiera forward after blk=",i, x.shape)
        



        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))
            print("x in hiera forward after blk=",i, x.shape)

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order        
        if return_intermediates:
            return x, intermediates




        #x = features_second_to_last(x)
        print("x in hiera forward FINAL", x.shape)

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

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
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
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x



def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values



################################################################################




class HieraDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        linear_upsampling: bool = False,
        settings = HieraSettings,
    ) -> None:
        super().__init__()
# UPSAMPLING
        padding = get_padding(upsample_kernel_size, upsample_kernel_size)
        if linear_upsampling:
            self.transp_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=upsample_kernel_size),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=1
                ),
            )
        else:
            self.transp_conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_kernel_size,
                padding=padding,
                output_padding=get_output_padding(
                    upsample_kernel_size, upsample_kernel_size, padding
                ),
                dilation=1,
            )
# ATTENTION BLOCK
        self.decoder_block = nn.ModuleList()
        stage_blocks = []
        for i in range(len(settings.stages)):
            out_channels = int(in_channels * settings.dim_mul)
            settings.num_heads = int(settings.num_heads * settings.head_mul)
            block = HieraBlock(
                dim = in_channels,
                dim_out = out_channels,
                heads = settings.num_heads,
                mlp_ratio = 4.0,
                drop_path = 0.0,
                norm_layer = nn.LayerNorm,
                q_stride = 1,
                window_size = 0,
                use_mask_unit_attn = False,
            )
            in_channels = out_channels
            stage_blocks.append(block)
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def forward(self, skip_list, x):
        for i in range(len(skip_list)):
            x_temp = self.transp_conv(x)
            x_temp = skip_list[i] + x_temp
            x = self.decoder_block[i](x_temp)
        return(x)