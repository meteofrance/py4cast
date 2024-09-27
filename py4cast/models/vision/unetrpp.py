"""
UnetR++ Vision transformer based on: "Shaker et al.,
Adapted from https://github.com/Amshaker/unetr_plus_plus
"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses_json import dataclass_json
from monai.networks.blocks.dynunet_block import (
    UnetOutBlock,
    UnetResBlock,
    get_conv_layer,
    get_output_padding,
    get_padding,
)
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from torch.nn.functional import scaled_dot_product_attention

from py4cast.models.base import ModelABC
from py4cast.models.vision.utils import features_last_to_second, features_second_to_last


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    lower = norm_cdf((a - mean) / std)
    upper = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * lower - 1, 2 * upper - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    Modified to work with both 2d and 3d data (spatial_dims).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed=False,
        spatial_dims=2,
        proj_size: int = 64,
        attention_code: str = "torch",
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
            proj_size: size of the projection space for Spatial Attention.
            use_scaled_dot_product_CA : bool argument to determine if torch's scaled_dot_product_attenton
            is used for Channel Attention.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
            proj_size=proj_size,
            attention_code=attention_code,
        )
        self.conv51 = UnetResBlock(
            spatial_dims,
            hidden_size,
            hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="batch",
        )

        if spatial_dims == 2:
            self.conv8 = nn.Sequential(
                nn.Dropout2d(0.1, False), nn.Conv2d(hidden_size, hidden_size, 1)
            )
        else:
            self.conv8 = nn.Sequential(
                nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1)
            )

        self.pos_embed = None
        self.spatial_dims = spatial_dims
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        if self.spatial_dims == 2:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).permute(0, 2, 1)
        else:
            B, C, H, W, D = x.shape
            x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        if self.spatial_dims == 2:
            attn_skip = attn.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)

        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class EPA(nn.Module):
    """
    Efficient Paired Attention Block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    Modifications :
    - adds compatibility with 2d inputs
    - adds an option to use torch's scaled dot product instead of the original implementation
    This should enable the use of flash attention in the future.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads=4,
        qkv_bias=False,
        channel_attn_drop=0.1,
        spatial_attn_drop=0.1,
        proj_size: int = 64,
        attention_code: str = "torch",
    ):
        super().__init__()
        self.num_heads = num_heads

        if attention_code not in ["torch", "flash", "manual"]:
            raise NotImplementedError(
                "Attention code should be one of 'torch', 'flash' or 'manual'"
            )
        if attention_code == "flash":
            from flash_attn import flash_attn_func

            self.attn_func = flash_attn_func
            self.use_scaled_dot_product_CA = True
        elif attention_code == "torch":
            self.attn_func = scaled_dot_product_attention
            self.use_scaled_dot_product_CA = True
        else:
            self.use_scaled_dot_product_CA = False

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))

        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        if not self.use_scaled_dot_product_CA:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        # Matrix index, Batch, Head, Dimensions, Features
        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        # Batch, Head, Dimensions, Features
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        # Batch, Head, Features, Dimensions
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected, v_SA_projected = map(
            lambda args: torch.einsum("bhdn,nk->bhdk", *args),
            zip((k_shared, v_SA), (self.EF, self.EF)),
        )

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1).type_as(q_shared)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1).type_as(k_shared)
        if self.use_scaled_dot_product_CA:
            x_CA = self.attn_func(q_shared, k_shared, v_CA, dropout_p=self.attn_drop.p)
        else:
            attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
            attn_CA = attn_CA.softmax(dim=-1)
            attn_CA = self.attn_drop(attn_CA)
            x_CA = attn_CA @ v_CA

        x_CA = x_CA.permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (
            q_shared.permute(0, 1, 3, 2) @ k_shared_projected
        ) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = attn_SA @ v_SA_projected.transpose(-2, -1)
        x_SA = x_SA.permute(0, 3, 1, 2).reshape(B, N, C)

        return x_CA + x_SA

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature", "temperature2"}


einops, _ = optional_import("einops")


class UnetrPPEncoder(nn.Module):
    def __init__(
        self,
        input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],
        dims=[32, 64, 128, 256],
        depths=[3, 3, 3, 3],
        num_heads=4,
        spatial_dims=2,
        in_channels=4,
        dropout=0.0,
        transformer_dropout_rate=0.1,
        downsampling_rate: int = 4,
        proj_size: int = 64,
        attention_code: str = "torch",
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                in_channels,
                dims[0],
                kernel_size=downsampling_rate,
                stride=downsampling_rate,
                dropout=dropout,
                conv_only=True,
            ),
            get_norm_layer(name=("group", {"num_groups": 4}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(
                    spatial_dims,
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                    dropout=dropout,
                    conv_only=True,
                ),
                get_norm_layer(
                    name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=input_size[i],
                        hidden_size=dims[i],
                        num_heads=num_heads,
                        dropout_rate=transformer_dropout_rate,
                        pos_embed=True,
                        proj_size=proj_size,
                        attention_code=attention_code,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        self.spatial_dims = spatial_dims

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                if self.spatial_dims == 2:
                    x = einops.rearrange(x, "b c h w -> b (h w) c")
                else:
                    x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
        linear_upsampling: bool = False,
        proj_size: int = 64,
        use_scaled_dot_product_CA: bool = True,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        padding = get_padding(upsample_kernel_size, upsample_kernel_size)
        if spatial_dims == 2:
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
        else:
            if linear_upsampling:
                self.transp_conv = nn.Sequential(
                    nn.Upsample(scale_factor=upsample_kernel_size, mode="trilinear"),
                    nn.Conv3d(
                        in_channels, out_channels, kernel_size=kernel_size, padding=1
                    ),
                )
            else:
                self.transp_conv = nn.ConvTranspose3d(
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

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder:
            self.decoder_block.append(
                UnetResBlock(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                )
            )
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=out_size,
                        hidden_size=out_channels,
                        num_heads=num_heads,
                        dropout_rate=0.1,
                        pos_embed=True,
                        proj_size=proj_size,
                        attention_code=use_scaled_dot_product_CA,
                    )
                )
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


@dataclass_json
@dataclass
class UNETRPPSettings:
    hidden_size: int = 256
    num_heads: int = 4
    pos_embed: str = "perceptron"
    norm_name: Union[Tuple, str] = "instance"
    dropout_rate: float = 0.0
    depths: Tuple[int, ...] = (3, 3, 3, 3)
    conv_op: str = "Conv2d"
    do_ds = False
    spatial_dims = 2
    linear_upsampling: bool = False
    downsampling_rate: int = 4
    proj_size: int = 64

    # Specify the attention implementation to use
    # Options: "torch" : scaled_dot_product_attention from torch.nn.functional
    #          "flash" : flash_attention from flash_attn (loose dependency imported only if needed)
    #          "manual" : manual implementation from the original paper
    attention_code: str = "torch"


class UNETRPP(ModelABC, nn.Module):

    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    onnx_supported = False
    input_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    output_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    settings_kls = UNETRPPSettings

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        settings: UNETRPPSettings,
        input_shape: tuple,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()

        self.do_ds = settings.do_ds
        self.conv_op = getattr(nn, settings.conv_op)
        self.num_classes = num_output_features
        if not (0 <= settings.dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if settings.pos_embed not in ["conv", "perceptron"]:
            raise KeyError(
                f"Position embedding layer of type {settings.pos_embed} is not supported."
            )
        # we have first a stem layer with stride=subsampling_rate and k_size=subsampling_rate
        # followed by 3 successive downsampling layer (k=2, stride=2)
        dim_divider = (2**3) * settings.downsampling_rate
        if settings.spatial_dims == 2:
            self.feat_size = (
                input_shape[0] // dim_divider,
                input_shape[1] // dim_divider,
            )
        else:
            self.feat_size = (
                input_shape[0] // dim_divider,
                input_shape[1] // dim_divider,
                input_shape[2] // dim_divider,
            )

        self.hidden_size = settings.hidden_size
        self.spatial_dims = settings.spatial_dims
        # Number of pixels after stem layer
        no_pixels = (input_shape[0] * input_shape[1]) // (
            settings.downsampling_rate**2
        )
        encoder_input_size = [
            no_pixels,
            no_pixels // 4,
            no_pixels // 16,
            no_pixels // 64,
        ]
        h_size = settings.hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=encoder_input_size,
            dims=(
                h_size // 8,
                h_size // 4,
                h_size // 2,
                h_size,
            ),
            depths=settings.depths,
            num_heads=settings.num_heads,
            spatial_dims=settings.spatial_dims,
            in_channels=num_input_features,
            downsampling_rate=settings.downsampling_rate,
            proj_size=settings.proj_size,
            attention_code=settings.attention_code,
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=settings.spatial_dims,
            in_channels=num_input_features,
            out_channels=settings.hidden_size // 16,
            kernel_size=3,
            stride=1,
            norm_name=settings.norm_name,
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
        self.decoder4 = UnetrUpBlock(
            spatial_dims=settings.spatial_dims,
            in_channels=settings.hidden_size // 2,
            out_channels=settings.hidden_size // 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=settings.norm_name,
            out_size=no_pixels // 4,
            linear_upsampling=settings.linear_upsampling,
            proj_size=settings.proj_size,
            use_scaled_dot_product_CA=settings.attention_code,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=settings.spatial_dims,
            in_channels=settings.hidden_size // 4,
            out_channels=settings.hidden_size // 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=settings.norm_name,
            out_size=no_pixels,
            linear_upsampling=settings.linear_upsampling,
            proj_size=settings.proj_size,
            use_scaled_dot_product_CA=settings.attention_code,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=settings.spatial_dims,
            in_channels=settings.hidden_size // 8,
            out_channels=settings.hidden_size // 16,
            kernel_size=3,
            upsample_kernel_size=settings.downsampling_rate,
            norm_name=settings.norm_name,
            out_size=no_pixels * (settings.downsampling_rate**2),
            conv_decoder=True,
            linear_upsampling=settings.linear_upsampling,
            proj_size=settings.proj_size,
            use_scaled_dot_product_CA=settings.attention_code,
        )
        self.out1 = UnetOutBlock(
            spatial_dims=settings.spatial_dims,
            in_channels=settings.hidden_size // 16,
            out_channels=num_output_features,
        )
        if self.do_ds:
            self.out2 = UnetOutBlock(
                spatial_dims=settings.spatial_dims,
                in_channels=settings.hidden_size // 8,
                out_channels=num_output_features,
            )
            self.out3 = UnetOutBlock(
                spatial_dims=settings.spatial_dims,
                in_channels=settings.hidden_size // 4,
                out_channels=num_output_features,
            )

    def proj_feat(self, x):
        if self.spatial_dims == 2:
            x = x.view(
                x.size(0), self.feat_size[0], self.feat_size[1], self.hidden_size
            )
        else:
            x = x.view(
                x.size(0),
                self.feat_size[0],
                self.feat_size[1],
                self.feat_size[2],
                self.hidden_size,
            )

        if self.spatial_dims == 2:
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    def forward(self, x_in):

        x_in = features_last_to_second(x_in)

        _, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        logits = features_second_to_last(logits)

        return logits
