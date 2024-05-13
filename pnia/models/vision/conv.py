"""
Convolutional neural network models
for pn-ia.
"""
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import Tuple

import torch
from dataclasses_json import dataclass_json
from torch import nn

from pnia.datasets.base import Item, Statics
from pnia.models.base import ModelABC, ModelInfo
from pnia.models.vision.utils import (
    features_last_to_second,
    features_second_to_last,
    transform_batch_vision,
)


@dataclass_json
@dataclass(slots=True)
class HalfUnetSettings:

    num_filters: int = 64
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


class HalfUnet(ModelABC, nn.Module):
    settings_kls = HalfUnetSettings
    onnx_supported = True

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        settings: HalfUnetSettings,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder1 = self._block(
            num_input_features,
            settings.num_filters,
            name="enc1",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc2",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc3",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
        )

        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc4",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
        )
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc5",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
        )
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=16)

        self.decoder = self._block(
            settings.num_filters,
            settings.num_filters,
            name="decoder",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
        )

        self.outconv = nn.Conv2d(
            in_channels=settings.num_filters,
            out_channels=num_output_features,
            kernel_size=1,
            bias=settings.bias,
        )

        self.activation = getattr(nn, settings.last_activation)()

    def transform_statics(self, statics: Statics) -> Statics:
        return statics

    def transform_batch(
        self, batch: Item
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return transform_batch_vision(batch)

    def forward(self, x):
        x = features_last_to_second(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        summed = reduce(
            torch.Tensor.add_,
            [enc1, self.up2(enc2), self.up3(enc3), self.up4(enc4), self.up5(enc5)],
            torch.zeros_like(enc1),
        )
        dec = self.decoder(summed)

        return features_second_to_last(self.activation(self.outconv(dec)))

    @cached_property
    def info(self) -> ModelInfo:
        """
        Return information on this model
        """
        return ModelInfo(output_dim=2)

    @staticmethod
    def _block(
        in_channels,
        features,
        name,
        bias=False,
        use_ghost: bool = False,
        dilation: int = 1,
        padding="same",
    ):
        if use_ghost:
            return nn.Sequential(
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

            return nn.Sequential(
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


@dataclass_json
@dataclass(slots=True)
class UnetSettings:

    init_features: int = 64


class Unet(ModelABC, nn.Module):
    """
    Returns a u_net architecture, with uninitialised weights, matching desired numbers of input and output channels.

    Implementation from the original paper: https://arxiv.org/pdf/1505.04597.pdf.
    """

    settings_kls = UnetSettings
    onnx_supported = True

    def __init__(
        self,
        num_input_features: int = 3,
        num_output_features: int = 1,
        settings: UnetSettings = UnetSettings(),
    ):
        super(Unet, self).__init__()

        num_channels = settings.init_features

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = Unet._block(num_input_features, num_channels, name="enc1")
        self.encoder2 = Unet._block(num_channels, num_channels * 2, name="enc2")
        self.encoder3 = Unet._block(num_channels * 2, num_channels * 4, name="enc3")
        self.encoder4 = Unet._block(num_channels * 4, num_channels * 8, name="enc4")
        self.bottleneck = Unet._block(
            num_channels * 8, num_channels * 16, name="bottleneck"
        )

        self.upconv4 = nn.ConvTranspose2d(
            num_channels * 16, num_channels * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Unet._block(
            (num_channels * 8) * 2, num_channels * 8, name="dec4"
        )
        self.upconv3 = nn.ConvTranspose2d(
            num_channels * 8, num_channels * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Unet._block(
            (num_channels * 4) * 2, num_channels * 4, name="dec3"
        )
        self.upconv2 = nn.ConvTranspose2d(
            num_channels * 4, num_channels * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Unet._block(
            (num_channels * 2) * 2, num_channels * 2, name="dec2"
        )
        self.upconv1 = nn.ConvTranspose2d(
            num_channels * 2, num_channels, kernel_size=2, stride=2
        )
        self.decoder1 = Unet._block(num_channels * 2, num_channels, name="dec1")

        self.conv = nn.Conv2d(num_channels, num_output_features, kernel_size=1)

    def forward(self, x):
        """
        Description of the architecture from the original paper (https://arxiv.org/pdf/1505.04597.pdf):
        The network architecture is illustrated in Figure 1. It consists of a contracting
        path (left side) and an expansive path (right side). The contracting path follows
        the typical architecture of a convolutional network. It consists of the repeated
        application of two 3x3 convolutions (unpadded convolutions), each followed by
        a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
        for downsampling. At each downsampling step we double the number of feature
        channels. Every step in the expansive path consists of an upsampling of the
        feature map followed by a 2x2 convolution (“up-convolution”) that halves the
        number of feature channels, a concatenation with the correspondingly cropped
        feature map from the contracting path, and two 3x3 convolutions, each fol-
        lowed by a ReLU. The cropping is necessary due to the loss of border pixels in
        every convolution. At the final layer a 1x1 convolution is used to map each 64-
        component feature vector to the desired number of classes. In total the network
        has 23 convolutional layers.
        To allow a seamless tiling of the output segmentation map (see Figure 2), it
        is important to select the input tile size such that all 2x2 max-pooling operations
        are applied to a layer with an even x- and y-size.
        """
        x = features_last_to_second(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.max_pool(enc1))
        enc3 = self.encoder3(self.max_pool(enc2))
        enc4 = self.encoder4(self.max_pool(enc3))

        bottleneck = self.bottleneck(self.max_pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return features_second_to_last(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
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
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def transform_statics(self, statics: Statics) -> Statics:
        return statics

    def transform_batch(
        self, batch: Item
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return transform_batch_vision(batch)
