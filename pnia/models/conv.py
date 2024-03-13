"""
Convolutional neural network models
for pn-ia.
"""
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from typing import Tuple

import einops
import torch
from dataclasses_json import dataclass_json
from torch import nn

from pnia.datasets.base import Item, Statics
from pnia.models.utils import expand_to_batch


@dataclass_json
@dataclass(slots=True)
class ConvSettings:

    input_features: int
    output_features: int
    network_name: str = "HalfUnet"
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Sigmoid"


class ConvModel(nn.Module):
    registry = {}

    def __init__(
        self,
        settings: ConvSettings,
        statics: Statics,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer("step_diff_std", statics.step_diff_std)
        self.register_buffer("step_diff_mean", statics.step_diff_mean)

    def __init_subclass__(cls) -> None:
        """
        Register subclasses in registry
        if not a base class
        """
        name = cls.__name__.lower()

        if "base" not in name:
            cls.registry[name] = cls

    @classmethod
    def build_from_settings(
        cls, settings: ConvSettings, statics: Statics, *args, **kwargs
    ):
        """
        Create model from settings and dataset statics
        """
        return cls.registry[settings.network_name.lower()](
            settings, statics, *args, **kwargs
        )

    def transform_statics(self, statics: Statics) -> Statics:
        """
        Take the statics in inputs.
        Return the statics as expected by the model.
        """
        # Ici on ne devrait rien faire pour le convolutionnel.
        # Pour l'instant ça n'est pas le cas.
        statics.grid_static_features = einops.rearrange(
            statics.grid_static_features,
            "(x y) n -> x y n",
            x=self.grid_shape[0],
        )
        statics.border_mask = einops.rearrange(
            statics.border_mask,
            "(x y) n -> x y n",
            x=self.grid_shape[0],
        )
        statics.interior_mask = einops.rearrange(
            statics.interior_mask,
            "(x y) n -> x y n",
            x=self.grid_shape[0],
        )
        return statics

    def transform_batch(
        self, batch: Item
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Take a batch as inputs.
        Return inputs, outputs, batch statics and forcing with the dimension appropriated for the model.
        """
        # To Do : rendre cela plus générique
        # Ici on suppose ce qu'on va trouver dans le batch.

        statics = torch.cat(
            [x.values for x in batch.statics if x.ndims == 2], dim=-1
        ).unsqueeze(3)

        in2D = torch.cat([x.values for x in batch.inputs if x.ndims == 2], dim=-1)
        in3D = torch.cat([x.values for x in batch.inputs if x.ndims == 3], dim=-1)
        inputs = torch.cat([in3D, in2D], dim=-1)

        out2D = torch.cat([x.values for x in batch.outputs if x.ndims == 2], dim=-1)
        out3D = torch.cat([x.values for x in batch.outputs if x.ndims == 3], dim=-1)
        outputs = torch.cat([out3D, out2D], dim=-1)

        sh = outputs.shape
        f1 = torch.cat(
            [
                x.values.unsqueeze(2).unsqueeze(3).expand(-1, -1, sh[2], sh[3], -1)
                for x in batch.forcing
                if x.ndims == 1
            ],
            dim=-1,
        )
        # Ici on suppose qu'on a des forcage 2D et uniquement 2D (en plus du 1D).
        f2 = torch.cat([x.values for x in batch.forcing if x.ndims == 2], dim=-1)
        forcing = torch.cat([f1, f2], dim=-1)

        return inputs, outputs, statics, forcing

    def predict_step(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        batch_static_features: torch.Tensor,
        grid_static_features: torch.Tensor,
        forcing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a prediction step.
        First input needs to be reshaped
        from (B, N_grid, feature_dim) to (B, feature_dim, lon, lat)
        to accomodate the graph dataloader to conv2d expected shapes
        """
        batch_size = prev_state.shape[0]

        x = torch.cat(
            [
                prev_state,
                prev_prev_state,
                batch_static_features,
                expand_to_batch(grid_static_features, batch_size),
                forcing,
            ],
            dim=3,
        )
        # On met les features en seconde position
        x = einops.rearrange(x, "b x y n -> b n x y")
        # On met les features pour la sortie en derniere position.
        y = einops.rearrange(self(x), "b n x y -> b x y n")
        return prev_state + y * self.step_diff_std + self.step_diff_mean


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


class HalfUnet(ConvModel):
    def __init__(
        self,
        settings: ConvSettings,
        statics: Statics,
        *args,
        **kwargs,
    ):
        super().__init__(settings, statics, *args, **kwargs)
        self.grid_shape = statics.grid_shape

        self.encoder1 = self._block(
            settings.input_features,
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
            out_channels=settings.output_features,
            kernel_size=1,
            bias=settings.bias,
        )

        self.activation = getattr(nn, settings.last_activation)()

    def forward(self, x):
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

        return self.activation(self.outconv(dec))

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
