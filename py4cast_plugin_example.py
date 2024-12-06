"""
A simple plugin example for py4cast model with a Identity model.
"""

from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
from mfai.torch.models.base import ModelABC, ModelType
from torch import nn


@dataclass_json
@dataclass(slots=True)
class IdentitySettings:
    name: str = "Identity"


class Identity(ModelABC, nn.Module):
    settings_kls = IdentitySettings
    onnx_supported = False
    features_last: bool = True
    supported_num_spatial_dims = (2,)
    num_spatial_dims = 2
    model_type = ModelType.CONVOLUTIONAL

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        settings: IdentitySettings,
        input_shape: tuple,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_output_features = out_channels
        self.scaler = nn.Parameter(torch.rand(1))
        self._settings = settings
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.check_required_attributes()

    @property
    def settings(self):
        return self._settings

    def forward(self, x):
        """
        Keep only num_output_features along the last dimension.
        We multiply by a scaler param to avoid torch complaining about
        the uselesness of a model not requiring grad descent.
        """
        return x[..., : self.num_output_features] * self.scaler
