"""
A simple plugin example for py4cast model with a Identity model.
"""


from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
from torch import nn

from py4cast.datasets.base import ItemBatch, Statics
from py4cast.models.base import ModelABC


@dataclass_json
@dataclass(slots=True)
class IdentitySettings:
    name: str = "Identity"


class Identity(ModelABC, nn.Module):
    settings_kls = IdentitySettings
    onnx_supported = True

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        settings: IdentitySettings,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_output_features = num_output_features
        self.scaler = nn.Parameter(torch.rand(1))

    def forward(self, x):
        """
        Keep only num_output_features along the last dimension.
        We multiply by a scaler param to avoid torch complaining about
        the uselesness of a model not requiring grad descent.
        """
        return x[..., : self.num_output_features] * self.scaler

    def transform_batch(self, batch: ItemBatch) -> ItemBatch:
        return batch

    def transform_statics(self, statics: Statics) -> Statics:
        return statics
