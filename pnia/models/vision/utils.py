"""
Shared utilities for vision models (convolutional neural networks and vision transformers).
"""
from typing import Tuple

import einops
import torch

from pnia.datasets.base import Item


def transform_batch_vision(
    batch: Item,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Take a batch as inputs.
    Return inputs, outputs, batch statics and forcing with the dimension appropriated for the vision model.
    """
    # To Do : rendre cela plus générique
    # Ici on suppose ce qu'on va trouver dans le batch.

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

    return inputs, outputs, forcing


def features_last_to_second(x: torch.Tensor) -> torch.Tensor:
    """
    Moves features from the last dimension to the second dimension.
    """
    return einops.rearrange(x, "b x y n -> b n x y").contiguous()


def features_second_to_last(y: torch.Tensor) -> torch.Tensor:
    """
    Moves features from the second dimension to the last dimension.
    """
    return einops.rearrange(y, "b n x y -> b x y n").contiguous()
