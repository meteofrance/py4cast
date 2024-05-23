"""
Shared utilities for vision models (convolutional neural networks and vision transformers).
"""

import einops
import torch


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
