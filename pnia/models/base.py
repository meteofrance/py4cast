"""
Abstract Base Class for all models
+ model registry
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch

from pnia.datasets.base import Item, Statics


class ModelBase(ABC):
    """
    All models should inherit from this class.
    """

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    @abstractmethod
    def transform_statics(self, statics: Statics) -> Statics:
        """
        Transform the dataset static features according to this model
        expected input shapes.
        """

    @abstractmethod
    def transform_batch(
        self, batch: Item
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Transform the batch into a set of tensors as expected
        by the model.
        """


@dataclass(slots=True)
class ModelInfo:
    """
    Information specific to a model
    """

    output_dim: int  # Spatial dimension of the output
