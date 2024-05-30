"""
Abstract Base Class for all models
Contains also a few functionnality used in various model.
"""
from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.utils.checkpoint import checkpoint

from py4cast.datasets.base import Item, Statics


class ModelABC(ABC):
    """
    All models should inherit from this class.
    It provides the two methods to plug
    a nn.Module subclass with our Statics and Item classes.
    """

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
        by this model forward call.
        """

    @abstractproperty
    def onnx_supported(self) -> bool:
        """
        Indicates if our model supports onnx export.
        """


def offload_to_cpu(model: nn.ModuleList):
    return nn.ModuleList([offload_wrapper(x) for x in model])


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module. Comes from AIFS"""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no parameters and only
    buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def expand_to_batch(x: torch.Tensor, batch_size: int):
    """
    Expand tensor with initial batch dimension
    """
    # In order to be generic (for 1D or 2D grid)
    sizes = [batch_size] + [-1 for i in x.shape]
    return x.unsqueeze(0).expand(*sizes)
