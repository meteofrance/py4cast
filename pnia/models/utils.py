import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tueplots import bundles, figsizes


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


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of the page width.
    """
    bundle = bundles.neurips2023(usetex=True, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (original_figsize[0] / fraction, original_figsize[1])
    return bundle


def expand_to_batch(x: torch.Tensor, batch_size: int):
    """
    Expand tensor with initial batch dimension
    """
    # In order to be generic (for 1D or 2D grid)
    sizes = [batch_size] + [-1 for i in x.shape]
    return x.unsqueeze(0).expand(*sizes)
