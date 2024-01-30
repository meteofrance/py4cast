import numpy as np
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tueplots import bundles, figsizes

val_step_log_errors = np.array([1, 2, 3, 5, 10, 15, 19])


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


def make_mlp(blueprint, layer_norm=True, checkpoint=False):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    mlp = nn.Sequential(*layers)
    if checkpoint:
        mlp = CheckpointWrapper(mlp)
    return mlp


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of the page width.
    """
    bundle = bundles.neurips2023(usetex=True, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (original_figsize[0] / fraction, original_figsize[1])
    return bundle


def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in val_step_log_errors:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")
