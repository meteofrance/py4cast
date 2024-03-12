import numpy as np
import torch.nn as nn

from pnia.models.utils import CheckpointWrapper

val_step_log_errors = np.array([1, 2, 3])


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


def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in val_step_log_errors:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")
