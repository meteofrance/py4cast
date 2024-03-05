from abc import ABC, abstractmethod

import torch
from torch.nn import L1Loss, MSELoss

from pnia.datasets.base import Statics


class WeightedLossMixin:
    def register_loss_state_buffers(
        self, statics: Statics, state_weight: torch.Tensor
    ) -> None:
        """
        We register the state_weight buffer to the lightning module
        and keep references to other buffers of interest
        """

        self.register_buffer("state_weight", state_weight)
        self.register_buffer("interior_mask", statics.interior_mask)
        self.N_interior = statics.N_interior

    def forward(self, prediction, target, reduce_spatial_dim=True):
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f)
        returns (B, pred_steps)
        """
        entry_loss = super().forward(prediction, target)  # (B, pred_steps, N_grid, d_f)
        grid_node_loss = torch.sum(
            entry_loss * self.state_weight, dim=-1
        )  # (B, pred_steps, N_grid), weighted sum over features
        if not reduce_spatial_dim:
            return grid_node_loss  # (B, pred_steps, N_grid)

        # Take (unweighted) mean over only non-border (interior) grid nodes
        time_step_loss = (
            torch.sum(grid_node_loss * self.interior_mask[:, 0], dim=-1)
            / self.N_interior
        )  # (B, pred_steps)

        return time_step_loss  # (B, pred_steps)


class PniaLoss(ABC):
    @abstractmethod
    def prepare(self, statics: Statics) -> None:
        """
        Prepare the loss function using the statics from the dataset
        """


class WeightedMSELoss(WeightedLossMixin, MSELoss, PniaLoss):
    def prepare(self, statics: Statics) -> None:
        inv_var = (
            statics.step_diff_std**-2.0
        )  # Comes from buffer registration of statistics
        loss_state_weights = statics.param_weights * inv_var  # (d_f,)
        super().register_loss_state_buffers(statics, loss_state_weights)


class WeightedL1Loss(WeightedLossMixin, L1Loss, PniaLoss):
    def prepare(self, statics: Statics) -> None:

        # Weight states with inverse std instead in this case
        state_weights = statics.param_weights / statics.step_diff_std  # (d_f,)
        super().register_loss_state_buffers(statics, state_weights)
