from abc import ABC, abstractmethod

import torch
from torch.nn import L1Loss, MSELoss

from py4cast.datasets.base import DatasetInfo, NamedTensor


class WeightedLossMixin:
    def register_loss_state_buffers(
        self, interior_mask: torch.Tensor, loss_state_weight: dict
    ) -> None:
        """
        We register the interior mask to the lightning module.
        loss_state_weight is no longer registered
        and keep references to other buffers of interest
        """
        self.loss_state_weight = loss_state_weight
        self.register_buffer("interior_mask", interior_mask.squeeze(-1))
        self.num_interior = torch.sum(interior_mask).item()
        # Store the aggregate onrmse which one should aggregate.
        # This dimension is grid dependent (not the same for 1D and 2D problems)
        dims = len(self.interior_mask.shape)
        self.aggregate_dims = tuple([-(x + 1) for x in range(0, dims)])

    def forward(
        self,
        prediction: NamedTensor,
        target: NamedTensor,
        reduce_spatial_dim=True,
    ) -> torch.Tensor:
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f)
        returns (B, pred_steps)
        """
        entry_loss = super().forward(
            prediction.tensor, target.tensor
        )  # (B, pred_steps, N_grid, d_f)
        weight = torch.stack(
            [self.loss_state_weight[name] for name in prediction.feature_names]
        ).to(entry_loss, non_blocking=True)
        grid_node_loss = torch.sum(
            entry_loss * weight, dim=-1
        )  # (B, pred_steps, N_grid), weighted sum over features
        if not reduce_spatial_dim:
            return grid_node_loss  # (B, pred_steps, N_grid)
        # Take (unweighted) mean over only non-border (interior) grid nodes
        time_step_loss = (
            torch.sum(grid_node_loss * self.interior_mask, dim=self.aggregate_dims)
            / self.num_interior
        )  # (B, pred_steps)
        return time_step_loss  # (B, pred_steps)


class RegisterSpatialMixin:
    def register_loss_state_buffers(
        self, interior_mask: torch.Tensor, loss_state_weight: dict
    ) -> None:
        """
        We register the state_weight buffer to the lightning module
        and keep references to other buffers of interest
        """

        self.loss_state_weight = loss_state_weight
        self.register_buffer("interior_mask", interior_mask)
        self.num_interior = torch.sum(interior_mask).item()
        # Store the aggregate onrmse which one should aggregate.
        # This dimension is grid dependent (not the same for 1D and 2D problems)
        # As we do not squeeze (in order to be able to multiply) the dimension is changed
        dims = len(self.interior_mask.shape) - 1
        self.aggregate_dims = tuple(
            [-(x + 2) for x in range(0, dims)]
        )  # Not the same as WeightedLossMixin


class SpatialLossMixin:
    def forward(self, prediction: NamedTensor, target: NamedTensor) -> torch.Tensor:
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f)
        returns (B, pred_steps)
        """
        entry_loss = super().forward(
            prediction.tensor, target.tensor
        )  # (B, pred_steps, N_grid, d_f)
        # Je ne comprend pas pourquoi j'ai besoin ici de le faire.
        weight = torch.stack(
            [self.loss_state_weight[name] for name in prediction.feature_names]
        ).to(entry_loss, non_blocking=True)
        entry_loss = entry_loss * weight
        mean_error = (
            torch.sum(
                entry_loss * self.interior_mask.to(entry_loss, non_blocking=True),
                dim=self.aggregate_dims,
            )
            / self.num_interior
        )
        return mean_error


class Py4castLoss(ABC):
    @abstractmethod
    def prepare(self, interior_mask: torch.Tensor, dataset_info: DatasetInfo) -> None:
        """
        Prepare the loss function using the statics from the dataset
        """


class ScaledRMSELoss(
    RegisterSpatialMixin, MSELoss, Py4castLoss
):  # We do not want to call the Mixin Forward.
    def prepare(self, interior_mask: torch.Tensor, dataset_info: DatasetInfo) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            loss_state_weight[name] = dataset_info.stats[name]["std"]
        super().register_loss_state_buffers(interior_mask, loss_state_weight)

    def forward(self, prediction: NamedTensor, target: NamedTensor):
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f)
        returns (B, pred_steps)
        """
        entry_loss = super().forward(
            prediction.tensor, target.tensor
        )  # (B, pred_steps, N_grid, d_f)
        entry_loss = entry_loss * self.interior_mask.to(entry_loss, non_blocking=True)

        mean_error = torch.sum(entry_loss, dim=self.aggregate_dims) / self.num_interior
        weight = torch.stack(
            [self.loss_state_weight[name] for name in prediction.feature_names]
        ).to(entry_loss, non_blocking=True)
        return torch.sqrt(mean_error) * weight


class ScaledL1Loss(RegisterSpatialMixin, SpatialLossMixin, L1Loss, Py4castLoss):
    def prepare(self, interior_mask: torch.Tensor, dataset_info: DatasetInfo) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            loss_state_weight[name] = dataset_info.stats[name]["std"]
        super().register_loss_state_buffers(interior_mask, loss_state_weight)

    # def prepare(self, statics: Statics) -> None:
    #    super().register_loss_state_buffers(statics, statics.data_std)


class WeightedMSELoss(WeightedLossMixin, MSELoss, Py4castLoss):
    def prepare(self, interior_mask: torch.Tensor, dataset_info: DatasetInfo) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            loss_state_weight[name] = dataset_info.state_weights[name] / (
                dataset_info.diff_stats[name]["std"] ** 2.0
            )
        super().register_loss_state_buffers(interior_mask, loss_state_weight)


class WeightedL1Loss(WeightedLossMixin, L1Loss, Py4castLoss):
    def prepare(self, interior_mask: torch.Tensor, dataset_info: DatasetInfo) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            loss_state_weight[name] = (
                dataset_info.state_weights[name] / dataset_info.diff_stats[name]["std"]
            )
        super().register_loss_state_buffers(interior_mask, loss_state_weight)
