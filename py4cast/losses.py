"""
This module contains the loss functions used in the training of the models.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple

import lightning.pytorch as pl
import torch
from mfai.pytorch.losses.perceptual import PerceptualLoss
from torch.nn import MSELoss

from py4cast.datasets.base import DatasetInfo, NamedTensor


class Py4CastLoss(ABC):
    """
    Abstract class to force the user to implement the prepare and forward method because
    They are expected by the rest of the system.
    See https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
    """

    def __init__(self, loss: str, *args, **kwargs) -> None:
        if hasattr(torch.nn, loss):
            self.loss = getattr(torch.nn, loss)(*args, **kwargs)
        elif loss in globals():
            self.loss = globals()[loss](*args, **kwargs)
        else:
            raise NameError(f"Loss: {loss} is not defined")

    @abstractmethod
    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        """
        Prepare the loss function using the dataset informations and the interior mask
        """

    @abstractmethod
    def forward(
        self, prediction: NamedTensor, target: NamedTensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function
        """

    def register_loss_state_buffers(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        loss_state_weight: dict,
        squeeze_mask: bool = False,
    ) -> None:
        """
        We register the state_weight buffer to the lightning module
        and keep references to other buffers of interest
        """

        self.loss_state_weight = loss_state_weight
        attr_name = "interior_mask_s" if squeeze_mask else "interior_mask"
        if not hasattr(lm, attr_name):
            lm.register_buffer(
                attr_name,
                interior_mask.squeeze(-1) if squeeze_mask else interior_mask,
                persistent=False,
            )
        self.num_interior = torch.sum(interior_mask).item()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    @lru_cache(maxsize=8)
    def weights(self, feature_names: Tuple[str], device: torch.device) -> torch.Tensor:
        """
        We build and cache the weights tensor for the given loss, feature names and device.
        """
        return torch.stack([self.loss_state_weight[name] for name in feature_names]).to(
            device, non_blocking=True
        )


def min_max_normalization(x: NamedTensor, lm: pl.LightningModule) -> torch.tensor:
    """
    Apply a min max normalization to the tensor x.
    Where x is of the shape (B, T, H, W, F)
    """
    min_list = lm.stats.to_list("min", x.feature_names).to(x.tensor, non_blocking=True)
    max_list = lm.stats.to_list("max", x.feature_names).to(x.tensor, non_blocking=True)
    mean_list = lm.stats.to_list("mean", x.feature_names).to(
        x.tensor, non_blocking=True
    )
    std_list = lm.stats.to_list("std", x.feature_names).to(x.tensor, non_blocking=True)
    min_list = (min_list - mean_list) / std_list
    max_list = (max_list - mean_list) / std_list
    return (x.tensor - min_list) / (max_list - min_list + 1e-8)


class WeightedLoss(Py4CastLoss):
    """
    Compute a weighted loss function with a weight for each feature.
    During the forward step, the loss is computed for each feature and then weighted
    and optionally averaged over the spatial dimensions.
    """

    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}

        exponent = 2.0 if self.loss.__class__ == MSELoss else 1.0

        for name in dataset_info.state_weights:
            loss_state_weight[name] = dataset_info.state_weights[name] / (
                dataset_info.diff_stats[name]["std"] ** exponent
            )
        self.register_loss_state_buffers(
            lm, interior_mask, loss_state_weight, squeeze_mask=True
        )
        self.lm = lm

    def forward(
        self,
        prediction: NamedTensor,
        target: NamedTensor,
        mask: torch.Tensor,
        reduce_spatial_dim: bool = True,
    ) -> torch.Tensor:
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        returns (B, pred_steps)
        """

        # Compute Torch loss (defined in the parent class when this Mixin is used)
        torch_loss = self.loss(prediction.tensor * mask, target.tensor * mask)

        # Retrieve the weights for each feature
        weights = self.weights(tuple(prediction.feature_names), prediction.device)

        # Apply the weights and sum over the feature dimension
        weighted_loss = torch.sum(torch_loss * weights, dim=-1)

        # if no reduction on spatial dimension is required, return the weighted loss
        if not reduce_spatial_dim:
            return weighted_loss

        union_mask = torch.any(mask, dim=(0, 1, 4))

        # Compute the mean loss over all spatial dimensions
        # Take (unweighted) mean over only non-border (interior) grid nodes/pixels
        # We use forward indexing for the spatial_dim_idx of the target tensor
        # so the code below works even if the feature dimension has been reduced
        # The final shape is (B, pred_steps)

        time_step_mean_loss = torch.sum(
            weighted_loss * self.lm.interior_mask_s,
            dim=target.spatial_dim_idx,
        ) / (self.num_interior - (~union_mask).sum())

        return time_step_mean_loss


class ScaledLoss(Py4CastLoss):
    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            loss_state_weight[name] = dataset_info.stats[name]["std"]
        self.register_loss_state_buffers(lm, interior_mask, loss_state_weight)
        self.lm = lm

    def forward(
        self, prediction: NamedTensor, target: NamedTensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computed weighted loss function averaged over all spatial dimensions.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        returns (B, pred_steps)
        """
        # Compute Torch loss (defined in the parent class when this Mixin is used)
        torch_loss = self.loss(prediction.tensor * mask, target.tensor * mask)

        union_mask = torch.any(mask, dim=(0, 1, 4))

        # Compute the mean loss value over spatial dimensions
        mean_loss = torch.sum(
            torch_loss * self.lm.interior_mask,
            dim=target.spatial_dim_idx,
        ) / (self.num_interior - (~union_mask).sum())

        if self.loss.__class__ == MSELoss:
            mean_loss = torch.sqrt(mean_loss)

        return mean_loss * self.weights(
            tuple(prediction.feature_names), prediction.device
        )


class PerceptualLossPy4Cast(Py4CastLoss):
    """
    Compute a perceptual loss.
    During the forward step, the perceptual loss is computed for all the feature in the same time.
    """

    def __init__(self, in_channels: int, *args, **kwargs) -> None:
        self.perceptual_loss = PerceptualLoss(
            in_channels=in_channels,
            device="cpu",  # We don't know the device at instantiation
            **kwargs,
        )

    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        self.lm = lm

    def forward(
        self, prediction: NamedTensor, target: NamedTensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a perceptual loss function over all the features.

        Args:
            prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)

        Returns:
            Tensor (B, pred_steps)
        """
        # Normalize between 0 and 1
        pred_tensor = min_max_normalization(prediction, self.lm) * mask
        target_tensor = min_max_normalization(target, self.lm) * mask

        # Clamp between 0 and 1
        pred_tensor = pred_tensor.clamp(0, 1)
        target_tensor = target_tensor.clamp(0, 1)

        # Compute the perceptual loss at each timestep
        shape_pred = pred_tensor.shape
        perc_loss = torch.zeros(shape_pred[1], device=pred_tensor.device)
        for t in range(shape_pred[1]):
            # feature in second dimension
            pred_tensor_t = pred_tensor[:, t].permute(0, 3, 1, 2)
            target_tensor_t = target_tensor[:, t].permute(0, 3, 1, 2)
            # Compute Torch loss
            perc_loss[t] = self.perceptual_loss(pred_tensor_t, target_tensor_t)

        return perc_loss.unsqueeze(0)


class CombinedLoss(Py4CastLoss):
    """
    Compute a combinaison of Py4castLoss.
    """

    def __init__(self, losses_config: list[dict]):
        self.losses = []
        for loss_conf in losses_config:
            LossClass = globals()[loss_conf["class"]]
            weight = loss_conf.get("weight", 1.0)
            kwargs = loss_conf.get("params", {})
            self.losses.append((LossClass(**kwargs), weight))

    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ):
        for loss, _ in self.losses:
            if hasattr(loss, "prepare"):
                loss.prepare(lm, interior_mask, dataset_info)

    def forward(
        self, prediction: NamedTensor, target: NamedTensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Computes each loss function and sums them.

        Args:
            prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)

        Returns:
                returns (B, pred_steps)
        """
        # shape (B, pred_step)
        loss_shape = (
            prediction.tensor.shape[:2]
            if kwargs.get("reduce_spatial_dim", True)
            else prediction.tensor.shape[:4]
        )
        total_loss = torch.zeros(loss_shape, device=prediction.tensor.device)
        for loss, weight in self.losses:
            total_loss += weight * loss(prediction, target, mask, **kwargs)
        return total_loss
