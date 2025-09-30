"""
This module contains the loss functions used in the training of the models.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple

import lightning.pytorch as pl
import torch
from torch.nn import MSELoss

from mfai.pytorch.losses.perceptual import PerceptualLoss
from py4cast.datasets.base import DatasetInfo, NamedTensor


class Py4CastLoss(ABC):
    """
    Abstract class to force the user to implement the prepare and forward method because
    They are expected by the rest of the system.
    See https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
    """

    def __init__(self, loss: str, *args, **kwargs) -> None:
        self.loss = getattr(torch.nn, loss)(*args, **kwargs)

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

def min_max_normalization(x: torch.tensor)-> torch.tensor:
    """
    Apply a min max normalization to the tensor x.
    Where x is of the shape (B, T, H, W, F)
    """
    x_min = x.amin(dim=(2,3,4), keepdim=True)
    x_max = x.amax(dim=(2,3,4), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)

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
        reduce_spatial_dim=True,
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

class CombinedPerceptualLoss(WeightedLoss):
    """
    Compute a combinaison between perceptual loss and weightedloss.
    During the forward step, the perceptual loss is computed for all the feature in the same time.
    """
    def __init__(self, num_output_features: int, alpha: float=0.5, device: str="cpu", *args, **kwargs) -> None:
        super().__init__(loss="MSELoss", *args, **kwargs)
        self.alpha = alpha
        self.perceptual_loss = PerceptualLoss(
            multi_scale = True,
            resize_input = True,
            pre_trained = True,
            channel_iterative_mode = False,
            in_channels = num_output_features,
            device= device
        )

    def forward(self, prediction: NamedTensor, target: NamedTensor, mask: torch.Tensor):
        """
        Computed a combinaison between a perceptual loss function over all the feature and batch and mseweightedloss.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        returns (B, pred_steps)
        """
        print("pred: ", prediction.tensor.is_nan().any())
        print("target: ", target.tensor.is_nan().any())
        torch.save(prediction.tensor, "pred_loss.pt")
        torch.save(target.tensor, "target_loss.pt")
        # MSE Weighted loss (B, pred_step)
        mse_weighted = super().forward(prediction, target, mask)

        # Compute perceptual loss
        # Normalize between 0 and 1
        pred_tensor = min_max_normalization(prediction.tensor * mask)
        target_tensor = min_max_normalization(target.tensor * mask)

        shape_pred = pred_tensor.shape

        # The loss have the shape (pred_steps)
        perc_loss = torch.zeros(shape_pred[1], device=pred_tensor.device)

        for t in range(shape_pred[1]):
            pred_tensor_t = pred_tensor[:,t].permute(0, 3, 1, 2)
            target_tensor_t = target_tensor[:,t].permute(0, 3, 1, 2)
            # Compute Torch loss
            perc_loss[t] = self.perceptual_loss(pred_tensor_t, target_tensor_t)

        print("mse: ", mse_weighted)
        print("perceptual: ", perc_loss)
        # Combine losses
        return (1 - self.alpha) * mse_weighted + self.alpha * perc_loss.unsqueeze(0)