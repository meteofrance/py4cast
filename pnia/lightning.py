from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, Union

import einops
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torchinfo import summary

from pnia.datasets.base import DatasetInfo, ItemBatch, NamedTensor
from pnia.losses import ScaledL1Loss, ScaledRMSELoss, WeightedL1Loss, WeightedMSELoss
from pnia.models import build_model_from_settings, get_model_kls_and_settings
from pnia.models.base import expand_to_batch
from pnia.observer import PredictionPlot, SpatialErrorPlot, StateErrorPlot


@dataclass
class ArLightningHyperParam:
    """
    Settings and hyperparameters for the lightning AR model.
    """

    dataset_info: DatasetInfo
    batch_size: int

    model_conf: Union[Path, None] = None
    model_name: str = "halfunet"

    lr: float = 0.1
    loss: str = "mse"

    num_input_steps: int = 2
    num_pred_steps_train: int = 2
    num_inter_steps: int = 1  # Number of intermediary steps (without any data)

    num_pred_steps_val_test: int = 2
    num_samples_to_plot: int = 1

    training_strategy: str = "diff_ar"

    def __post_init__(self):
        """
        Check the configuration

        Raises:
            AttributeError: raise an exception if the set of attribute is not well designed.
        """
        if self.num_inter_steps > 1 and self.num_input_steps > 1:
            raise AttributeError(
                "It is not possible to have multiple input steps when num_inter_steps > 1."
                f"Get num_input_steps :{self.num_input_steps} and num_inter_steps: {self.num_inter_steps}"
            )
        ALLOWED_STRATEGIES = ("diff_ar", "scaled_ar")
        if self.training_strategy not in ALLOWED_STRATEGIES:
            raise AttributeError(
                f"Unknown strategy {self.training_strategy}, allowed strategies are {ALLOWED_STRATEGIES}"
            )

    def summary(self):
        self.dataset_info.summary()
        print(f"Number of input_steps : {self.num_input_steps}")
        print(f"Number of pred_steps : {self.num_pred_steps_train}")
        print(f"Number of intermediary steps :{self.num_inter_steps}")
        print(
            f"Model step duration : {self.dataset_info.step_duration /self.num_inter_steps}"
        )
        print(f"Model conf {self.model_conf}")
        print("---------------------")
        print(f"Loss {self.loss}")
        print(f"Batch size {self.batch_size}")
        print(f"Learning rate {self.lr}")
        print("---------------------------")


@rank_zero_only
def rank_zero_init(model_kls, model_settings, statics):
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(model_settings, statics)


@rank_zero_only
def exp_summary(hparams: ArLightningHyperParam, model: nn.Module):
    hparams.summary()
    summary(model)


class AutoRegressiveLightning(pl.LightningModule):
    """
    Auto-regressive lightning module for predicting meteorological fields.
    """

    def __init__(self, hparams: ArLightningHyperParam, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        # Load static features for grid/data
        # We do not want to change dataset statics inplace
        # Otherwise their is some problem with transform_statics and parameters_saving
        # when relaoding from checkpoint
        statics = deepcopy(hparams.dataset_info.statics)
        # Init object of register_dict
        self.diff_stats = hparams.dataset_info.diff_stats
        self.stats = hparams.dataset_info.stats

        # Keeping track of grid shape
        self.grid_shape = statics.grid_shape

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # Set model input/output grid features based on dataset tensor shapes
        num_grid_static_features = statics.grid_static_features.dim_size("features")

        # Compute the number of input features for the neural network
        # Should be directly supplied by datasetinfo ?

        num_input_features = (
            hparams.num_input_steps * hparams.dataset_info.weather_dim
            + num_grid_static_features
            + hparams.dataset_info.forcing_dim
        )

        num_output_features = hparams.dataset_info.weather_dim

        model_kls, model_settings = get_model_kls_and_settings(
            hparams.model_name, hparams.model_conf
        )

        rank_zero_init(model_kls, model_settings, statics)

        self.model, model_settings = build_model_from_settings(
            hparams.model_name,
            num_input_features,
            num_output_features,
            hparams.model_conf,
        )

        exp_summary(hparams, self.model)

        # We transform and register the statics after the model has been set up
        # This change the dimension of all statics
        statics = self.model.transform_statics(statics)
        # Register interior and border mask.
        statics.register_buffers(self)

        self.register_buffer(
            "grid_static_features",
            expand_to_batch(statics.grid_static_features.tensor, hparams.batch_size),
            persistent=False,
        )
        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if hparams.loss == "mse":
            self.loss = WeightedMSELoss(reduction="none")
        elif hparams.loss == "mae":
            self.loss = WeightedL1Loss(reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {hparams.loss}")

        # self.loss.prepare(statics)
        self.loss.prepare(statics.interior_mask, hparams.dataset_info)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["hparams"].lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt

    def _next_x(
        self, batch: ItemBatch, prev_states: torch.Tensor, step_idx: int
    ) -> torch.Tensor:
        """
        Build the next x input for the model at step_idx using the :
        - previous states
        - forcing
        - static features
        """
        forcing = batch.forcing[0].tensor[:, step_idx]
        x = torch.cat(
            [prev_states[:, idx] for idx in range(batch.num_input_steps)]
            + [self.grid_static_features[: batch.batch_size], forcing],
            dim=-1,
        )
        return x

    def _step_diffs(
        self, feature_names: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and std of the differences between two consecutive states on the desired device.
        """
        step_diff_std = self.diff_stats.to_list("std", feature_names).to(
            device, non_blocking=True
        )
        step_diff_mean = self.diff_stats.to_list("mean", feature_names).to(
            device, non_blocking=True
        )
        return step_diff_std, step_diff_mean

    def _strategy_params(self) -> Tuple[bool, bool, int]:
        """
        Return the parameters for the desired strategy:
        - force_border
        - scale_y
        - num_inter_steps
        """
        force_border: bool = (
            True if self.hparams["hparams"].training_strategy == "scaled_ar" else False
        )
        scale_y: bool = (
            True if self.hparams["hparams"].training_strategy == "scaled_ar" else False
        )
        # raise if mismatch between strategy and num_inter_steps
        if self.hparams["hparams"].training_strategy == "diff_ar":
            if self.hparams["hparams"].num_inter_steps != 1:
                raise ValueError(
                    "Diff AR strategy requires exactly 1 intermediary step."
                )

        return force_border, scale_y, self.hparams["hparams"].num_inter_steps

    def common_step(self, batch: ItemBatch) -> Tuple[NamedTensor, NamedTensor]:
        """
        Two Autoregressive strategies are implemented here:
        - scaled_ar:
            * Boundary forcing with y_true/true_state
            * Scaled Differential update next_state = prev_state + y * std + mean
            * Intermediary steps for which we have no y_true data

        - diff_ar:
            * No Boundary forcing
            * Differential update next_state = prev_state + y
            * No Intermediary steps

        Derived/Inspired from https://github.com/joeloskarsson/neural-lam/
        """
        force_border, scale_y, num_inter_steps = self._strategy_params()

        # Right now we postpone that we have a single input/output/forcing
        batch = self.model.transform_batch(batch)
        prev_states = batch.inputs[0].tensor

        prediction_list = []

        # Here we do the autoregressive prediction looping
        # for the desired number of ar steps.
        for i in range(batch.num_pred_steps):

            border_state = batch.outputs[0].tensor[:, i]

            if scale_y:
                step_diff_std, step_diff_mean = self._step_diffs(
                    batch.outputs[0].feature_names, prev_states.device
                )

            # Intermediary steps for which we have no y_true data
            # Should be greater or equal to 1 (otherwise nothing is done).
            for k in range(num_inter_steps):
                x = self._next_x(batch, prev_states, i)

                # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
                y = self.model(x)

                # We update the latest of our prev_states with the network output
                if scale_y:
                    predicted_state = (
                        prev_states[:, -1] + y * step_diff_std + step_diff_mean
                    )
                else:
                    predicted_state = prev_states[:, -1] + y

                # Overwrite border with true state
                # Force it to true state for all intermediary step
                if force_border:
                    new_state = (
                        self.border_mask * border_state
                        + self.interior_mask * predicted_state
                    )
                else:
                    new_state = predicted_state

                # Only update the prev_states if we are not at the last step
                if i < batch.num_pred_steps - 1 or k < num_inter_steps - 1:
                    # Update input states for next iteration: drop oldest, append new_state
                    prev_states = torch.cat(
                        [prev_states[:, 1:], new_state.unsqueeze(1)], dim=1
                    )
            # Append prediction to prediction list only "normal steps"
            prediction_list.append(new_state)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)
        pred_out = NamedTensor.new_like(prediction, batch.outputs[0])

        return pred_out, batch.outputs[0]

    def on_train_start(self):
        self.train_plotters = []

    def training_step(self, batch: ItemBatch) -> torch.Tensor:
        """
        Train on single batch
        """
        prediction, target = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(prediction, target)
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        # Notify every plotters
        for plotter in self.train_plotters:
            plotter.update(self, prediction=prediction, target=target)

        return batch_loss

    def on_validation_start(self):
        """
        Add some observers when starting validation
        """
        l1_loss = ScaledL1Loss(reduction="none")
        l1_loss.prepare(self.interior_mask, self.hparams["hparams"].dataset_info)
        metrics = {"mae": l1_loss}
        self.valid_plotters = [StateErrorPlot(metrics, kind="Validation")]

    def validation_step(self, batch: ItemBatch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(prediction, target), dim=0
        )  # (time_steps-1)

        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in range(time_step_loss.shape[0])
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(val_log_dict, on_step=False, on_epoch=True, sync_dist=True)

        # Notify every plotters
        for plotter in self.valid_plotters:
            plotter.update(self, prediction=prediction, target=target)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Notify every plotter that this is the end
        for plotter in self.valid_plotters:
            plotter.on_step_end(self)

    def on_test_start(self):
        """
        Attach observer when starting test
        """
        # this is very strange that we need to do that
        # Think at this statics problem (may be we just need to have a self.statics).

        l1_loss = ScaledL1Loss(reduction="none")
        rmse_loss = ScaledRMSELoss(reduction="none")
        l1_loss.prepare(self.interior_mask, self.hparams["hparams"].dataset_info)
        rmse_loss.prepare(self.interior_mask, self.hparams["hparams"].dataset_info)

        metrics = {"mae": l1_loss, "rmse": rmse_loss}
        # I do not like settings things outside the __init__
        self.test_plotters = [
            StateErrorPlot(metrics),
            SpatialErrorPlot(),
            PredictionPlot(self.hparams["hparams"].num_samples_to_plot),
        ]

    def test_step(self, batch: ItemBatch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target = self.common_step(batch)

        # Computing basic scores
        time_step_loss = torch.mean(
            self.loss(prediction, target), dim=0
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in range(time_step_loss.shape[0])
        }
        test_log_dict["test_mean_loss"] = mean_loss
        self.log_dict(test_log_dict, on_step=False, on_epoch=True, sync_dist=True)

        # Notify plotters
        for plotter in self.test_plotters:
            plotter.update(self, prediction=prediction, target=target)

    @cached_property
    def interior_2d(self) -> torch.Tensor:
        """
        Get the interior mask as a 2d mask.
        Usefull when stored as 1D in statics.
        """
        if self.model.info.output_dim == 1:
            return einops.rearrange(
                self.interior_mask, "(x y) h -> x y h", x=self.grid_shape[0]
            )
        return self.interior_mask

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        """
        # Notify plotters that the test epoch end
        for plotter in self.test_plotters:
            plotter.on_step_end(self)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, hparams):
        """
        Overwrite the load_from_checkpoint method in order to be able to read our parameters.
        """
        from io import BytesIO

        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
        # remove hyperparameters
        checkpoint["hyper_parameters"] = {}
        # Do no iniate loss state (as it has been changed from previous version)
        keys = list(checkpoint["state_dict"].keys())
        # Relmoving information for loss and statistics
        for key in keys:
            if "loss_" in key or "stats" in key:
                checkpoint["state_dict"].pop(key)
        buffer = BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        # Don't do strict mode. Indeed a lot of variable had changed (but are initiate in the __init__).
        return super().load_from_checkpoint(buffer, hparams=hparams, strict=False)
