import getpass
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, Union

import einops
import matplotlib
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup

from py4cast.datasets.base import DatasetInfo, ItemBatch, NamedTensor
from py4cast.losses import ScaledLoss, WeightedLoss
from py4cast.metrics import MetricPSDK, MetricPSDVar
from py4cast.models import build_model_from_settings, get_model_kls_and_settings
from py4cast.models.base import expand_to_batch
from py4cast.observer import (
    PredictionEpochPlot,
    PredictionTimestepPlot,
    SpatialErrorPlot,
    StateErrorPlot,
)

# learning rate scheduling period in steps (update every nth step)
LR_SCHEDULER_PERIOD: int = 10

# PNG plots period in epochs. Plots are made, logged and saved every nth epoch.
PLOT_PERIOD: int = 10


@dataclass
class ArLightningHyperParam:
    """
    Settings and hyperparameters for the lightning AR model.
    """

    dataset_info: DatasetInfo
    dataset_name: str
    dataset_conf: Path
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

    len_train_loader: int = 1
    save_path: Path = None
    use_lr_scheduler: bool = False

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
        print(f"Number of pred_steps (training) : {self.num_pred_steps_train}")
        print(f"Number of pred_steps (test/val) : {self.num_pred_steps_val_test}")
        print(f"Number of intermediary steps :{self.num_inter_steps}")
        print(f"Training strategy :{self.training_strategy}")
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

        self.save_hyperparameters()  # write hparams.yaml in save folder

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

        # class variables to log loss during training, for tensorboad custom scalar
        self.training_step_losses = []
        self.validation_step_losses = []

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

        # All processes should wait until rank zero
        # has done the initialization (like creating a graph)
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
        if len(self.model.input_dims) == 3:
            # Graph model, we flatten the statics spatial dims
            statics.grid_static_features.flatten_("ngrid", 0, 1)
            statics.border_mask = statics.border_mask.flatten(0, 1)
            statics.interior_mask = statics.interior_mask.flatten(0, 1)

        # Register interior and border mask.
        statics.register_buffers(self)

        self.num_spatial_dims = statics.grid_static_features.num_spatial_dims

        self.register_buffer(
            "grid_static_features",
            expand_to_batch(statics.grid_static_features.tensor, hparams.batch_size),
            persistent=False,
        )
        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if hparams.loss == "mse":
            self.loss = WeightedLoss("MSELoss", reduction="none")
        elif hparams.loss == "mae":
            self.loss = WeightedLoss("L1Loss", reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {hparams.loss}")

        self.loss.prepare(self, statics.interior_mask, hparams.dataset_info)

        # Metrics
        self.rmse_psd_plot_metric = MetricPSDVar()
        save_path = self.hparams["hparams"].save_path
        self.psd_plot_metric = MetricPSDK(save_path)

    @rank_zero_only
    def log_hparams_tb(self):
        if self.logger:
            hparams = self.hparams["hparams"]
            # Log hparams in tensorboard hparams window
            dict_log = asdict(hparams)
            dict_log["username"] = getpass.getuser()
            self.logger.log_hyperparams(dict_log, metrics={"val_mean_loss": 0.0})
            # Save model & dataset conf as files
            if hparams.dataset_conf is not None:
                shutil.copyfile(
                    hparams.dataset_conf, hparams.save_path / "dataset_conf.json"
                )
            if hparams.model_conf is not None:
                shutil.copyfile(
                    hparams.model_conf, hparams.save_path / "model_conf.json"
                )

    def on_fit_start(self):
        self.log_hparams_tb()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        lr = self.hparams["hparams"].lr
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.95))
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        if self.hparams["hparams"].use_lr_scheduler:

            len_loader = self.hparams["hparams"].len_train_loader // LR_SCHEDULER_PERIOD
            epochs = self.trainer.max_epochs
            lr_scheduler = get_cosine_schedule_with_warmup(
                opt, 1000 // LR_SCHEDULER_PERIOD, len_loader * epochs
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
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
        forcing = batch.forcing.tensor[:, step_idx]
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

        self.original_shape = None

        if len(self.model.input_dims) == 3:
            # Stack original shape to reshape later
            self.original_shape = batch.inputs.tensor.shape
            # Graph model, we flatten the batch spatial dims
            batch.inputs.flatten_("ngrid", 2, 3)
            batch.outputs.flatten_("ngrid", 2, 3)
            batch.forcing.flatten_("ngrid", 2, 3)

        prev_states = batch.inputs.tensor
        prediction_list = []

        # Here we do the autoregressive prediction looping
        # for the desired number of ar steps.

        for i in range(batch.num_pred_steps):

            border_state = batch.outputs.tensor[:, i]

            if scale_y:
                step_diff_std, step_diff_mean = self._step_diffs(
                    batch.outputs.feature_names, prev_states.device
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
        pred_out = NamedTensor.new_like(prediction, batch.outputs)

        return pred_out, batch.outputs

    def on_train_start(self):
        self.train_plotters = []

    def _shared_epoch_end(self, outputs: List[torch.Tensor], label: str) -> None:
        """Computes and logs the averaged metrics at the end of an epoch.
        Step shared by training and validation epochs.
        """
        avg_loss = torch.stack([x for x in outputs]).mean()
        tb = self.logger.experiment
        tb.add_scalar(f"mean_loss_epoch/{label}", avg_loss, self.current_epoch)

    def training_step(self, batch: ItemBatch) -> torch.Tensor:
        """
        Train on single batch
        """
        prediction, target = self.common_step(batch)
        # Compute loss: mean over unrolled times and batch
        batch_loss = torch.mean(self.loss(prediction, target))

        self.training_step_losses.append(batch_loss)

        # Notify every plotters
        for plotter in self.train_plotters:
            plotter.update(self, prediction=self.prediction, target=self.target)

        return batch_loss

    def on_train_epoch_end(self):
        outputs = self.training_step_losses
        self._shared_epoch_end(outputs, "train")
        self.training_step_losses.clear()  # free memory

    def on_validation_start(self):
        """
        Add some observers when starting validation
        """
        l1_loss = ScaledLoss("L1Loss", reduction="none")
        l1_loss.prepare(self, self.interior_mask, self.hparams["hparams"].dataset_info)
        metrics = {"mae": l1_loss}
        save_path = self.hparams["hparams"].save_path
        self.valid_plotters = [
            StateErrorPlot(metrics, prefix="Validation"),
            PredictionTimestepPlot(
                num_samples_to_plot=1,
                num_features_to_plot=4,
                prefix="Validation",
                save_path=save_path,
            ),
            PredictionEpochPlot(
                num_samples_to_plot=1,
                num_features_to_plot=4,
                prefix="Validation",
                save_path=save_path,
            ),
        ]

    def _shared_val_test_step(self, batch: ItemBatch, batch_idx, label: str):
        with torch.no_grad():
            prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(self.loss(prediction, target), dim=0)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per timestep
        loss_dict = {
            f"timestep_losses/{label}_step_{step}": time_step_loss[step]
            for step in range(time_step_loss.shape[0])
        }
        self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
        self.log(
            f"{label}_mean_loss",
            mean_loss,
            on_epoch=True,
            sync_dist=True,
            prog_bar=(label == "val"),
        )
        return prediction, target, mean_loss

    def validation_step(self, batch: ItemBatch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, mean_loss = self._shared_val_test_step(
            batch, batch_idx, "val"
        )
        self.validation_step_losses.append(mean_loss)

        self.val_mean_loss = mean_loss

        # Notify every plotters
        if self.current_epoch % PLOT_PERIOD == 0:
            for plotter in self.valid_plotters:
                plotter.update(self, prediction=prediction, target=target)
            self.psd_plot_metric.update(prediction, target, self.original_shape)
            self.rmse_psd_plot_metric.update(prediction, target, self.original_shape)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """

        # Get dict of metrics' results
        dict_metrics = dict()
        dict_metrics.update(self.psd_plot_metric.compute())
        dict_metrics.update(self.rmse_psd_plot_metric.compute())

        for name, elmnt in dict_metrics.items():
            if isinstance(elmnt, matplotlib.figure.Figure):
                self.logger.experiment.add_figure(f"{name}", elmnt, self.current_epoch)
            elif isinstance(elmnt, torch.Tensor):
                self.log_dict(
                    {name: elmnt},
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        outputs = self.validation_step_losses
        self._shared_epoch_end(outputs, "validation")

        # free memory
        self.validation_step_losses.clear()

        # Notify every plotters
        if self.current_epoch % PLOT_PERIOD == 0:
            for plotter in self.valid_plotters:
                plotter.on_step_end(self)

    def on_test_start(self):
        """
        Attach observer when starting test
        """
        metrics = {}
        for torch_loss, alias in ("L1Loss", "mae"), ("MSELoss", "rmse"):
            loss = ScaledLoss(torch_loss, reduction="none")
            loss.prepare(self, self.interior_mask, self.hparams["hparams"].dataset_info)
            metrics[alias] = loss

        save_path = self.hparams["hparams"].save_path
        self.test_plotters = [
            StateErrorPlot(metrics),
            SpatialErrorPlot(),
            PredictionTimestepPlot(self.hparams["hparams"].num_samples_to_plot),
            PredictionTimestepPlot(
                num_samples_to_plot=self.hparams["hparams"].num_samples_to_plot,
                num_features_to_plot=4,
                prefix="Test",
                save_path=save_path,
            ),
        ]

    def test_step(self, batch: ItemBatch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, _ = self._shared_val_test_step(batch, batch_idx, "test")

        # Notify plotters & metrics
        for plotter in self.test_plotters:
            plotter.update(self, prediction=prediction, target=target)
            self.psd_plot_metric.update(prediction, target, self.original_shape)
            self.rmse_psd_plot_metric.update(prediction, target, self.original_shape)

    @cached_property
    def interior_2d(self) -> torch.Tensor:
        """
        Get the interior mask as a 2d mask.
        Usefull when stored as 1D in statics.
        """
        if self.num_spatial_dims == 1:
            return einops.rearrange(
                self.interior_mask, "(x y) h -> x y h", x=self.grid_shape[0]
            )
        return self.interior_mask

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        """
        self.psd_plot_metric.compute()
        self.rmse_psd_plot_metric.compute()

        # Notify plotters that the test epoch end
        for plotter in self.test_plotters:
            plotter.on_step_end(self)
