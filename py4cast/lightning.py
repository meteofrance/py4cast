import json
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import einops
import matplotlib
import mlflow.pytorch
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
from mfai.torch.models.base import ModelType
from mfai.torch.models.utils import (
    expand_to_batch,
    features_last_to_second,
    features_second_to_last,
)
from mlflow.models.signature import infer_signature
from torchinfo import summary

from py4cast.datasets import get_datasets
from py4cast.datasets.base import DatasetInfo, ItemBatch, NamedTensor, Statics
from py4cast.io.outputs import GribSavingSettings, save_named_tensors_to_grib
from py4cast.losses import ScaledLoss, WeightedLoss
from py4cast.metrics import MetricACC, MetricPSDK, MetricPSDVar
from py4cast.models import build_model_from_settings, get_model_kls_and_settings
from py4cast.models import registry as model_registry
from py4cast.plots import (
    PredictionEpochPlot,
    PredictionTimestepPlot,
    SpatialErrorPlot,
    StateErrorPlot,
)
from py4cast.utils import str_to_dtype

PLOT_PERIOD: int = 10


@dataclass
class PlDataModule(LightningDataModule):
    """
    DataModule to encapsulate data splits and data loading.
    """

    def __init__(
        self,
        dataset_name: str = "dummy",
        num_input_steps: int = 1,
        num_pred_steps_train: int = 1,
        num_pred_steps_val_test: int = 1,
        batch_size: int = 2,
        num_workers: int = 1,
        prefetch_factor: int | None = None,
        pin_memory: bool = False,
        dataset_conf: Path | None = None,
        predict_conf: Dict | None = None,
    ):
        super().__init__()
        self.num_input_steps = num_input_steps
        self.num_pred_steps_train = num_pred_steps_train
        self.num_pred_steps_val_test = num_pred_steps_val_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        # Get dataset in initialisation to have access to this attribute before method trainer.fit
        self.train_ds, self.val_ds, self.test_ds = get_datasets(
            dataset_name,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
            dataset_conf,
            predict_conf,
        )

    @property
    def train_dataset_info(self) -> DatasetInfo:
        return self.train_ds.dataset_info

    @property
    def infer_ds(self):
        return self.test_ds

    def train_dataloader(self):
        return self.train_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return self.val_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return self.test_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return self.test_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )


@rank_zero_only
def rank_zero_init(model_kls, model_settings, statics: Statics):
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(model_settings, statics.meshgrid)


class AutoRegressiveLightning(LightningModule):
    """
    Auto-regressive lightning module for predicting meteorological fields.
    """

    def __init__(
        self,
        settings_init_args: dict,
        # args linked from trainer and datamodule
        dataset_info,  # Don't put type hint here or CLI doesn't work
        infer_ds,
        dataset_name: str = "dummy",
        dataset_conf: Path | None = None,  # TO DELETE ?
        num_input_steps: int = 1,
        num_pred_steps_train: int = 1,
        num_pred_steps_val_test: int = 1,
        batch_size: int = 2,
        # non-linked args
        model_name: Literal[tuple(model_registry.keys())] = "HalfUNet",
        model_conf: Path | None = None,
        loss_name: Literal["mse", "mae"] = "mse",
        num_inter_steps: int = 1,
        num_samples_to_plot: int = 1,
        training_strategy: Literal["diff_ar", "scaled_ar", "downscaling_only"] = "diff_ar",
        channels_last: bool = False,
        io_conf: Path | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.infer_ds = infer_ds
        self.settings_init_args = settings_init_args
        self.dataset_name = dataset_name
        self.dataset_conf = dataset_conf
        self.dataset_info = dataset_info
        self.batch_size = batch_size
        self.model_conf = model_conf
        self.model_name = model_name
        self.num_input_steps = num_input_steps
        self.num_pred_steps_train = num_pred_steps_train
        self.num_pred_steps_val_test = num_pred_steps_val_test
        self.num_inter_steps = num_inter_steps
        self.num_samples_to_plot = num_samples_to_plot
        self.training_strategy = training_strategy
        self.channels_last = channels_last
        self.io_conf = io_conf

        
        if self.training_strategy == "downscaling_only":
            print("WARNING : You are using downscaling_only mode: this is experimental.")

        if self.num_inter_steps > 1 and self.num_input_steps > 1:
            raise AttributeError(
                "It is not possible to have multiple input steps when num_inter_steps > 1."
                f"Get num_input_steps :{self.num_input_steps} and num_inter_steps: {self.num_inter_steps}"
            )
        ALLOWED_STRATEGIES = ("diff_ar", "scaled_ar", "downscaling_only")
        if self.training_strategy not in ALLOWED_STRATEGIES:
            raise AttributeError(
                f"Unknown strategy {self.training_strategy}, allowed strategies are {ALLOWED_STRATEGIES}"
            )

        self.save_hyperparameters()  # write hparams.yaml in save folder

        # Load static features for grid/data
        # We do not want to change dataset statics inplace
        # Otherwise their is some problem with transform_statics and parameters_saving
        # when relaoding from checkpoint
        statics = deepcopy(dataset_info.statics)
        # Init object of register_dict
        self.diff_stats = dataset_info.diff_stats
        self.stats = dataset_info.stats

        # Keeping track of grid shape
        self.grid_shape = statics.grid_shape
        # For example plotting
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # class variables to log loss during training, for tensorboad custom scalar
        self.training_step_losses = []
        self.validation_step_losses = []

        # Set model input/output grid features based on dataset tensor shapes
        num_grid_static_features = statics.grid_statics.dim_size("features")

        # Compute the number of input features for the neural network
        # Should be directly supplied by datasetinfo ?

        num_input_features = (
            num_input_steps * dataset_info.weather_dim
            + num_grid_static_features
            + dataset_info.forcing_dim
        )

        num_output_features = dataset_info.weather_dim

        model_kls, model_settings = get_model_kls_and_settings(
            model_name, self.settings_init_args
        )

        # All processes should wait until rank zero
        # has done the initialization (like creating a graph)
        rank_zero_init(model_kls, model_settings, statics)

        self.model, model_settings = build_model_from_settings(
            model_name,
            num_input_features,
            num_output_features,
            self.settings_init_args,
            statics.grid_shape,
        )
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # We transform and register the statics after the model has been set up
        # This change the dimension of all statics
        if self.model.model_type == ModelType.GRAPH:
            # Graph model, we flatten the statics spatial dims
            statics.grid_statics.flatten_("ngrid", 0, 1)
            statics.border_mask = statics.border_mask.flatten(0, 1)
            statics.interior_mask = statics.interior_mask.flatten(0, 1)

        # Register interior and border mask.
        statics.register_buffers(self)

        self.num_spatial_dims = statics.grid_statics.num_spatial_dims

        self.register_buffer(
            "grid_static_features",
            expand_to_batch(statics.grid_statics.tensor, batch_size),
            persistent=False,
        )
        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if loss_name == "mse":
            self.loss = WeightedLoss("MSELoss", reduction="none")
        elif loss_name == "mae":
            self.loss = WeightedLoss("L1Loss", reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {loss_name}")
        print(dataset_info)
        self.loss.prepare(self, statics.interior_mask, dataset_info)

        exp_summary(self)

    def setup(self, stage=None):
        self.logger.log_hyperparams(self.hparams, metrics={"val_mean_loss": 0.0})
        if self.logging_enabled:
            self.save_path = Path(self.trainer.logger.log_dir)
            max_pred_step = self.num_pred_steps_val_test - 1
            self.rmse_psd_plot_metric = MetricPSDVar(pred_step=max_pred_step)
            self.psd_plot_metric = MetricPSDK(self.save_path, pred_step=max_pred_step)
            self.acc_metric = MetricACC(self.dataset_info)
            self.configure_loggers()

    def configure_loggers(self):
        layout = {
            "Check Overfit": {
                "loss": [
                    "Multiline",
                    ["mean_loss_epoch/train", "mean_loss_epoch/validation"],
                ],
            },
        }
        self.logger.experiment.add_custom_scalars(layout)

    @property
    def logging_enabled(self) -> bool:
        """
        Check if logging is enabled
        """
        return self.trainer.logger.log_dir is not None

    @property
    def dtype(self):
        """
        Return the appropriate torch dtype for the desired precision in hparams.
        """
        return str_to_dtype[self.trainer.precision]

    def print_summary_model(self):
        self.dataset_info.summary()
        print(f"Number of input_steps : {self.num_input_steps}")
        print(f"Number of pred_steps (training) : {self.num_pred_steps_train}")
        print(f"Number of pred_steps (test/val) : {self.num_pred_steps_val_test}")
        print(f"Number of intermediary steps :{self.num_inter_steps}")
        print(f"Training strategy :{self.training_strategy}")
        print(
            f"Model step duration : {self.dataset_info.pred_step /self.num_inter_steps}"
        )
        print(f"Model conf {self.model_conf}")
        print("---------------------")
        print(f"Loss {self.loss}")
        print(f"Batch size {self.batch_size}")
        print("---------------------------")

    @rank_zero_only
    def inspect_tensors(self):
        """
        Prints all tensor parameters and buffers
        of the model with name, shape and dtype.
        """
        # trainable parameters
        for name, param in self.named_parameters():
            print(name, param.shape, param.dtype)
        # buffers
        for name, buffer in self.named_buffers():
            print(name, buffer.shape, buffer.dtype)

    @rank_zero_only
    def log_hparams_tb(self):
        if self.logging_enabled and self.logger:
            # Save model & dataset conf as files
            if self.dataset_conf is not None:
                shutil.copyfile(self.dataset_conf, self.save_path / "dataset_conf.json")
            # Write commit and state of git repo in log file
            dest_git_log = self.save_path / "git_log.txt"
            out_log = (
                subprocess.check_output(["git", "log", "-n", "1"])
                .strip()
                .decode("utf-8")
            )
            out_status = (
                subprocess.check_output(["git", "status"]).strip().decode("utf-8")
            )
            with open(dest_git_log, "w") as f:
                f.write(out_log)
                f.write(out_status)

    def on_fit_start(self):
        self.log_hparams_tb()

    def _next_x(
        self, batch: ItemBatch, prev_states: NamedTensor, step_idx: int
    ) -> torch.Tensor:
        """
        Build the next x input for the model at timestep step_idx using the :
        - previous states
        - forcing
        - static features
        """
        forcing = batch.forcing.select_dim("timestep", step_idx, bare_tensor=False)
        ds = self.training_strategy == "downscaling_only"
        inputs = [
            prev_states.select_dim("timestep", idx) * (1 - ds)
            for idx in range(batch.num_input_steps)
        ]
        x = torch.cat(
            inputs
            + [self.grid_static_features[: batch.batch_size], forcing.tensor],
            dim=forcing.dim_index("features"),
        )
        return x

    def _step_diffs(
        self, feature_names: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and std of the differences between two consecutive states on the desired device.
        """
        step_diff_std = self.diff_stats.to_list("std", feature_names).to(
            device,
            non_blocking=True,
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
        force_border: bool = True if self.training_strategy == "scaled_ar" else False
        scale_y: bool = True if self.training_strategy == "scaled_ar" else False
        # raise if mismatch between strategy and num_inter_steps
        if self.training_strategy == "diff_ar":
            if self.num_inter_steps != 1:
                raise ValueError(
                    "Diff AR strategy requires exactly 1 intermediary step."
                )

        return force_border, scale_y, self.num_inter_steps

    def common_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Handling autocast subtelty for mixed precision on GPU and CPU (only bf16 for the later).
        """
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                return self._common_step(batch, inference)
        else:
            if not inference and "bf16" in self.trainer.precision:
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    return self._common_step(batch, inference)
            else:
                return self._common_step(batch, inference)

    def _common_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Two Autoregressive strategies are implemented here for train, val, test and inference:
        - scaled_ar:
            * Boundary forcing with y_true/true_state
            * Scaled Differential update next_state = prev_state + y * std + mean
            * Intermediary steps for which we have no y_true data

        - diff_ar:
            * No Boundary forcing
            * Differential update next_state = prev_state + y
            * No Intermediary steps

        Derived/Inspired from https://github.com/joeloskarsson/neural-lam/

        In inference mode, we assume batch.outputs is None and we disable output based border forcing.
        """
        force_border, scale_y, num_inter_steps = self._strategy_params()
        # Right now we postpone that we have a single input/output/forcing

        self.original_shape = None

        if self.model.model_type == ModelType.GRAPH:
            # Stack original shape to reshape later
            self.original_shape = batch.inputs.tensor.shape
            # Graph model, we flatten the batch spatial dims
            batch.inputs.flatten_("ngrid", *batch.inputs.spatial_dim_idx)

            if not inference:
                batch.outputs.flatten_("ngrid", *batch.outputs.spatial_dim_idx)

            batch.forcing.flatten_("ngrid", *batch.forcing.spatial_dim_idx)

        prev_states = batch.inputs
        prediction_list = []

        # Here we do the autoregressive prediction looping
        # for the desired number of ar steps.
        for i in range(batch.num_pred_steps):
            if not inference:
                border_state = batch.outputs.select_dim("timestep", i)

            if scale_y:
                step_diff_std, step_diff_mean = self._step_diffs(
                    (
                        self.output_feature_names
                        if inference
                        else batch.outputs.feature_names
                    ),
                    prev_states.device,
                )

            # Intermediary steps for which we have no y_true data
            # Should be greater or equal to 1 (otherwise nothing is done).
            for k in range(num_inter_steps):
                x = self._next_x(batch, prev_states, i)
                # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
                if self.channels_last:
                    x = x.to(memory_format=torch.channels_last)

                # Here we adapt our tensors to the order of dimensions of CNNs and ViTs
                if self.model.features_second:
                    x = features_last_to_second(x)
                    y = self.model(x)
                    y = features_second_to_last(y)
                else:
                    y = self.model(x)

                # We update the latest of our prev_states with the network output
                if scale_y:
                    predicted_state = (
                        # select the last timestep
                        prev_states.select_dim("timestep", -1)
                        + y * step_diff_std
                        + step_diff_mean
                    )
                else:
                    ds = self.training_strategy == "downscaling_only"
                    predicted_state = prev_states.select_dim("timestep", -1) * (1 - ds)+ y

                # Overwrite border with true state
                # Force it to true state for all intermediary step
                if not inference and force_border:
                    new_state = (
                        self.border_mask * border_state
                        + self.interior_mask * predicted_state
                    )
                else:
                    new_state = predicted_state

                # Only update the prev_states if we are not at the last step
                if i < batch.num_pred_steps - 1 or k < num_inter_steps - 1:
                    # Update input states for next iteration: drop oldest, append new_state
                    timestep_dim_index = batch.inputs.dim_index("timestep")
                    new_prev_states_tensor = torch.cat(
                        [
                            # Drop the oldest timestep (select all but the first)
                            prev_states.index_select_dim(
                                "timestep",
                                range(1, prev_states.dim_size("timestep")),
                            ),
                            # Add the timestep dimension to the new state
                            new_state.unsqueeze(timestep_dim_index),
                        ],
                        dim=timestep_dim_index,
                    )

                    # Make a new NamedTensor with the same dim and
                    # feature names as the original prev_states
                    prev_states = NamedTensor.new_like(
                        new_prev_states_tensor, prev_states
                    )
            # Append prediction to prediction list only "normal steps"
            prediction_list.append(new_state)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

        # In inference mode we use a "trained" module which MUST have the output feature names
        # and the output dim names attributes set.
        if inference:
            pred_out = NamedTensor(
                prediction.type(self.output_dtype),
                self.output_dim_names,
                self.output_feature_names,
            )
        else:
            pred_out = NamedTensor.new_like(
                prediction.type_as(batch.outputs.tensor), batch.outputs
            )
        return pred_out, batch.outputs

    def on_train_start(self):
        self.train_plotters = []

    def _shared_epoch_end(self, outputs: List[torch.Tensor], label: str) -> None:
        """Computes and logs the averaged metrics at the end of an epoch.
        Step shared by training and validation epochs.
        """
        if self.logging_enabled:
            avg_loss = torch.stack([x for x in outputs]).mean()
            tb = self.logger.experiment
            tb.add_scalar(f"mean_loss_epoch/{label}", avg_loss, self.current_epoch)

    def training_step(self, batch: ItemBatch, batch_idx: int) -> torch.Tensor:
        """
        Train on single batch
        """

        # we save the feature names at the first batch
        # to check at inference time if the feature names are the same
        # also useful to build NamedTensor outputs with same feature and dim names
        if batch_idx == 0:
            self.input_feature_names = batch.inputs.feature_names
            self.output_feature_names = batch.outputs.feature_names
            self.output_dim_names = batch.outputs.names
            self.output_dtype = batch.outputs.tensor.dtype

        prediction, target = self.common_step(batch)
        # Compute loss: mean over unrolled times and batch
        batch_loss = torch.mean(self.loss(prediction, target))

        self.training_step_losses.append(batch_loss)

        # Notify every plotters
        if self.logging_enabled:
            for plotter in self.train_plotters:
                plotter.update(self, prediction=self.prediction, target=self.target)

        return batch_loss

    def on_save_checkpoint(self, checkpoint):
        """
        We store our feature and dim names in the checkpoint
        """
        checkpoint["input_feature_names"] = self.input_feature_names
        checkpoint["output_feature_names"] = self.output_feature_names
        checkpoint["output_dim_names"] = self.output_dim_names
        checkpoint["output_dtype"] = self.output_dtype

    def on_load_checkpoint(self, checkpoint):
        """
        We load our feature and dim names from the checkpoint
        """
        self.input_feature_names = checkpoint["input_feature_names"]
        self.output_feature_names = checkpoint["output_feature_names"]
        self.output_dim_names = checkpoint["output_dim_names"]
        self.output_dtype = checkpoint["output_dtype"]

    def predict_step(self, batch: ItemBatch, batch_idx: int) -> torch.Tensor:
        """
        Check if the feature names are the same as the one used during training
        and make a prediction and accumulate if io_conf =/= none.
        """
        if batch_idx == 0:
            if self.input_feature_names != batch.inputs.feature_names:
                raise ValueError(
                    f"Input Feature names mismatch between training and inference. "
                    f"Training: {self.input_feature_names}, Inference: {batch.inputs.feature_names}"
                )
        preds = self.forward(batch)
        if not (self.io_conf is None):
            self.grib_writing(preds)
        return preds

    def grib_writing(self, preds):
        with open(self.io_conf, "r") as f:
            save_settings = GribSavingSettings(**json.load(f))
            ph = len(save_settings.output_fmt.split("{}")) - 1
            kw = len(save_settings.output_kwargs)
            fi = len(save_settings.sample_identifiers)
            if ph != (fi + kw):
                raise ValueError(
                    f"Filename fmt has {ph} placeholders,\
                    but {kw} output_kwargs and {fi} sample identifiers."
                )
        for sample, pred in zip(self.infer_ds.sample_list, preds):
            save_named_tensors_to_grib(pred, self.infer_ds, sample, save_settings)

    def forward(self, x: ItemBatch) -> NamedTensor:
        """
        Forward pass of the model
        """
        return self.common_step(x, inference=True)[0]

    def on_train_epoch_end(self):
        outputs = self.training_step_losses
        self._shared_epoch_end(outputs, "train")
        self.training_step_losses.clear()  # free memory

    def on_train_end(self):
        if self.mlflow_logger:

            # Get random sample to infer the signature of the model
            dataloader = self.trainer.datamodule.test_dataloader()
            data = next(iter(dataloader))
            signature = infer_signature(
                data.inputs.tensor.detach().numpy(),
                data.outputs.tensor.detach().numpy(),
            )

            # Manually log the trained model in Mlflow style with its signature
            run_id = self.mlflow_logger.version
            with mlflow.start_run(run_id=run_id):
                mlflow.pytorch.log_model(
                    pytorch_model=self.trainer.model,
                    artifact_path="model",
                    signature=signature,
                )

    def on_validation_start(self):
        """
        Add some plots when starting validation
        """
        if self.logging_enabled:
            l1_loss = ScaledLoss("L1Loss", reduction="none")
            l1_loss.prepare(self, self.interior_mask, self.dataset_info)
            metrics = {"mae": l1_loss}
            self.valid_plotters = [
                StateErrorPlot(metrics, prefix="Validation"),
                PredictionTimestepPlot(
                    num_samples_to_plot=1,
                    num_features_to_plot=4,
                    prefix="Validation",
                    save_path=self.save_path,
                ),
                PredictionEpochPlot(
                    num_samples_to_plot=1,
                    num_features_to_plot=4,
                    prefix="Validation",
                    save_path=self.save_path,
                ),
            ]

    def _shared_val_test_step(self, batch: ItemBatch, batch_idx, label: str):
        with torch.no_grad():
            prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(self.loss(prediction, target), dim=0)
        mean_loss = torch.mean(time_step_loss)

        if self.logging_enabled:
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

        if self.logging_enabled:
            # Notify every plotters
            if self.current_epoch % PLOT_PERIOD == 0:
                for plotter in self.valid_plotters:
                    plotter.update(self, prediction=prediction, target=target)
            self.psd_plot_metric.update(prediction, target, self.original_shape)
            self.rmse_psd_plot_metric.update(prediction, target, self.original_shape)
            self.acc_metric.update(prediction, target)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """

        if self.logging_enabled:
            # Get dict of metrics' results
            dict_metrics = dict()
            dict_metrics.update(self.psd_plot_metric.compute())
            dict_metrics.update(self.rmse_psd_plot_metric.compute())
            dict_metrics.update(self.acc_metric.compute())
            for name, elmnt in dict_metrics.items():
                if isinstance(elmnt, matplotlib.figure.Figure):
                    # Tensorboard logger
                    self.logger.experiment.add_figure(
                        f"{name}", elmnt, self.current_epoch
                    )
                    # If MLFlowLogger activated
                    if self.mlflow_logger:
                        run_id = self.mlflow_logger.version
                        self.mlflow_logger.experiment.log_figure(
                            run_id=run_id,
                            figure=elmnt,
                            artifact_file=f"figures/{name}.png",
                        )
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

        if self.logging_enabled:
            # Notify every plotters
            if self.current_epoch % PLOT_PERIOD == 0:
                for plotter in self.valid_plotters:
                    plotter.on_step_end(self, label="Valid")

    def on_test_start(self):
        """
        Attach observer when starting test
        """
        if self.logging_enabled:
            metrics = {}
            for torch_loss, alias in ("L1Loss", "mae"), ("MSELoss", "rmse"):
                loss = ScaledLoss(torch_loss, reduction="none")
                loss.prepare(self, self.interior_mask, self.dataset_info)
                metrics[alias] = loss

            self.test_plotters = [
                StateErrorPlot(metrics, save_path=self.save_path),
                SpatialErrorPlot(),
                PredictionTimestepPlot(
                    num_samples_to_plot=self.num_samples_to_plot,
                    num_features_to_plot=4,
                    prefix="Test",
                    save_path=self.save_path,
                ),
            ]

    def test_step(self, batch: ItemBatch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, _ = self._shared_val_test_step(batch, batch_idx, "test")

        if self.logging_enabled:
            # Notify plotters & metrics
            for plotter in self.test_plotters:
                plotter.update(self, prediction=prediction, target=target)

            self.acc_metric.update(prediction, target)
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
        if self.logging_enabled:
            dict_metrics = {}
            dict_metrics.update(self.psd_plot_metric.compute(prefix="test"))
            dict_metrics.update(self.rmse_psd_plot_metric.compute(prefix="test"))
            dict_metrics.update(self.acc_metric.compute(prefix="test"))

            for name, elmnt in dict_metrics.items():
                if isinstance(elmnt, matplotlib.figure.Figure):
                    # Tensorboard logger
                    self.logger.experiment.add_figure(
                        f"{name}", elmnt, self.current_epoch
                    )
                    # If MLFlowLogger activated
                    if self.mlflow_logger:
                        run_id = self.mlflow_logger.version
                        self.mlflow_logger.experiment.log_figure(
                            run_id=run_id,
                            figure=elmnt,
                            artifact_file=f"figures/{name}.png",
                        )
                elif isinstance(elmnt, torch.Tensor):
                    self.log_dict(
                        {name: elmnt},
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

            # Notify plotters that the test epoch end
            for plotter in self.test_plotters:
                plotter.on_step_end(self, label="Test")

    @cached_property
    def mlflow_logger(self) -> Union[MLFlowLogger, None]:
        """
        Get the MLFlowLogger if it has been set.
        """
        return next(
            iter([o for o in self.loggers if isinstance(o, MLFlowLogger)]), None
        )


@rank_zero_only
def exp_summary(model: AutoRegressiveLightning):
    model.print_summary_model()
    summary(model.model)
