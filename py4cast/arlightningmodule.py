import torch
import torchmetrics as tm
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
from typing import Tuple
import matplotlib.figure as pltfig
import einops
from functools import cached_property
from copy import deepcopy
from transformers import get_cosine_schedule_with_warmup
from py4cast.models import build_model_from_settings
from py4cast.losses import WeightedLoss
from py4cast.metrics import MetricACC, MetricPSDK, MetricPSDVar
from py4cast.models.base import expand_to_batch
from py4cast.utils import str_to_dtype
from py4cast.datasets.base import ItemBatch, NamedTensor
from py4cast.plots import (
    PredictionEpochPlot,
    PredictionTimestepPlot,
    SpatialErrorPlot,
    StateErrorPlot,
)

# learning rate scheduling period in steps (update every nth step)
LR_SCHEDULER_PERIOD: int = 10
# PNG plots period in epochs. Plots are made, logged and saved every nth epoch.
PLOT_PERIOD: int = 1

class AutoRegressiveLightningModule(pl.LightningModule):
    """A lightning module adapted for test.
    Computes metrics and manages logging in tensorboard."""

    def __init__(
        self,
        # args linked from datamodule
        dataset_info,
        dataset_name: str,
        batch_shape: Tuple[int, int, int, int, int],
        batch_size: int,
        num_input_steps: int = 1, # Ti, Ti-1 Ti-2, etc.
        num_pred_steps: int = 1, # Ti+1, Ti+2, Ti+3, etc.
        len_train_loader: int = 1,
        # args exclusive to lightningmodule
        model_conf: Path | None = None,  # Path | None
        model_name: str = "halfunet",
        lr: float = 0.1, # initial value of learning rate
        loss_name: str = "mse",
        save_path: Path = None, # Metrics and Plots directory
        use_lr_scheduler: bool = False,
        precision: str = "bf16", # degree of precision
        no_log: bool = False, # False = dont log things on tensorboard
        channels_last: bool = False, # False ~ (B,T,C,H,W)
    ):
        super().__init__()
        # exclusive args
        self.batch_size = batch_size
        self.model_conf = model_conf
        self.model_name = model_name
        self.lr = lr
        self.loss_name = loss_name
        self.num_input_steps = num_input_steps
        self.num_pred_steps = num_pred_steps
        self.len_train_loader = len_train_loader
        self.save_path = save_path
        self.use_lr_scheduler = use_lr_scheduler
        self.precision = precision
        self.no_log = no_log
        self.channels_last = channels_last
        # linked args
        self.batch_shape = batch_shape # (B, T, H, W, C)
        self.dataset_info = dataset_info
        self.dataset_name = dataset_name

        # Creates a model with the config file (.json) if available.
        self.input_shape = self.batch_shape[2:4]
        self.num_output_features = self.batch_shape[4]  # = nombre de feature du dataset
        self.num_input_features = self.num_output_features*self.num_input_steps + 10 # concat(x,statics)
        self.model, self.model_settings = build_model_from_settings(
            network_name=self.model_name,
            num_input_features=self.num_input_features,
            num_output_features=self.num_output_features,
            settings_path=self.model_conf,
            input_shape=self.input_shape,
        )

        self.save_path = Path(self.save_path) / f"version_{3}"

        self.opt_state = (
            None  # For making restoring of optimizer state optional (slight hack)
        )
        self.max_pred_step = self.num_pred_steps - 1
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
        self.test_metrics = self.get_metrics()
        self.save_hyperparameters()  # write hparams.yaml in save folder
        statics = deepcopy(self.dataset_info.statics)
        self.grid_shape = statics.grid_shape
        self.num_spatial_dims = statics.grid_static_features.num_spatial_dims
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # We transform and register the statics after the model has been set up
        # This change the dimension of all statics
        if len(self.model.input_dims) == 3:
            # Graph model, we flatten the statics spatial dims
            statics.grid_static_features.flatten_("ngrid", 0, 1)
            statics.border_mask = statics.border_mask.flatten(0, 1)
            statics.interior_mask = statics.interior_mask.flatten(0, 1)

        # Register interior and border mask.
        statics.register_buffers(self)
        self.register_buffer(
            "grid_static_features",
            expand_to_batch(statics.grid_static_features.tensor, self.batch_size),
            persistent=False,
        )

        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if self.loss_name == "mse":
            self.loss = WeightedLoss("MSELoss", reduction="none")
        elif self.loss_name == "mae":
            self.loss = WeightedLoss("L1Loss", reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {self.loss}")
        self.loss.prepare(self, statics.interior_mask, self.dataset_info)
        self.batch_losses = []
        self.loss_for_plot = {self.loss_name : self.loss}

    ###--------------------- MISCELLANEOUS ---------------------###

    @property
    def dtype(self):
        """
        Return the appropriate torch dtype for the desired precision in hparams.
        """
        return str_to_dtype[self.precision]

    @property
    def logging_enabled(self):
        """
        Check if logging is enabled
        """
        return not self.no_log

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

    def get_metrics(self):
        """Defines the metrics that will be computed during valid and test steps."""
        metrics_dict = torch.nn.ModuleDict(
            {
                "rmse_psd_plot_metric": MetricPSDVar(pred_step=self.max_pred_step),
                "psd_plot_metric": MetricPSDK(
                    self.save_path, pred_step=self.max_pred_step
                ),
                "acc_metric": MetricACC(self.dataset_info),
            }
        )
        return metrics_dict

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Path to the folder where checkpoints and figures are saved
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        if self.use_lr_scheduler:
            len_loader = self.len_train_loader // LR_SCHEDULER_PERIOD
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

    def forward(self, batch: ItemBatch):
        """Runs data through the model. Separate from training step."""
        y_hat, _ = self.shared_step(batch)
        return y_hat

    ###--------------------- SHARED ---------------------###

    def shared_metrics_step(self, y, y_hat, shape, label):
        """Updates metrics for a batch."""
        if label == 'train':
            metric_list = self.train_metrics
        elif label == 'val':
            metric_list = self.val_metrics
        elif label == 'test':
            metric_list = self.test_metrics
        else:
            raise ValueError(f"Unknown label: {label}")

        for metric_name, metric in metric_list.items():
            if isinstance(metric, tm.Metric):
                try:
                    metric.update(y_hat, y)
                except TypeError:
                    metric.update(y_hat, y, shape=shape)

    def shared_metrics_end(self, label):
        """Computes and logs the metrics (to be used at the end of each epoch)."""
        if label == 'train':
            metric_list = self.train_metrics
        elif label == 'val':
            metric_list = self.val_metrics
        elif label == 'test':
            metric_list = self.test_metrics
        else:
            raise ValueError(f"Unknown label: {label}")
    
        for metric_name, metric in metric_list.items():
            try:
                computed_metrics = metric.compute(label)
                for name, elmnt in computed_metrics.items():
                    if isinstance(elmnt, pltfig.Figure):
                        self.logger.experiment.add_figure(
                            f"{label}/{name}",
                            elmnt,
                            self.current_epoch,
                        )
                    elif isinstance(elmnt, torch.Tensor):
                        if elmnt.dim() == 0:
                            self.log(
                                f"{label}/{name}", 
                                elmnt.item(),
                                prog_bar=False,
                                on_step=False,
                                on_epoch=True,
                                sync_dist=True,
                            ) 
                        else:
                            self.log(
                                f"{label}/{name}",
                                elmnt,
                                prog_bar=False,
                                on_step=False,
                                on_epoch=True,
                                sync_dist=True,
                            )
                    else:
                        print(f"Error logging metric {metric_name}")
            except Exception as e:
                print(f"Error computing metric '{metric_name}': {e}")

    def _shared_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        self.original_shape = None
        if len(self.model.input_dims) == 3:
            self.original_shape = batch.inputs.tensor.shape
            # Graph model, we flatten the batch spatial dims
            batch.inputs.flatten_("ngrid", *batch.inputs.spatial_dim_idx)
            if not inference:
                batch.outputs.flatten_("ngrid", *batch.outputs.spatial_dim_idx)
            batch.forcing.flatten_("ngrid", *batch.forcing.spatial_dim_idx)
        prev_states = batch.inputs
        prediction_list = []
        # autoregressive prediction loop
        for i in range(batch.num_pred_steps):
            """
            Build the next x input for the model at timestep step_idx using the :
            - previous states
            - forcing
            - static features
            """
            forcing = batch.forcing.select_dim("timestep", i, bare_tensor=False)
            x = torch.cat(
                [
                    prev_states.select_dim("timestep", idx)
                    for idx in range(batch.num_input_steps)
                ]
                + [self.grid_static_features[: batch.batch_size], forcing.tensor],
                dim=forcing.dim_index("features"),
            )
            if (
                self.channels_last
            ):  # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
                x = x.to(memory_format=torch.channels_last)
            y = self.model(x)
            predicted_state = (
                prev_states.select_dim("timestep", -1) + y
            )  # update the latest of our prev_states with the network output
            new_state = predicted_state

            if (
                i < batch.num_pred_steps - 1
            ):  # Only update the prev_states if we are not at the last step
                # Update input states for next iteration: drop oldest, append new_state
                timestep_dim_index = batch.inputs.dim_index("timestep")
                new_prev_states_tensor = torch.cat(
                    [
                        prev_states.index_select_dim(  # Drop the oldest timestep (select all but the first)
                            "timestep",
                            range(1, prev_states.dim_size("timestep")),
                        ),
                        new_state.unsqueeze(
                            timestep_dim_index
                        ),  # Add the timestep dimension to the new state
                    ],
                    dim=timestep_dim_index,
                )
                prev_states = NamedTensor.new_like(new_prev_states_tensor, prev_states)
            prediction_list.append(new_state)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

        # In inference mode we use a "trained" module which MUST have the output feature names and the output dim names attributes set.
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

    def shared_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Handling autocast subtelty for mixed precision on GPU and CPU (only bf16 for the later).
        """
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                return self._shared_step(batch, inference)
        else:
            if not inference and "bf16" in self.trainer.precision:
                with torch.cpu.amp.autocast(dtype=self.dtype):
                    return self._shared_step(batch, inference)
            else:
                return self._shared_step(batch, inference)

    ###--------------------- TRAIN ---------------------###

    def training_step(self, batch):
        y_hat, y = self.shared_step(batch)
        loss = self.loss(y_hat, y).mean()
        self.log(f"train_loss", loss, on_step=False, on_epoch=True, reduce_fx="mean", prog_bar=True, sync_dist=True)
        self.shared_metrics_step(y, y_hat, self.original_shape, "train")
        return loss

    def on_train_epoch_end(self):
        self.shared_metrics_end("train")

    ###--------------------- VALIDATION ---------------------###

    def on_validation_start(self):
        self.valid_plotters = [
            StateErrorPlot(self.loss_for_plot, prefix="Validation"),
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

    def validation_step(self, batch):
        y_hat, y = self.shared_step(batch)
        loss = self.loss(y_hat, y).mean()
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, reduce_fx="mean", prog_bar=True, sync_dist=True)
        self.shared_metrics_step(y, y_hat, self.original_shape, "val")
        if self.current_epoch % PLOT_PERIOD == 0:
            for plotter in self.valid_plotters:
                plotter.update(self, prediction=y_hat, target=y)
        return loss
    
    def on_validation_epoch_end(self):
        self.shared_metrics_end("val")
        if self.current_epoch % PLOT_PERIOD == 0:
            for plotter in self.valid_plotters:
                plotter.on_step_end(self, label="Valid")

    ###--------------------- TEST ---------------------###

    def on_test_start(self):
        self.test_plotters = [
            StateErrorPlot(self.loss_for_plot, save_path=self.save_path),
            SpatialErrorPlot(),
            PredictionTimestepPlot(
                num_samples_to_plot=1,
                num_features_to_plot=4,
                prefix="Test",
                save_path=self.save_path,
            ),
        ]

    def test_step(self, batch):
        y_hat, y = self.shared_step(batch)
        loss = self.loss(y_hat, y).mean()
        self.log(f"test_loss", loss, on_step=False, on_epoch=True, reduce_fx="mean", prog_bar=True, sync_dist=True)
        self.shared_metrics_step(y, y_hat, self.original_shape, "test")
        if self.current_epoch % PLOT_PERIOD == 0:
            for plotter in self.test_plotters:
                plotter.update(self, prediction=y_hat, target=y)
        return loss

    def on_test_epoch_end(self):
        self.shared_metrics_end("test")
        if self.current_epoch % PLOT_PERIOD == 0:
            for plotter in self.test_plotters:
                plotter.on_step_end(self, label="Test")
