from dataclasses import asdict

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torch import optim

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Union

import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup


from py4cast.datasets.base import DatasetInfo
from py4cast.models import build_model_from_settings
from py4cast.utils import str_to_dtype

# learning rate scheduling period in steps (update every nth step)
LR_SCHEDULER_PERIOD: int = 10

@dataclass
class MAELightningHyperParam:
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
    precision: str = "bf16"
    no_log: bool = False
    channels_last: bool = False
    dev_mode: bool = False

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
def exp_summary(hparams: MAELightningHyperParam, model: nn.Module):
    hparams.summary()
    summary(model)





class MAELightningModule(pl.LightningModule):
    """A lightning module adapted for MAE.
    Computes metrics and manages logging in tensorboard."""

    @classmethod
    def from_hyperparams(cls, hparams: MAELightningHyperParam):
        """Builds the lightning instance using hyper-parameters."""
        if hparams.load_from_checkpoint is None:
            model = cls(hparams)
        else:
            model = cls.load_from_checkpoint(hparams.load_from_checkpoint)
        return model




    def __init__(self, hparams: MAELightningHyperParam):
        super().__init__()

        self.statics = deepcopy(hparams.dataset_info.statics)
        self.hyparams = hparams

        if self.hyparams.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None


        self.metrics = self.get_metrics()

        # class variables to log metrics during training
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_metrics = {}
        '''
        # log model hparams, needed for 'load_from_checkpoint'
        if hparams.profiler == "pytorch":
            self.save_hyperparameters(ignore=["loss"])
        else:
            self.save_hyperparameters()
        '''
        self.save_hyperparameters()
  
        if hparams.loss == "mse":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {hparams.loss}")

        # Set model input/output grid features based on dataset tensor shapes
        num_grid_static_features = self.statics.grid_static_features.dim_size("features")
        # Compute the number of input features for the neural network
        # Should be directly supplied by datasetinfo ?
        self.num_input_features = hparams.dataset_info.weather_dim
        
        self.num_output_features = hparams.dataset_info.weather_dim

        self.model, self.hyparams.model_settings = self.create_model()


    def create_model(self):
        """Creates a model with the config file (.json) if available."""
        model, model_settings = build_model_from_settings(
            self.hyparams.model_name,
            self.num_input_features,
            self.num_output_features,
            self.hyparams.model_conf,
            self.statics.grid_shape,
        )
        return model, model_settings

    def get_metrics(self):
        """Defines the metrics that will be computed during valid and test steps."""

        metrics_dict = torch.nn.ModuleDict(
                {
                    "rmse": tm.MeanSquaredError(squared=False),
                    "mae": tm.MeanAbsoluteError(),
                    "mape": tm.MeanAbsolutePercentageError(),
                }
            )
        return metrics_dict

    def forward(self, inputs):
        """Runs data through the model. Separate from training step."""
        if self.hyparams.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        y_hat = self.model(inputs)
        return y_hat

    def _shared_forward_step(self, x, y, train=False):
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
        if self.hyparams.channels_last:
            x = x.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)
        y_hat = self.model(x, train=train)
        loss = self.loss(y_hat.float(), y).mean()
        return y_hat, loss

    def _shared_metrics_step(self, loss, x, y, y_hat, label):
        """Computes metrics for a batch.
        Step shared by validation and test steps."""
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics.items():
            
            metric.update(y_hat.reshape(-1), y.int().reshape(-1))
            batch_dict[metric_name] = metric.compute()
            metric.reset()
        return batch_dict

    def _shared_epoch_end(self, outputs, label):
        """Computes and logs the averaged metrics at the end of an epoch.
        Step shared by training and validation epochs.
        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tb = self.logger.experiment
        tb.add_scalar(f"loss/{label}", avg_loss, self.current_epoch)
        if label == "validation":
            for metric in self.metrics:
                avg_m = torch.stack([x[metric] for x in outputs]).mean()
                tb.add_scalar(f"metrics/{label}_{metric}", avg_m, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1), torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1)
        _, loss = self._shared_forward_step(x, y, train=True)
        batch_dict = {"loss": loss}
        self.log(
            "loss_step", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.training_step_outputs.append(batch_dict)
        return loss

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        self._shared_epoch_end(outputs, "train")
        self.training_step_outputs.clear()  # free memory

    def val_plot_step(self, batch_idx, y, y_hat):
        """Plots images on first batch of validation and log them in tensorboard."""
        if batch_idx == 0:
            tb = self.logger.experiment
            step = self.current_epoch
            dformat = "CHW"
            if step == 0:
                tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
            tb.add_image("val_plots/test_image", y_hat[0], step, dataformats=dformat)

    def validation_step(self, batch, batch_idx):
        x, y = torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1), torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1)
        y_hat, loss = self._shared_forward_step(x, y)
        batch_dict = self._shared_metrics_step(loss, x, y, y_hat, "validation")
        self.log("val_mean_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append(batch_dict)
        if not self.hyparams.dev_mode:
            self.val_plot_step(batch_idx, y, y_hat)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self._shared_epoch_end(outputs, "validation")
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        """Computes metrics for each sample, at the end of the run."""
        x, y, name = torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1), torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1)
        y_hat, loss = self._shared_forward_step(x, y)
        batch_dict = self._shared_metrics_step(loss, x, y, y_hat, "test")
        self.test_metrics[name[0]] = batch_dict

    def on_test_epoch_end(self):
        """Logs metrics in tensorboard hparams view, at the end of run."""
        metrics = {}
        list_metrics = list(self.test_metrics.values())[0].keys()
        for metric_name in list_metrics:
            data = []
            for metrics_dict in self.test_metrics.values():
                data.append(metrics_dict[metric_name])
            metrics[metric_name] = torch.stack(data).mean().item()
        hparams = asdict(self.hyparams)
        hparams["loss"] = str(hparams["loss"])
        self.logger.log_hyperparams(hparams, metrics=metrics)

    def use_lr_scheduler(self, optimizer):
        lr = self.hyparams.learning_rate
        if self.hyparams.reduce_lr_plateau:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_mean_loss",
                },
            }
        elif self.hyparams.cyclic_lr:
            lr_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=lr, max_lr=10 * lr, cycle_momentum=False
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler},
            }
        else:
            return None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyparams.learning_rate)
        if self.hyparams.reduce_lr_plateau or self.hyparams.cyclic_lr:
            return self.use_lr_scheduler(optimizer)
        else:
            return optimizer
        
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
