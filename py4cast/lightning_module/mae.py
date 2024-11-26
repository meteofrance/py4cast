import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from pathlib import Path
from typing import Union

# A GARDER
from py4cast.settings import HIERA_MASKED_DIR
from py4cast.lightning_module.utils import (
    InitModelLightningModule,
    PlotLightningModule,
    MaskLightningModule,
)

# PAS CLAIR
from copy import deepcopy
from transformers import get_cosine_schedule_with_warmup


# learning rate scheduling period in steps (update every nth step)
LR_SCHEDULER_PERIOD: int = 10


class MAELightningModule(
    PlotLightningModule,
    MaskLightningModule,
    pl.LightningModule,
):
    """A lightning module adapted for MAE (masked auto-encoder).
    Computes metrics and manages logging in tensorboard."""

    def __init__(
        self,
        model_conf: Union[Path, None] = None,
        model_name: str = "unetrpp",
        lr: float = 0.1,
        loss_name: str = "mse",
        len_train_loader: int = 1,
        save_path: Path = None,
        use_lr_scheduler: bool = False,
        mask_ratio: float = 0.3,
    ):
        super().__init__(
            model_name=model_name, model_conf=model_conf
        )  # Initialize InitModelLightningModule
        super(
            PlotLightningModule, self
        ).__init__()  # Initialize InitModelLightningModule
        super(
            MaskLightningModule, self
        ).__init__()  # Initialize InitModelLightningModule

        # init model
        self.model, self.model_settings = self.setup("train")

        self.model_conf = model_conf
        self.model_name = model_name
        self.lr = lr
        self.loss_name = loss_name
        self.len_train_loader = len_train_loader
        self.save_path = save_path
        self.use_lr_scheduler = use_lr_scheduler

        self.training_step_metrics_loss = []
        self.validation_step_metrics_loss = []
        self.test_step_metrics_loss = []

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None
        self.loss = nn.MSELoss(reduction="none")
        self.metrics = self.get_metrics()

        # Specific to this model
        self.mask_ratio = mask_ratio

    ###--------------------- MISCELLANEOUS ---------------------###

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
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

    def forward(self, inputs):
        """Runs data through the model. Separate from training step."""
        y_hat = self.model(inputs)
        return y_hat

    ###--------------------- SHARED ---------------------###

    def shared_metrics_loss_step(self, loss, y, y_hat):
        """Computes metrics and loss for a batch.
        Step shared by train, validation and test steps."""
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics.items():
            metric.update(y_hat.reshape(-1), y.int().reshape(-1))
            batch_dict[metric_name] = metric.compute()
            metric.reset()
        return batch_dict

    ###--------------------- TRAIN ---------------------###

    def training_step(self, batch):
        x, y = (
            torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1),
            torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1),
        )
        mask = self.create_mask(x, self.mask_ratio)
        x_masked = x * mask  # element-wise product
        y_hat = self.model(x_masked)
        loss = self.loss(y_hat.float(), y).mean()
        batch_dict = self.shared_metrics_loss_step(loss, y, y_hat)
        self.training_step_metrics_loss.append(batch_dict)
        return loss

    def on_train_epoch_end(self):
        tb = self.logger.experiment
        self.log_loss(tb, self.training_step_metrics_loss, "train")
        self.log_metrics(tb, self.training_step_metrics_loss, "train")
        self.training_step_metrics_loss.clear()  # free memory

    def on_train_end(self):
        best_model_state = deepcopy(self.model.state_dict())
        torch.save(best_model_state, HIERA_MASKED_DIR)

    ###--------------------- VALIDATION ---------------------###

    def validation_step(self, batch):
        x, y = (
            torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1),
            torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1),
        )
        y_hat = self.model(x)
        loss = self.loss(y_hat.float(), y).mean()
        batch_dict = self.shared_metrics_loss_step(loss, y, y_hat)
        self.validation_step_metrics_loss.append(batch_dict)
        return loss

    def on_validation_epoch_end(self):
        tb = self.logger.experiment
        self.log_loss(tb, self.validation_step_metrics_loss, "val")
        self.log_metrics(tb, self.validation_step_metrics_loss, "val")
        self.validation_step_metrics_loss.clear()  # free memory

    ###--------------------- TEST ---------------------###

    def test_step(self, batch):
        x, y = (
            torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1),
            torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1),
        )
        y_hat = self.model(x)
        loss = self.loss(y_hat.float(), y).mean()
        batch_dict = self.shared_metrics_loss_step(loss, y, y_hat)
        self.test_step_metrics_loss.append(batch_dict)
        return loss

    def on_test_epoch_end(self):
        tb = self.logger.experiment
        self.log_loss(tb, self.test_step_metrics_loss, "test")
        self.log_metrics(tb, self.test_step_metrics_loss, "test")
        self.test_step_metrics_loss.clear()  # free memory
