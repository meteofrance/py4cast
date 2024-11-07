import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from copy import deepcopy
from pathlib import Path
from typing import Union, Dict
from lightning.pytorch.utilities import rank_zero_only
from transformers import get_cosine_schedule_with_warmup

from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.models import build_model_from_settings
from py4cast.datasets import get_datasets
from py4cast.settings import HIERA_MASKED_DIR

# learning rate scheduling period in steps (update every nth step)
LR_SCHEDULER_PERIOD: int = 10

class PlDataModule(pl.LightningDataModule):
    """
    DataModule to encapsulate data splits and data loading.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_conf: Union[Path, None],
        config_override: Union[Dict, None] = None,
        num_input_steps: int = 1,
        num_pred_steps_train: int = 1,
        num_pred_steps_val_test: int = 1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dl_settings = TorchDataloaderSettings()
        self.dataset_conf = dataset_conf
        self.config_override = config_override
        self.num_input_steps = num_input_steps
        self.num_pred_steps_train = num_pred_steps_train
        self.num_pred_steps_val_test = num_pred_steps_val_test

        # Get dataset in initialisation to have access to this attribute before method trainer.fit
        self.train_ds, self.val_ds, self.test_ds = get_datasets(
            self.dataset_name,
            self.num_input_steps,
            self.num_pred_steps_train,
            self.num_pred_steps_val_test,
            self.dataset_conf,
            self.config_override,
        )

    @property
    def len_train_dl(self):
        return len(self.train_ds.torch_dataloader(self.dl_settings))

    @property
    def train_dataset_info(self):
        return self.train_ds.dataset_info

    @property
    def infer_ds(self):
        return self.test_ds

    def train_dataloader(self):
        return self.train_ds.torch_dataloader(self.dl_settings)

    def val_dataloader(self):
        return self.val_ds.torch_dataloader(self.dl_settings)

    def test_dataloader(self):
        return self.test_ds.torch_dataloader(self.dl_settings)

    def predict_dataloader(self):
        return self.test_ds.torch_dataloader(self.dl_settings)

@rank_zero_only
def rank_zero_init(model_kls, model_settings, statics):
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(model_settings, statics)


class MAELightningModule(pl.LightningModule):
    """A lightning module adapted for MAE.
    Computes metrics and manages logging in tensorboard."""

    def __init__(
        self,
        dataset_name: str,
        model_conf: Union[Path, None] = None,
        model_name: str = "hiera",
        lr: float = 0.1,
        loss_name: str = "mse",
        num_input_steps: int = 1,
        num_pred_steps_train: int = 1,
        num_inter_steps: int = 0,  # Number of intermediary steps (without any data)
        num_pred_steps_val_test: int = 1,
        len_train_loader: int = 1,
        save_path: Path = None,
        use_lr_scheduler: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_conf = model_conf
        self.model_name = model_name
        self.lr = lr
        self.loss_name = loss_name
        self.num_input_steps = num_input_steps
        self.num_pred_steps_train = num_pred_steps_train
        self.num_inter_steps = num_inter_steps
        self.num_pred_steps_val_test = num_pred_steps_val_test
        self.len_train_loader = len_train_loader
        self.save_path = save_path
        self.use_lr_scheduler = use_lr_scheduler

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None
        self.metrics = self.get_metrics()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_metrics = {}
        if self.loss_name == "mse":
            self.loss = nn.MSELoss(reduction="none")
        if dataset_name == "titan":
            self.num_input_features = 21
            self.num_output_features = 21
            self.grid_shape = (512, 640)
        else:  # to change
            self.num_input_features = 3
            self.num_output_features = 3
            self.grid_shape = (64, 64)
        self.model, self.model_settings = self.create_model()

    def create_model(self):
        """Creates a model with the config file (.json) if available."""
        model, model_settings = build_model_from_settings(
            self.model_name,
            self.num_input_features,
            self.num_output_features,
            self.model_conf,
            self.grid_shape,
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
        y_hat = self.model(inputs)
        return y_hat

    def _shared_forward_step(self, x, y, train=False):
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
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
        x, y = (
            torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1),
            torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1),
        )
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

    def on_train_end(self):
        best_model_state = deepcopy(self.model.state_dict())
        torch.save(best_model_state, HIERA_MASKED_DIR)

    def val_plot_step(self, batch_idx, y, y_hat):
        """Plots images on first batch of validation and log them in tensorboard."""
        if batch_idx == 0:
            tb = self.logger.experiment
            step = self.current_epoch
            # dformat = "CHW"
            if step == 0:
                image = y.permute(0, 3, 1, 2)[0]
                for i in range(image.shape[0]):
                    field_image = image[i]
                    tb.add_image(
                        f"val_plots/true_image_field_{i}", field_image, dataformats="HW"
                    )
                # tb.add_image("val_plots/true_image", y.permute(0, 3, 1, 2)[0], dataformats=dformat)
            image = y.permute(0, 3, 1, 2)[0]
            for i in range(image.shape[0]):
                field_image = image[i]
                tb.add_image(
                    f"val_plots/test_image_field_{i}", field_image, dataformats="HW"
                )
            # tb.add_image("val_plots/test_image", y_hat.permute(0, 3, 1, 2)[0], step, dataformats=dformat)

    def validation_step(self, batch, batch_idx):
        x, y = (
            torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1),
            torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1),
        )
        y_hat, loss = self._shared_forward_step(x, y)
        batch_dict = self._shared_metrics_step(loss, x, y, y_hat, "validation")
        self.log("val_mean_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append(batch_dict)
        self.val_plot_step(batch_idx, y, y_hat)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self._shared_epoch_end(outputs, "validation")
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        """Computes metrics for each sample, at the end of the run."""
        x, y, name = (
            torch.squeeze(batch.inputs.tensor[:, 0, :, :, :], dim=1),
            torch.squeeze(batch.outputs.tensor[:, 0, :, :, :], dim=1),
        )
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
        self.logger.log_hyperparams(metrics=metrics)

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
