from typing import Union, Dict
from pathlib import Path
import lightning.pytorch as pl

from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings


class TitanDataModule(pl.LightningDataModule):
    """
    DataModule adapted to gridded inputs (Titan, Dummy)
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
        self.dl_settings = TorchDataloaderSettings()
        self.dataset_name = dataset_name
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

    @property
    def batch_shape(self):
        item_batch = next(iter(self.train_dataloader()))
        return item_batch.inputs.tensor.shape
