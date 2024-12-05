from typing import Dict
import lightning.pytorch as pl
from py4cast.datasets import get_datasets


class TitanDataModule(pl.LightningDataModule):
    """
    DataModule adapted to gridded inputs (Titan, Dummy)
    """
    def __init__(
        self,
        dataset_conf: str | None,
        dataset_name: str,
        num_input_steps: int,
        num_pred_steps: int,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int | None,
        pin_memory: bool,
        persistent_workers: bool,
        config_override: Dict | None,
    ):
        super().__init__()
        self.num_input_steps = num_input_steps
        self.num_pred_steps = num_pred_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Get dataset in initialisation to have access to this attribute before method trainer.fit
        self.train_ds, self.val_ds, self.test_ds = get_datasets(
            dataset_name,
            num_input_steps,
            num_pred_steps,
            num_pred_steps,
            dataset_conf,
            config_override,
        )

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

    @property
    def len_train_dl(self):
        return len(self.train_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            )
        )

    @property
    def train_dataset_info(self):
        return self.train_ds.dataset_info
    
    @property
    def batch_shape(self):
        item_batch = next(iter(self.train_dataloader()))
        return item_batch.inputs.tensor.shape

    @property
    def infer_ds(self):
        return self.test_ds