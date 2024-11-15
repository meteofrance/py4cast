import torch
import os
from pathlib import Path
from lightning.pytorch.cli import LightningCLI

from py4cast.lightning import AutoRegressiveLightning, PlDataModule
from lightning.pytorch.callbacks import BasePredictionWriter

from py4cast.io.outputs import GribSavingSettings, save_named_tensors_to_grib
default_config_root = Path(__file__).parents[1] / "config/IO/"

class GribWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def setup(self, trainer, pl_module, stage):
        self.stage = stage
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        if self.stage == 'predict':
            model_ds = trainer.datamodule.infer_ds

            with open(default_config_root / "poesy_grib_settings.json", "r") as f:
                save_settings = GribSavingSettings.schema().loads((f.read()))
                ph = len(save_settings.output_fmt.split("{}")) - 1
                kw = len(save_settings.output_kwargs)
                fi = len(save_settings.sample_identifiers)
                try:
                    assert ph == (fi + kw)
                except AssertionError:
                    raise ValueError(
                        f"Filename fmt has {ph} placeholders,\
                        but {kw} output_kwargs and {fi} sample identifiers."
                    )

            for sample, pred in zip(model_ds.sample_list, predictions):
                save_named_tensors_to_grib(pred, model_ds, sample, save_settings)


class LCli(LightningCLI):
    def __init__(self, model_class, datamodule_class):
        super().__init__(
            model_class, datamodule_class, save_config_kwargs={"overwrite": True}
        )

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.dataset",
            "model.dataset_name",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.dataset_conf",
            "model.dataset_conf",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.batch_size",
            "model.batch_size",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_input_steps",
            "model.num_input_steps",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_pred_steps_train",
            "model.num_pred_steps_train",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_pred_steps_val_test",
            "model.num_pred_steps_val_test",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.train_dataset_info",
            "model.dataset_info",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.len_train_dl",
            "model.len_train_loader",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "model.save_path",
            "trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "model.save_path",
            "trainer.callbacks.init_args.dirpath",
            apply_on="instantiate",
        )


def cli_main():
    LCli(AutoRegressiveLightning, PlDataModule)
