import torch
import os
from lightning.pytorch.cli import LightningCLI

from py4cast.lightning import AutoRegressiveLightning, PlDataModule
from lightning.pytorch.callbacks import BasePredictionWriter

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))

# pred_writer = CustomWriter(output_dir="./", write_interval="epoch")

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
