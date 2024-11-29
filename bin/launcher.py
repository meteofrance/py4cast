"""
Main script to use the model with lightning CLI
Training with fit and infer with predict
Exemple usage:
    - Train
    python bin/launcher.py fit --config config/CLI/trainer.yaml --config config/CLI/poesy.yaml --config config/CLI/halfunet.yaml

    - Inference
    python -m pdb bin/launcher.py predict --ckpt_path /scratch/shared/py4cast/logs/test_cli/last.ckpt --config config/CLI/trainer.yaml --config config/CLI/poesy.yaml --config config/CLI/halfunet.yaml

"""

from lightning.pytorch.cli import LightningCLI

from py4cast.lightning import AutoRegressiveLightning, PlDataModule


class LCli(LightningCLI):
    """
    CLI - Command Line Interface from lightning

    Args:
        A model which inherits from LightningModule
        A datamodule which inherits from LightningDataModule
        save_config_kwargs define if checkpoint should be store even if one is already
        present in the folder, useful for development.
    """

    def __init__(self, model_class, datamodule_class, parser_kwargs):
        super().__init__(
            model_class, datamodule_class, save_config_kwargs={"overwrite": True}, parser_kwargs=parser_kwargs
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


def cli_main():
    LCli(AutoRegressiveLightning, PlDataModule, parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    cli_main()
