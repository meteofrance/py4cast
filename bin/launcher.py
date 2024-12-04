"""
Main script to use the model with lightning CLI
Training with fit and infer with predict
Exemple usage:
    - Train
    python bin/launcher.py fit --config config/CLI/trainer.yaml --config config/CLI/poesy.yaml --config config/CLI/halfunet.yaml

    - Inference
    python bin/launcher.py predict --ckpt_path /scratch/shared/py4cast/logs/test_cli/last.ckpt --config config/CLI/trainer.yaml --config config/CLI/poesy.yaml --config config/CLI/halfunet.yaml

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

    def __init__(self, model_class, datamodule_class):

        super().__init__(model_class, datamodule_class, save_config_kwargs={"overwrite": True})

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--dev_mode', action='store_true')
        
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

    def before_instantiate_classes(self):
        """
        Modify values if dev_mode, printing the path to the log.
        """
        if hasattr(self.config, 'fit'):
            config = self.config.fit
        elif hasattr(self.config, 'predict'):
            config = self.config.predict

        if config.dev_mode:
            trainer = config.trainer
            trainer.max_epochs = 2
            trainer.limit_train_batches= 20
            trainer.limit_val_batches= 20
            trainer.limit_test_batches= 20
            trainer.logger[0].init_args.version = "dev_run"
        
        dataset = config.data.dataset
        model = config.model.model_name
        for logger in  config.trainer.logger:
            logger.init_args.save_dir += f"/{dataset}/{model}" 
        config.trainer.logger[1].init_args.save_dir+="/mlflow"
        print(f"\nLogs are saved at {config.trainer.logger[0].init_args.save_dir}\n")


def cli_main():
    LCli(AutoRegressiveLightning, PlDataModule)


if __name__ == "__main__":
    cli_main()
