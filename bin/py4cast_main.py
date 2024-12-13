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
from py4cast.arlightningmodule import AutoRegressiveLightningModule
from py4cast.titandatamodule import TitanDataModule

class MyLightningCLI(LightningCLI):
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
            model_class,
            datamodule_class,
            save_config_kwargs={"overwrite": True},
            parser_kwargs=parser_kwargs
        )

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.dataset_name",
            "model.dataset_name",
        )
        parser.link_arguments(
            "data.train_dataset_info",
            "model.dataset_info",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.batch_shape",
            "model.batch_shape",
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
            "data.num_pred_steps",
            "model.num_pred_steps",
            apply_on="instantiate",
        ) 
        parser.link_arguments(
            "data.len_train_dl",
            "model.len_train_loader",
            apply_on="instantiate",
        )

    # def before_instantiate_classes(self):
    #     """
    #     Correct path of logs to add dataset/model folder
    #     """
    #     # Get correct config
    #     if hasattr(self.config, "fit"):
    #         config = self.config.fit
    #     elif hasattr(self.config, "predict"):
    #         config = self.config.predict
    #     else:
    #         config = self.config

    #     # Add the dataset and the model to the path
    #     dataset = config.data.dataset
    #     model = config.model.model_name
    #     for logger in config.trainer.logger:
    #         logger.init_args.save_dir += f"/{dataset}/{model}"
    #         # Create a specific folder for mlflow
    #         if logger.class_path == "lightning.pytorch.loggers.MLFlowLogger":
    #             logger.init_args.save_dir += "/mlflow"

    #     print(f"\nLogs are saved at {config.trainer.logger[0].init_args.save_dir}\n")

def cli_main():
    cli = MyLightningCLI(
        AutoRegressiveLightningModule,
        TitanDataModule,
        parser_kwargs={"parser_mode": "yaml"},
    )

if __name__ == "__main__":
    cli_main()
