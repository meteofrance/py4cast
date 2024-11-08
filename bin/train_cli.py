from lightning.pytorch.cli import LightningCLI

from py4cast.lightning import (
    AutoRegressiveLightning,
    PlDataModule,
)


# install : pip install -U 'jsonargparse[signatures]>=4.27.7'
# Launch : python bin/train_cli.py fit --config config/CLI/test.yaml 

class LCli(LightningCLI):
    def __init__(self, model_class, datamodule_class):
        super().__init__(model_class, datamodule_class, save_config_kwargs={"overwrite": True})

    def add_arguments_to_parser(self, parser):

        parser.link_arguments(
            "data.dataset",
            "model.hparams.dataset_name",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.batch_size",
            "model.hparams.batch_size",
            apply_on="instantiate",
        )
        
        parser.link_arguments(
            "data.train_dataset_info",
            "model.dataset_info",
            apply_on="instantiate",
        )
        
        parser.link_arguments(
            "data.dataset_conf",
            "model.hparams.dataset_conf",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.len_train_dl",
            "model.hparams.len_train_loader",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_input_steps",
            "model.hparams.num_input_steps",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_pred_steps_train",
            "model.hparams.num_pred_steps_train",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_pred_steps_val_test",
            "model.hparams.num_pred_steps_val_test",
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

if __name__ == "__main__":
    cli_main()
