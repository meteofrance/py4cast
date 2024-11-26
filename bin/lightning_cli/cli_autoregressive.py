from lightning.pytorch.cli import LightningCLI, ArgsType
from py4cast.lightning_module.autoregressive import AutoRegressiveLightningModule
from py4cast.data_module.dummy import DummyDataModule

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.train_dataset_info", "model.dataset_info", apply_on="instantiate"
        )
        parser.link_arguments("data.dataset_name", "model.dataset_name")
        parser.link_arguments(
            "data.batch_shape", "model.batch_shape", apply_on="instantiate"
        )


def cli_main(args: ArgsType = None):
    cli = MyLightningCLI(
        AutoRegressiveLightningModule,
        DummyDataModule,
        run=False,
        args=args,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    cli_main(
        [
            "--config=config/lightning_cli/config_cli_autoregressive.yaml",
        ]
    )
