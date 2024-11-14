from lightning.pytorch.cli import LightningCLI, ArgsType
from py4cast.lightning_module.mae import MAELightningModule
from py4cast.data_module.dummy import DummyDataModule


def cli_main(args: ArgsType = None):
    cli = LightningCLI(
        MAELightningModule,
        DummyDataModule,
        run=False,
        args=args,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    cli_main(
        [
            "--config=config/lightning_cli/config_cli_mae.yaml",
            "--trainer.fast_dev_run=False",
        ]
    )
