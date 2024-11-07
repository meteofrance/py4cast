from lightning.pytorch.cli import LightningCLI, ArgsType
from py4cast.MAE_2_lightning import MAELightningModule, PlDataModule


def cli_main(args: ArgsType = None):
    cli = LightningCLI(
        MAELightningModule,
        PlDataModule,
        run=False,
        args=args,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    cli_main(["--config=bin/config_cli_masked.yaml", "--trainer.fast_dev_run=False"])
