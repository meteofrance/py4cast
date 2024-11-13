from lightning.pytorch.cli import LightningCLI, ArgsType
from py4cast.lightning_data_module.dummy import DummyDataModule, DummyLightningModule


def cli_main(args: ArgsType = None):
    cli = LightningCLI(
        DummyLightningModule,
        DummyDataModule,
        run=False,
        args=args,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    cli_main(["--config=config/lightning_cli/config_cli_test.yaml", "--trainer.fast_dev_run=False"])