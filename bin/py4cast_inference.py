from lightning.pytorch.cli import LightningCLI, ArgsType
from py4cast.ARLightningModule import AutoRegressiveLightningModule
from py4cast.TitanDataModule import TitanDataModule

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)  # Appeler la méthode parente
        parser.link_arguments("data.dataset_name", "model.dataset_name")
        parser.link_arguments(
            "data.train_dataset_info", "model.dataset_info", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.batch_shape", "model.batch_shape", apply_on="instantiate"
        )

def cli_main(args: ArgsType = None):
    cli = MyLightningCLI(
        AutoRegressiveLightningModule,
        TitanDataModule,
        run=False,
        args=args,
    )
    
    # Exécuter les inférences
    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path="best")

if __name__ == "__main__":
    cli_main(
        [
            "--config=config/config_AR_inference.yaml",
        ]
    )
