from lightning.pytorch.cli import LightningCLI

from py4cast.MAE_lightning import MAELightningModule
from py4cast.datasets.titan.__init__ import TitanDataset

def cli_main():
    cli = LightningCLI(MAELightningModule, BoringDataModule)


if __name__ == "__main__":
    cli_main()
