from lightning.pytorch.cli import LightningCLI

from py4cast.datasets import registry as dataset_registry
from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.lightning import (
    ArLightningHyperParam,
    AutoRegressiveLightning,
    PlDataModule,
)
from py4cast.models import registry as model_registry
from py4cast.settings import ROOTDIR



class MyCLI(LightningCLI):
    
    def __init__(self, model_class, datamodule_class):
        super().__init__(model_class, datamodule_class)
    
    # def add_arguments_to_parser(self, parser):
    #     parser.link_arguments("data.train_dataset_info", "model.hparams.dataset_info", apply_on="instantiate")
    #     parser.link_arguments("data.len_train_dl", "model.hparams.len_train_loader", apply_on="instantiate")
    #     pass

def cli_main():
    cli = MyCLI(AutoRegressiveLightning, PlDataModule)


if __name__ == "__main__":
    cli_main()