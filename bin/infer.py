from pathlib import Path

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from pnia.datasets import SmeagolDataset
from pnia.datasets.base import TorchDataloaderSettings
from pnia.lightning import AutoRegressiveLightning

path = Path(__file__)

if __name__ == "__main__":

    # ckpt = "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32graph-031914:46-8984/last.ckpt"
    # )
    ckpt = "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32halfunet-032715:57-9520/last.ckpt"

    model = AutoRegressiveLightning.load_from_checkpoint(ckpt)
    # For debbugging skip plotting
    model.hparams.n_example_pred = 1
    # model.eval()
    training_dataset, validation_dataset, test_dataset = SmeagolDataset.from_json(
        path.parent.parent / "config" / "smeagoldev.json",
        args={
            "train": {
                "nb_pred_steps": 1,
                "standardize": True,
            },
            "valid": {"nb_pred_steps": 3, "standardize": True},
            "test": {"nb_pred_steps": 3, "standardize": True},
        },
    )
    print("Starting Logger")
    logger = TensorBoardLogger(
        save_dir="/scratch/shared/pnia/logs/infer/",
        name="test",
        default_hp_metric=False,
    )
    dl_settings = TorchDataloaderSettings(batch_size=1)
    print("Initialising trainer")
    trainer = pl.Trainer(logger=logger, devices=1)  # ,strategy="ddp")
    print("Testing")
    trainer.test(
        model=model, dataloaders=validation_dataset.torch_dataloader(dl_settings)
    )
