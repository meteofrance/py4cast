from pathlib import Path

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from pnia.datasets import get_datasets

# from pnia.datasets import SmeagolDataset
from pnia.datasets.base import TorchDataloaderSettings
from pnia.lightning import AutoRegressiveLightning

path = Path(__file__)

if __name__ == "__main__":

    # ckpt = "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32graph-031914:46-8984/last.ckpt"
    # )

    train_ds, val_ds, test_ds = get_datasets(
        "smeagol", 2, 1, 17, path.parent.parent / "config" / "smeagol.json"
    )

    ckpt = "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32hilam-040520:02-4832/last.ckpt"

    model = AutoRegressiveLightning.load_from_checkpoint(ckpt)
    # For debbugging skip plotting
    model.hparams.num_samples_to_plot = 1
    # model.eval()

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
    trainer.test(model=model, dataloaders=val_ds.torch_dataloader(dl_settings))
