from pathlib import Path

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from pnia.datasets import SmeagolDataset
from pnia.models import HiLAM

path = Path(__file__)

if __name__ == "__main__":
    #    "/home/mrpa/chabotv/pnia/saved_models/hi_lam_full_ds_12h/min_val_loss-v1.ckpt"
    # "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32hi_lam-4x64-021308:46-1307/last.ckpt"
    model = HiLAM.load_from_checkpoint(
        "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32hi_lam-4x64-021415:52-2529/last.ckpt"
    )
    # For debbugging skip plotting
    model.n_example_pred = 1
    model.eval()
    training_dataset, validation_dataset, test_dataset = SmeagolDataset.from_json(
        path.parent.parent / "pnia/xp_conf/smeagol.json",
        args={
            "train": {
                "nb_pred_steps": 1,
                "standardize": True,
            },
            "valid": {"nb_pred_steps": 10, "standardize": True, "subset": 10},
            "test": {"nb_pred_steps": 4, "standardize": True, "subset": 10},
        },
    )
    logger = TensorBoardLogger(
        save_dir="/scratch/shared/pnia/logs/infer/",
        name="test",
        default_hp_metric=False,
    )
    trainer = pl.Trainer(logger=logger)
    trainer.test(model=model, dataloaders=validation_dataset.loader)
    # model(validation_dataset.__getitem__(0))
