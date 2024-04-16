from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from pnia.datasets import get_datasets
from pnia.datasets.base import TorchDataloaderSettings
from pnia.lightning import ArLightningHyperParam, AutoRegressiveLightning

path = Path(__file__)

if __name__ == "__main__":

    parser = ArgumentParser(
        description="Inference on validation dataset for weather forecasting."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="smeagol",
        help="Dataset to use",
        choices=["titan", "smeagol"],
    )
    parser.add_argument(
        "--dataset_conf",
        type=str,
        default=None,
        help="Configuration file for the dataset. If None, default configuration is used.",
    )
    parser.add_argument(
        "--num_pred_steps",
        type=int,
        default=17,
        help="Number of auto-regressive steps/prediction steps",
    )
    parser.add_argument(
        "--num_input_steps",
        type=int,
        default=2,
        help="Number of previous timesteps supplied as inputs to the model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to load model parameters from",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="halfunet",
        help="Neural Network architecture to train",
    )

    parser.add_argument(
        "--model_conf",
        default=None,
        type=Path,
        help="Configuration file for the model. If None default model settings are used.",
    )
    args, other = parser.parse_known_args()

    # Raise an exception if there are unknown arguments
    if other:
        raise ValueError(f"Unknown command line argument(s): {other}")
    # ckpt = "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32graph-031914:46-8984/last.ckpt"
    # )

    _, val_ds, _ = get_datasets(
        args.dataset, args.num_input_steps, 1, args.num_pred_steps, args.dataset_conf
    )

    ckpt = "/home/mrpa/chabotv/pnia/saved_models/smeagol_franmgsp32hilam-040520:02-4832/last.ckpt"
    print(ckpt, args.ckpt)

    hp = ArLightningHyperParam(
        dataset_info=val_ds.dataset_info,
        batch_size=1,
        model_name=args.model,
        model_conf=args.model_conf,
        num_input_steps=args.num_input_steps,
        num_pred_steps_val_test=args.num_pred_steps,
        num_inter_steps=1,
    )

    model = AutoRegressiveLightning.load_from_checkpoint(args.ckpt, hparams=hp)
    model.eval()

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
