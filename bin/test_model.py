from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.lightning import ArLightningHyperParam, AutoRegressiveLightning
from py4cast.settings import ROOTDIR

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
        help="Number of auto-regressive steps/prediction steps. Used only if reset_hparams is True.",
    )
    parser.add_argument(
        "--num_input_steps",
        type=int,
        default=2,
        help="Number of previous timesteps supplied as inputs to the model. Used only if reset_hparams is True.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to load model checkpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="halfunet",
        help="Neural Network architecture trained. Used only if reset_hparams is True.",
    )

    parser.add_argument(
        "--model_conf",
        default=None,
        type=Path,
        help="Configuration file for the model. Used only if reset_hparams is True.",
    )
    parser.add_argument(
        "--reset_hparams",
        action=BooleanOptionalAction,
        default=False,
        help="Reset hyperParameters pass to the lightning module. Be sure of what you are doing.",
    )
    args, other = parser.parse_known_args()

    # Raise an exception if there are unknown arguments
    if other:
        raise ValueError(f"Unknown command line argument(s): {other}")

    _, val_ds, _ = get_datasets(
        args.dataset, args.num_input_steps, 1, args.num_pred_steps, args.dataset_conf
    )
    if args.reset_hparams:
        hp = ArLightningHyperParam(
            dataset_info=val_ds.dataset_info,
            batch_size=1,
            model_name=args.model,
            model_conf=args.model_conf,
            num_input_steps=args.num_input_steps,
            num_pred_steps_val_test=args.num_pred_steps,
            training_strategy="scaled_ar",
            num_inter_steps=4,
        )
    else:
        hp = None

    model = AutoRegressiveLightning.load_from_checkpoint(args.ckpt, hparams=hp)
    model.eval()

    print("Starting Logger")
    logger = TensorBoardLogger(
        save_dir=ROOTDIR / "logs/infer/",
        name="test",
        default_hp_metric=False,
    )
    dl_settings = TorchDataloaderSettings(batch_size=1)
    print("Initialising trainer")
    trainer = pl.Trainer(logger=logger, devices=1)
    print("Testing")
    trainer.test(model=model, dataloaders=val_ds.torch_dataloader(dl_settings))
