import random
import time
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl

# Torch import
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning_fabric.utilities import seed
from torch.utils.data import Dataset

from pnia.datasets import SmeagolDataset, TitanDataset
from pnia.datasets.base import TorchDataloaderSettings
from pnia.datasets.titan import TitanHyperParams

# From the package
from pnia.lightning import ArLightningHyperParam, AutoRegressiveLightning

DATASETS = {
    "titan": (TitanDataset,),
    "smeagol": (SmeagolDataset,),
}


@dataclass
class TrainingParams:
    precision: str = "32"
    val_interval: int = 1
    epochs: int = 200
    devices: int = 1
    profiler: str = "None"
    run_id: int = 1
    no_log: bool = False
    limit_train_batches: Optional[Union[int, float]] = None


def main(
    training_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tp: TrainingParams,
    hp: ArLightningHyperParam,
    dl_settings: TorchDataloaderSettings,
):

    train_loader = training_dataset.torch_dataloader(dl_settings)
    val_loader = val_dataset.torch_dataloader(dl_settings)
    test_loader = test_dataset.torch_dataloader(dl_settings)

    # Instatiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # torch.set_float32_matmul_precision("high") # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    prefix = ""

    run_name = (
        f"{prefix}{str(hp.dataset)}{hp.model_name}-"
        f"{time.strftime('%m%d%H:%M')}-{tp.run_id}"
    )

    callback_list = []
    if not tp.no_log:
        print(f"Model and checkpoint will be saved in saved_models/{run_name}")
        callback_list.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                filename="min_val_loss",
                monitor="val_mean_loss",
                mode="min",
                save_last=True,
            )
        )

    # Logger
    if tp.no_log:
        logger = None
    else:
        logger = TensorBoardLogger(
            save_dir="/scratch/shared/pnia/logs",
            name=f"{args.model}/{args.dataset}",
            default_hp_metric=False,
        )

    if tp.profiler == "pytorch":
        profiler = PyTorchProfiler(
            dirpath=f"/scratch/shared/pnia/logs/{args.model}/{args.dataset}",
            filename=f"profile_{tp.run_id}",
            export_to_chrome=True,
        )
        print("Initiate pytorchProfiler")
    else:
        profiler = None
        print(f"No profiler set {tp.profiler}")

    trainer = pl.Trainer(
        devices=tp.devices,
        num_nodes=1,
        max_epochs=tp.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        logger=logger,
        profiler=profiler,
        log_every_n_steps=1,
        callbacks=callback_list,
        check_val_every_n_epoch=tp.val_interval,
        precision=tp.precision,
        limit_train_batches=tp.limit_train_batches,
        limit_val_batches=tp.limit_train_batches,  # No reason to spend hours on validation if we limit the training.
    )

    lightning_module = AutoRegressiveLightning(hp)

    # Train model
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    trainer.test(ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":

    path = Path(__file__)

    parser = ArgumentParser(description="Train or evaluate models for LAM")
    # For our implementation
    parser.add_argument(
        "--dataset",
        type=str,
        default="titan",
        help="Dataset to use",
        choices=["titan", "smeagol"],
    )
    parser.add_argument(
        "--data_conf",
        type=str,
        default=path.parent.parent / "config" / "smeagol.json",
        help="Configuration file for this dataset. Used only for smeagol dataset right now.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="graph",
        help="Model architecture to train/evaluate (default: graph)",
    )

    parser.add_argument(
        "--model_conf",
        default=None,
        type=Path,
        help="Configuration file for the model.",
    )

    # Old arguments from nlam
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )

    # Training options
    parser.add_argument(
        "--loss", type=str, default="mse", help="Loss function to use (default: mse)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run (default: 1)",
    )

    # General options
    parser.add_argument(
        "--epochs", type=int, default=200, help="upper epoch limit (default: 200)"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Gpus asked.")
    parser.add_argument(
        "--profiler",
        type=str,
        default="None",
        help="Profiler required. Possibilities are ['simple', 'pytorch', 'None']",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load", type=str, help="Path to load model parameters from (default: None)"
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state shoudl be restored with model (default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    parser.add_argument(
        "--memory",
        action=BooleanOptionalAction,
        default=False,
        help="Do we need to save memory ?",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="Number of batches to use for training",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="How many ARSteps do we use ? ",
    )
    parser.add_argument(
        "--no_log",
        action=BooleanOptionalAction,
        default=False,
        help="When activated, log are not stored and models are not saved. Use in dev mode.",
    )
    args, other = parser.parse_known_args()

    # Raise an exception if there are unknown arguments
    if other:
        raise ValueError(f"Unknown command line argument(s): {other}")

    random_run_id = random.randint(0, 9999)
    seed.seed_everything(args.seed)

    # Initialisation du dataset
    if args.dataset == "smeagol":
        training_dataset, validation_dataset, test_dataset = SmeagolDataset.from_json(
            args.data_conf,
            args={
                "train": {
                    "nb_pred_steps": args.steps,
                },
                "valid": {
                    "nb_pred_steps": 5,
                },
                "test": {
                    "nb_pred_steps": 5,
                },
            },
        )

    elif args.dataset == "titan":
        hparams_train_dataset = TitanHyperParams(**{"nb_pred_steps": 1})
        training_dataset = DATASETS["titan"][0](hparams_train_dataset)
        hparams_val_dataset = TitanHyperParams(
            **{"nb_pred_steps": 19, "split": "valid"}
        )
        validation_dataset = DATASETS["titan"][0](hparams_val_dataset)
        hparams_test_dataset = TitanHyperParams(
            **{"nb_pred_steps": 19, "split": "test"}
        )
        test_dataset = DATASETS["titan"][0](hparams_test_dataset)

    hp = ArLightningHyperParam(
        dataset=training_dataset,
        batch_size=args.batch_size,
        model_name=args.model,
        model_conf=args.model_conf,
        lr=args.lr,
        loss=args.loss,
    )

    # Parametre pour le training uniquement
    tp = TrainingParams(
        precision=args.precision,
        val_interval=args.val_interval,
        epochs=args.epochs,
        devices=args.gpus,
        profiler=args.profiler,
        no_log=args.no_log,
        run_id=random_run_id,
        limit_train_batches=args.limit_train_batches,
    )
    dl_settings = TorchDataloaderSettings(batch_size=args.batch_size)
    main(training_dataset, validation_dataset, test_dataset, tp, hp, dl_settings)
