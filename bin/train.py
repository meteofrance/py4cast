import random
import time
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import pytorch_lightning as pl

# Torch import
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning_fabric.utilities import seed

from pnia.datasets import get_datasets
from pnia.datasets.base import DatasetABC, TorchDataloaderSettings

# From the package
from pnia.lightning import ArLightningHyperParam, AutoRegressiveLightning


@dataclass
class TrainerParams:
    precision: str = "32"
    val_interval: int = 1
    epochs: int = 200
    profiler: str = "None"
    run_id: int = 1
    no_log: bool = False
    limit_train_batches: Optional[Union[int, float]] = None


def main(
    tp: TrainerParams,
    hp: ArLightningHyperParam,
    dl_settings: TorchDataloaderSettings,
    datasets: Tuple[DatasetABC, DatasetABC, DatasetABC],
):
    """
    Main function to first train the model
    using the (train, val) datasets and lightning .fit
    and then evaluate the model on the test dataset calling .test
    """
    train_ds, val_ds, test_ds = datasets

    train_loader = train_ds.torch_dataloader(dl_settings)
    val_loader = val_ds.torch_dataloader(dl_settings)
    test_loader = test_ds.torch_dataloader(dl_settings)

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
        f"{prefix}{str(hp.dataset_info.name)}{hp.model_name}-"
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
    if args.load:
        lightning_module = AutoRegressiveLightning.load_from_checkpoint(
            args.load, hparams=hp
        )
    else:
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

    parser = ArgumentParser(
        description="Train Neural networks for weather forecasting."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="titan",
        help="Dataset to use",
        choices=["titan", "smeagol"],
    )
    parser.add_argument(
        "--dataset_conf",
        type=str,  # Union[str, None] # Union does not work from CLI.
        default=None,
        help="Configuration file for the dataset. If None, default configuration is used.",
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
    parser.add_argument(
        "--profiler",
        type=str,
        default="None",
        help="Profiler required. Possibilities are ['simple', 'pytorch', 'None']",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="Number of batches to use for training",
    )
    parser.add_argument(
        "--num_pred_steps_train",
        type=int,
        default=1,
        help="Number of auto-regressive steps/prediction steps during training forward pass",
    )
    parser.add_argument(
        "--num_pred_steps_val_test",
        type=int,
        default=5,
        help="Number of auto-regressive steps/prediction steps during validation and tests",
    )
    parser.add_argument(
        "--num_input_steps",
        type=int,
        default=2,
        help="Number of previous timesteps supplied as inputs to the model",
    )
    parser.add_argument(
        "--no_log",
        action=BooleanOptionalAction,
        default=False,
        help="When activated, log are not stored and models are not saved. Use in dev mode.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load model parameters from (default: None)",
    )
    args, other = parser.parse_known_args()

    # Raise an exception if there are unknown arguments
    if other:
        raise ValueError(f"Unknown command line argument(s): {other}")

    random_run_id = random.randint(0, 9999)
    seed.seed_everything(args.seed)

    datasets = get_datasets(
        args.dataset,
        args.num_input_steps,
        args.num_pred_steps_train,
        args.num_pred_steps_val_test,
        args.dataset_conf,
    )

    hp = ArLightningHyperParam(
        dataset_info=datasets[0].dataset_info,
        batch_size=args.batch_size,
        model_name=args.model,
        model_conf=args.model_conf,
        num_input_steps=args.num_input_steps,
        num_pred_steps_train=args.num_pred_steps_train,
        num_pred_steps_val_test=args.num_pred_steps_val_test,
        lr=args.lr,
        loss=args.loss,
    )

    # Parametre pour le training uniquement
    tp = TrainerParams(
        precision=args.precision,
        val_interval=args.val_interval,
        epochs=args.epochs,
        profiler=args.profiler,
        no_log=args.no_log,
        run_id=random_run_id,
        limit_train_batches=args.limit_train_batches,
    )
    dl_settings = TorchDataloaderSettings(batch_size=args.batch_size)

    main(tp, hp, dl_settings, datasets)
