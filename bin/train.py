import os
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

from py4cast.datasets import get_datasets
from py4cast.datasets import registry as dataset_registry
from py4cast.datasets.base import DatasetABC, TorchDataloaderSettings
from py4cast.lightning import ArLightningHyperParam, AutoRegressiveLightning
from py4cast.models import registry as model_registry
from py4cast.settings import ROOTDIR


@dataclass
class TrainerParams:
    precision: str = "32"
    val_interval: int = 1
    epochs: int = 200
    profiler: str = "None"
    run_id: int = 1
    no_log: bool = False
    dev_mode: bool = False
    limit_train_batches: Optional[Union[int, float]] = None

    def __post_init__(self):
        if self.dev_mode:
            self.epochs = 3
            self.limit_train_batches = 3


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
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="{epoch:02d}-{val_mean_loss:.2f}",  # Custom filename pattern
            monitor="val_mean_loss",
            mode="min",
            save_top_k=1,  # Save only the best model
            save_last=True,  # Also save the last model
        )
        callback_list.append(checkpoint_callback)

    # Logger
    if tp.no_log:
        logger = None
    else:
        logger = TensorBoardLogger(
            save_dir=ROOTDIR / "logs",
            name=f"{args.model}/{args.dataset}",
            default_hp_metric=False,
        )

    if tp.profiler == "pytorch":
        profiler = PyTorchProfiler(
            dirpath=ROOTDIR / f"logs/{args.model}/{args.dataset}",
            filename=f"profile_{tp.run_id}",
            export_to_chrome=True,
        )
        print("Initiate pytorchProfiler")
    else:
        profiler = None
        print(f"No profiler set {tp.profiler}")

    trainer = pl.Trainer(
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        devices="auto",
        max_epochs=tp.epochs,
        deterministic=True,
        strategy="ddp",
        accumulate_grad_batches=10,
        accelerator=device_name,
        logger=logger,
        profiler=profiler,
        log_every_n_steps=1,
        callbacks=callback_list,
        check_val_every_n_epoch=tp.val_interval,
        precision=tp.precision,
        limit_train_batches=tp.limit_train_batches,
        limit_val_batches=tp.limit_train_batches,  # No reason to spend hours on validation if we limit the training.
        limit_test_batches=tp.limit_train_batches,
    )
    if args.load_model_ckpt:
        lightning_module = AutoRegressiveLightning.load_from_checkpoint(
            args.load_model_ckpt, hparams=hp
        )
    else:
        lightning_module = AutoRegressiveLightning(hp)

    # Train model
    print("Starting training !")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    if not tp.no_log:
        # If we have saved a model, we test it.
        best_checkpoint = checkpoint_callback.best_model_path
        last_checkpoint = checkpoint_callback.last_model_path

        model_to_test = best_checkpoint if best_checkpoint else last_checkpoint
        print(
            f"Testing using {'best' if best_checkpoint else 'last'} model at {model_to_test}"
        )
        trainer.test(ckpt_path=model_to_test, dataloaders=test_loader)


if __name__ == "__main__":

    # Variables for multi-nodes multi-gpu training
    if int(os.environ.get("SLURM_NNODES", 1)) > 1:
        gpus_per_node = len(os.environ.get("SLURM_STEP_GPUS", "1").split(","))
        global_rank = int(os.environ.get("SLURM_PROCID", 0))
        local_rank = global_rank - gpus_per_node * (global_rank // gpus_per_node)
        print(
            f"Global rank: {global_rank}, Local rank: {local_rank}, Gpus per node: {gpus_per_node}"
        )
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["GLOBAL_RANK"] = os.environ.get("SLURM_PROCID", 0)
        os.environ["NODE_RANK"] = os.environ.get("SLURM_NODEID", 0)

    parser = ArgumentParser(
        description="Train Neural networks for weather forecasting."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="titan",
        help="Dataset to use",
        choices=dataset_registry.keys(),
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
        choices=model_registry.keys(),
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
        "--num_inter_steps",
        type=int,
        default=1,
        help="Number of model steps between two samples",
    )
    parser.add_argument(
        "--no_log",
        action=BooleanOptionalAction,
        default=False,
        help="When activated, log are not stored and models are not saved. Use in dev mode.",
    )
    parser.add_argument(
        "--dev_mode",
        action=BooleanOptionalAction,
        default=False,
        help="When activated, reduce number of epoch and steps.",
    )
    parser.add_argument(
        "--load_model_ckpt",
        type=str,
        default=None,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="diff_ar",
        help="Strategy for training ('diff_ar', 'scaled_ar')",
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
        num_inter_steps=args.num_inter_steps,
        lr=args.lr,
        loss=args.loss,
        training_strategy=args.strategy,
    )

    # Parametre pour le training uniquement
    tp = TrainerParams(
        precision=args.precision,
        val_interval=args.val_interval,
        epochs=args.epochs,
        profiler=args.profiler,
        no_log=args.no_log,
        dev_mode=args.dev_mode,
        run_id=random_run_id,
        limit_train_batches=args.limit_train_batches,
    )
    dl_settings = TorchDataloaderSettings(batch_size=args.batch_size)

    main(tp, hp, dl_settings, datasets)
