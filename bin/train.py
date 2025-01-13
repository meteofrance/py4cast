"""
Main script to first train the model
using the (train, val) datasets and lightning .fit
and then evaluate the model on the test dataset calling .test
"""

import getpass
import os
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import mlflow.pytorch
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler
from lightning_fabric.utilities import seed
from mlflow.models.signature import infer_signature

from py4cast.lightning import AutoRegressiveLightning, PlDataModule
from py4cast.models import registry as model_registry
from py4cast.settings import ROOTDIR

layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["mean_loss_epoch/train", "mean_loss_epoch/validation"]],
    },
}

# Variables for multi-nodes multi-gpu training
nb_nodes = int(os.environ.get("SLURM_NNODES", 1))
if nb_nodes > 1:
    gpus_per_node = len(os.environ.get("SLURM_STEP_GPUS", "1").split(","))
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = global_rank - gpus_per_node * (global_rank // gpus_per_node)
    print(
        f"Global rank: {global_rank}, Local rank: {local_rank}, Gpus per node: {gpus_per_node}"
    )
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["GLOBAL_RANK"] = os.environ.get("SLURM_PROCID", 0)
    os.environ["NODE_RANK"] = os.environ.get("SLURM_NODEID", 0)


parser = ArgumentParser(description="Train Neural networks for weather forecasting.")
parser.add_argument(
    "--dataset",
    type=str,
    default="titan",
    help="Dataset to use",
)
parser.add_argument(
    "--dataset_conf",
    type=Path,  # Union[str, None] # Union does not work from CLI.
    default=None,
    help="Configuration file for the dataset. If None, default configuration is used.",
)
parser.add_argument(
    "--model",
    type=str,
    default="UNETRPP",
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
parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

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
    default=1,
    help="Number of auto-regressive steps/prediction steps during validation and tests",
)
parser.add_argument(
    "--num_input_steps",
    type=int,
    default=1,
    help="Number of previous timesteps supplied as inputs to the model",
)
parser.add_argument(
    "--num_inter_steps",
    type=int,
    default=1,
    help="Number of model steps between two samples",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=10,
    help="Number of workers for dataloaders",
)
parser.add_argument(
    "--prefetch_factor",
    type=int,
    default=None,
    help="Number of batches loaded in advance by each worker",
)
parser.add_argument(
    "--mlflow_log",
    action=BooleanOptionalAction,
    default=False,
    help="When activated, the MLFlowLogger is used and the model is saved in the MLFlow style.",
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
    "--resume_from_ckpt",
    type=str,
    default=None,
    help="Path to load the whole training parameters from (default: None)",
)
parser.add_argument(
    "--strategy",
    type=str,
    default="diff_ar",
    help="Strategy for training ('diff_ar', 'scaled_ar')",
)
parser.add_argument(
    "--campaign_name",
    "-cn",
    type=str,
    default="camp0",
    help="Name of folder that regroups runs.",
)
parser.add_argument(
    "--run_name",
    "-rn",
    type=str,
    default="run",
    help="Name of the run.",
)
parser.add_argument(
    "--pin_memory",
    "-pm",
    action=BooleanOptionalAction,
    default=False,
    help="Use pin_memory in dataloader",
)
parser.add_argument(
    "--channels_last",
    "-cl",
    action=BooleanOptionalAction,
    default=False,
    help="Use torch's channel last",
)

args, other = parser.parse_known_args()
username = getpass.getuser()
date = datetime.now()

# Raise an exception if there are unknown arguments
if other:
    raise ValueError(f"Unknown command line argument(s): {other}")

if args.dev_mode:
    args.epochs = 3
    args.limit_train_batches = 3

run_id = date.strftime("%b-%d-%Y-%M-%S")
seed.seed_everything(args.seed)

# Wrap dataset with lightning datamodule
dm = PlDataModule(
    dataset_name=args.dataset,
    num_input_steps=args.num_input_steps,
    num_pred_steps_train=args.num_pred_steps_train,
    num_pred_steps_val_test=args.num_pred_steps_val_test,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    prefetch_factor=args.prefetch_factor,
    pin_memory=args.pin_memory,
    dataset_conf=args.dataset_conf,
    config_override=None,
)

# Get essential info to instantiate ArLightningHyperParam
len_loader = dm.len_train_dl
dataset_info = dm.train_dataset_info

# Setup GPU usage + get len of loader for LR scheduler
if torch.cuda.is_available():
    device_name = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # Allows using Tensor Cores on A100s
    len_loader = len_loader // (torch.cuda.device_count() * nb_nodes)
else:
    device_name = "cpu"
    len_loader = len_loader
# Get Log folders
log_dir = ROOTDIR / "logs"
folder = Path(args.campaign_name) / args.dataset / args.model
run_name = f"{username[:4]}_{args.run_name}"
if args.dev_mode:
    run_name += "_dev"
list_subdirs = list((log_dir / folder).glob(f"{run_name}*"))
list_versions = sorted([int(d.name.split("_")[-1]) for d in list_subdirs])
version = 0 if list_subdirs == [] else list_versions[-1] + 1
subfolder = f"{run_name}_{version}"
save_path = log_dir / folder / subfolder


# Logger & checkpoint callback
callback_list = []
if args.no_log:
    loggers = None
else:
    loggers = {
        "TensorBoardLogger": TensorBoardLogger(
            save_dir=log_dir, name=folder, version=subfolder, default_hp_metric=False
        ),
    }

    if args.mlflow_log:
        mlflow_logger = {
            "MLFlowLogger": MLFlowLogger(
                experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", str(folder)),
                run_name=subfolder,
                log_model=True,
                save_dir=log_dir / "mlflow",
            )
        }
        loggers.update(mlflow_logger)

    print(
        "--> Model, checkpoints, and tensorboard artifacts "
        + f"will be saved in {save_path}."
    )

    loggers["TensorBoardLogger"].experiment.add_custom_scalars(layout)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="{epoch:02d}-{val_mean_loss:.2f}",  # Custom filename pattern
        monitor="val_mean_loss",
        mode="min",
        save_top_k=1,  # Save only the best model
        save_last=True,  # Also save the last model
    )
    callback_list.append(checkpoint_callback)
    callback_list.append(LearningRateMonitor(logging_interval="step"))
    callback_list.append(
        EarlyStopping(monitor="val_mean_loss", mode="min", patience=50)
    )

# Setup profiler
if args.profiler == "pytorch":
    profiler = PyTorchProfiler(
        dirpath=ROOTDIR / f"logs/{args.model}/{args.dataset}",
        filename=f"torch_profile_{run_id}",
        export_to_chrome=True,
        profile_memory=True,
    )
    print("Initiate pytorchProfiler")
elif args.profiler == "advanced":
    profiler = AdvancedProfiler(
        dirpath=ROOTDIR / f"logs/{args.model}/{args.dataset}",
        filename=f"advanced_profile_{run_id}",
        line_count_restriction=50,  # Display top 50 lines
    )
elif args.profiler == "simple":
    profiler = args.profiler
else:
    profiler = None
    print(f"No profiler set {args.profiler}")

trainer = pl.Trainer(
    num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
    devices="auto",
    max_epochs=args.epochs,
    deterministic=True,
    strategy="ddp",
    accumulate_grad_batches=10,
    accelerator=device_name,
    logger=loggers.values(),
    profiler=profiler,
    log_every_n_steps=1,
    callbacks=callback_list,
    check_val_every_n_epoch=args.val_interval,
    precision=args.precision,
    limit_train_batches=args.limit_train_batches,
    limit_val_batches=args.limit_train_batches,  # No reason to spend hours on validation if we limit the training.
    limit_test_batches=args.limit_train_batches,
)

dict_args = {
    "dataset_name": args.dataset,
    "dataset_conf": args.dataset_conf,
    "batch_size": args.batch_size,
    "model_name": args.model,
    "model_conf": args.model_conf,
    "num_input_steps": args.num_input_steps,
    "num_pred_steps_train": args.num_pred_steps_train,
    "num_pred_steps_val_test": args.num_pred_steps_val_test,
    "num_inter_steps": args.num_inter_steps,
    "lr": args.lr,
    "loss_name": args.loss,
    "training_strategy": args.strategy,
    "len_train_loader": len_loader,
    "channels_last": args.channels_last,
    "dataset_info": dataset_info,
}

if args.load_model_ckpt and not args.resume_from_ckpt:
    lightning_module = AutoRegressiveLightning.load_from_checkpoint(
        args.load_model_ckpt, **dict_args
    )
else:
    lightning_module = AutoRegressiveLightning(**dict_args)

# Train model
print("Starting training !")
trainer.fit(model=lightning_module, datamodule=dm, ckpt_path=args.resume_from_ckpt)

if not args.no_log:
    # If we saved a model, we test it.
    best_checkpoint = checkpoint_callback.best_model_path
    last_checkpoint = checkpoint_callback.last_model_path

    model_to_test = best_checkpoint if best_checkpoint else last_checkpoint
    print(
        f"Testing using {'best' if best_checkpoint else 'last'} model at {model_to_test}"
    )
    trainer.test(ckpt_path=model_to_test, datamodule=dm)

# Finally log the model in a MLFlow fashion
if trainer.is_global_zero and args.mlflow_log:
    # Get random sample to infer the signature of the model
    dataloader = dm.test_dataloader()
    data = next(iter(dataloader))
    signature = infer_signature(
        data.inputs.tensor.detach().numpy(), data.outputs.tensor.detach().numpy()
    )

    # Manually log the trained model in Mlflow style with its signature
    run_id = loggers["MLFlowLogger"].version
    with mlflow.start_run(run_id=run_id):
        mlflow.pytorch.log_model(
            pytorch_model=trainer.model, artifact_path="model", signature=signature
        )
