"""Tests a trained model.
Takes a path to the checkpoint of a trained model, computes metrics on the valid set and
saves scores plots and forecast animations in the log folder of the model.
You can change the number of auto-regressive steps with option `num_pred_steps` to
make long forecasts.
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.lightning import AutoRegressiveLightning
from py4cast.settings import ROOTDIR


def get_log_dirs(path: Path):
    """Retrieves log folders of the checkpoint's run."""
    log_dir = ROOTDIR / "logs"
    subfolder = path.parent.name
    folder = Path(path.parents[3].name) / path.parents[2].name / path.parents[1].name
    return log_dir, folder, subfolder


parser = ArgumentParser(
    description="Inference on test dataset for weather forecasting."
)
parser.add_argument("ckpt_path", type=Path, help="Path to model checkpoint.")
parser.add_argument(
    "--num_pred_steps",
    type=int,
    default=12,
    help="Number of auto-regressive steps/prediction steps.",
)
parser.add_argument(
    "--only_plots",
    dest="only_plots",
    default=False,
    action="store_true",
    help="Doesn't compute metrics on whole dataset",
)
parser.add_argument(
    "--profiler",
    type=str,
    default="None",
    help="Profiler required. Possibilities are ['simple', 'pytorch', 'None']",
)
args = parser.parse_args()

print(f"Loading model {args.ckpt_path}...")
model = AutoRegressiveLightning.load_from_checkpoint(args.ckpt_path)
model.eval()

# Change number of auto regresive steps for long forecasts
print(f"Changing number of val pred steps to {args.num_pred_steps}...")
hparams = model.hparams["hparams"]
hparams.num_pred_steps_val_test = args.num_pred_steps
hparams.save_path = args.ckpt_path.parent

log_dir, folder, subfolder = get_log_dirs(args.ckpt_path)
logger = TensorBoardLogger(
    save_dir=log_dir, name=folder, version=subfolder, default_hp_metric=False
)
run_id = datetime.now().strftime("%b-%d-%Y-%M-%S")

# Setup profiler
if args.profiler == "pytorch":
    profiler = PyTorchProfiler(
        dirpath=f"{log_dir}/{folder}/{subfolder}",
        filename=f"profile_test_{run_id}",
        export_to_chrome=True,
        profile_memory=True,
    )
    print("Initiate pytorchProfiler")
else:
    profiler = None
    print(f"No profiler set {args.profiler}")

trainer = pl.Trainer(
    logger=logger, devices="auto", profiler=profiler, fast_dev_run=args.only_plots
)

# Initializing data loader
_, val_ds, _ = get_datasets(
    hparams.dataset_name,
    hparams.num_input_steps,
    hparams.num_pred_steps_train,
    hparams.num_pred_steps_val_test,
    hparams.dataset_conf,
)
dataloader = val_ds.torch_dataloader(batch_size=1, num_workers=5, prefetch_factor=2)

print("Testing...")
trainer.test(model=model, dataloaders=val_ds.torch_dataloader(batch_size=1, num_workers=5, prefetch_factor=2))
