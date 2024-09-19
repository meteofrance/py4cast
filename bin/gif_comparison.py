"""Plots animations comparing forecasts of multiple models with ground truth.
Warnings - For now this script only works with models trained with Titan dataset.
         - If you want to use AROME as a model, you have to manually download the forecast before.

usage: gif_comparison.py [-h] --ckpt CKPT --date DATE [--num_pred_steps NUM_PRED_STEPS]

options:
  -h, --help            show this help message and exit
  --ckpt CKPT           Paths to the model checkpoint or AROME
  --date DATE           Date for inference. Format YYYYMMDDHH.
  --num_pred_steps NUM_PRED_STEPS
                        Number of auto-regressive steps/prediction steps.

example: python bin/gif_comparison.py --ckpt AROME --ckpt /.../logs/my_run/epoch=247.ckpt
                                      --date 2023061812 --num_pred_steps 10
"""

import argparse


from py4cast.datasets import get_datasets
from py4cast.datasets.base import collate_fn, Item
from py4cast.lightning import AutoRegressiveLightning, ArLightningHyperParam
from typing import List, Tuple
import torch
from py4cast.plots import DomainInfo
from tqdm import trange
import einops
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import gif
import datetime as dt
from skimage.transform import resize
import xarray as xr

from py4cast.datasets.titan.settings import METADATA, AROME_PATH

COLORMAPS = {
    "t2m": {"cmap": "Spectral_r", "vmin": 240, "vmax": 320},
    "r2": {"cmap": "Spectral", "vmin": 0, "vmax": 100},
    "tp": {"cmap": "Spectral_r", "vmin": 0.5, "vmax": 100},
    "u10": {"cmap": "RdBu", "vmin": -20, "vmax": 20},
    "v10": {"cmap": "RdBu", "vmin": -20, "vmax": 20},
}


def downscale(
    array: np.ndarray,
    grid: str = "PAAROME_1S40",
    domain: Tuple[int] = [100, 612, 240, 880],
) -> np.ndarray:
    """Downscales an array from Titan grid 1S100 to another grid and subdomain."""
    grid_info = METADATA["GRIDS"][grid]
    array = resize(array, grid_info["size"], anti_aliasing=True)
    array = array[domain[0] : domain[1], domain[2] : domain[3]]
    return array


def get_param(path: Path, param: str) -> np.ndarray:
    """Extracts a weather param from an AROME forecast in grib."""
    ds = xr.open_dataset(path, engine="cfgrib")
    array = ds[param].values
    arr_list = [downscale(array[t]) for t in range(array.shape[0])]
    array = np.stack(arr_list)[:, ::-1]
    return array


def post_process_tp_arome(array: np.ndarray) -> np.ndarray:
    """Converts AROME precip forecast in mm/h.
    By default, AROME accumulates mm starting from t0."""
    diff_arrs = [array[t + 1] - array[t] for t in range(12)]
    return np.stack(diff_arrs)


def read_arome(date: str) -> Tuple[np.ndarray]:
    """Extracts 5 parameters (t2m, r2, tp, u10, v10) of an AROME forecast."""
    path = AROME_PATH / date
    r2 = get_param(path / "AROME_1S100_ECH0_2M.grib", "r2")
    t2m = get_param(path / "AROME_1S100_ECH0_2M.grib", "t2m")
    u10 = get_param(path / "AROME_1S100_ECH0_10M.grib", "u10")
    v10 = get_param(path / "AROME_1S100_ECH0_10M.grib", "v10")
    tp = get_param(path / "AROME_1S100_ECH1_SOL.grib", "tp")
    tp = post_process_tp_arome(tp)
    return t2m, r2, tp, u10, v10


def get_model_and_hparams(
    ckpt: Path, num_pred_steps: int
) -> Tuple[AutoRegressiveLightning, ArLightningHyperParam]:
    """Loads a model from its checkpoint and changes the nb of forecast steps."""
    model = AutoRegressiveLightning.load_from_checkpoint(ckpt)
    hparams = model.hparams["hparams"]
    hparams.num_pred_steps_val_test = num_pred_steps
    model.eval()
    return model, hparams


def get_item_for_date(date: str, hparams: ArLightningHyperParam) -> Item:
    """Returns an Item containing one sample for a chosen date.
    Date should be in format YYYYMMDDHH.
    """
    config_override = {"periods": {"test": {"start": date, "end": date}}}
    _, _, test_ds = get_datasets(
        hparams.dataset_name,
        hparams.num_input_steps,
        hparams.num_pred_steps_train,
        hparams.num_pred_steps_val_test,
        hparams.dataset_conf,
        config_override=config_override,
    )
    item = test_ds[0]
    return item


def make_forecast(model: AutoRegressiveLightning, item: Item) -> torch.tensor:
    """Applies a model an Item to make a forecast."""
    batch_item = collate_fn([item])
    preds = model(batch_item)
    forecast = preds.tensor
    # Here we reshape output from GNNS to be on the grid
    if preds.num_spatial_dims == 1:
        forecast = einops.rearrange(
            forecast, "b t (x y) n -> b t x y n", x=model.grid_shape[0]
        )
    return forecast[0]


def post_process_outputs(
    y: torch.tensor, feature_names: List[str], feature_names_to_idx: dict
) -> np.ndarray:
    """Post-processes one forecast by de-normalizing the values of each feature."""
    arrays = []
    for feature_name in feature_names:
        idx_feature = feature_names_to_idx[feature_name]
        mean = hparams.dataset_info.stats[feature_name]["mean"]
        std = hparams.dataset_info.stats[feature_name]["std"]
        y_norm = (y[:, :, :, idx_feature] * std + mean).cpu().detach().numpy()
        arrays.append(y_norm)
    return np.stack(arrays, axis=-1)


@gif.frame
def plot_frame(
    feature_name: str,
    target: np.ndarray,
    predictions: List[np.ndarray],
    domain_info: DomainInfo,
    title: str = None,
    models_names: List[str] = None,
    colorbar_label: str = None,
    vmin: float = None,
    vmax: float = None,
) -> None:
    """Plots one frame of the animation."""

    nb_preds = len(predictions) + 1
    lines = int(math.sqrt(nb_preds))
    cols = nb_preds // lines
    if nb_preds % lines != 0:
        cols += 1

    param = feature_name.split("_")[1]
    if param in COLORMAPS.keys():
        cmap = COLORMAPS[param]["cmap"]
        vmin = COLORMAPS[param]["vmin"]
        vmax = COLORMAPS[param]["vmax"]
    else:
        cmap = "plasma"

    fig = plt.figure(constrained_layout=True, figsize=(4 * cols, 5 * lines), dpi=200)
    subfig = fig.subfigures(nrows=1, ncols=1)
    axes = subfig.subplots(
        nrows=lines, ncols=cols, subplot_kw={"projection": domain_info.projection}
    )
    extent = domain_info.grid_limits

    for i, data in enumerate([target] + predictions):
        axes[i].coastlines()
        if param == "tp":  # precipitations
            data = np.where(data < 0.5, np.nan, data)
        im = axes[i].imshow(
            data,
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        if models_names:
            axes[i].set_title(models_names[i], size=15)

    subfig.colorbar(im, ax=axes, location="bottom", label=colorbar_label, aspect=40)

    if title:
        fig.suptitle(title, size=20)

    copyright = "Météo-France, Py4cast project."
    fig.text(0, 0.02, copyright, fontsize=8, ha="left")


def make_gif(
    feature: str,
    date: str,
    target: np.ndarray,
    preds: List[np.ndarray],
    models_names: List[str],
    domain_info: DomainInfo,
):
    """Plots a gifs comparing multiple forecasts of one feature."""
    vmin, vmax = target.min(), target.max()
    date = dt.datetime.strptime(date, "%Y%m%d%H")
    date_str = date.strftime("%Y-%m-%d %Hh UTC")
    short_name = "_".join(feature.split("_")[:2])
    feature_str = METADATA["WEATHER_PARAMS"][short_name]["long_name"][6:]
    unit = f"{feature_str} ({hparams.dataset_info.units[feature]})"

    frames = []
    for t in trange(target.shape[0]):
        title = f"{date_str} +{t+1}h"
        preds_t = [pred[t] for pred in preds]
        frame = plot_frame(
            feature,
            target[t],
            preds_t,
            domain_info,
            title,
            models_names,
            unit,
            vmin,
            vmax,
        )
        frames.append(frame)
    gif.save(frames, f"{args.date}_{feature}.gif", duration=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot animations")
    parser.add_argument(
        "--ckpt",
        type=str,
        action="append",
        help="Paths to the model checkpoint or AROME",
        required=True,
    )
    parser.add_argument(
        "--date", type=str, help="Date for inference. Format YYYYMMDDHH.", required=True
    )
    parser.add_argument(
        "--num_pred_steps",
        type=int,
        default=12,
        help="Number of auto-regressive steps/prediction steps.",
    )
    args = parser.parse_args()

    feature_names = [
        "aro_t2m_2m",
        "aro_r2_2m",
        "aro_tp_0m",
        "aro_u10_10m",
        "aro_v10_10m",
    ]

    y_preds = []
    models_names = []
    for ckpt in args.ckpt:
        if ckpt == "AROME":
            t2m, r2, tp, u10, v10 = read_arome(args.date)
            forecast = np.stack([t2m, r2, tp, u10, v10], axis=-1)
            models_names.append("AROME Oper")
        else:
            model, hparams = get_model_and_hparams(ckpt, args.num_pred_steps)
            item = get_item_for_date(args.date, hparams)
            forecast = make_forecast(model, item)
            feature_idx_dict = item.inputs.feature_names_to_idx
            forecast = post_process_outputs(forecast, feature_names, feature_idx_dict)
            models_names.append(f"{hparams.model_name}\n{hparams.save_path.name}")
        y_preds.append(forecast)

    y_true = item.outputs.tensor
    y_true = post_process_outputs(y_true, feature_names, feature_idx_dict)
    domain_info = hparams.dataset_info.domain_info
    models_names = ["AROME Analysis"] + models_names

    print(f"Model {models_names[0]} - shape {y_true.shape}")
    for i in range(len(y_preds)):
        print(f"Model {models_names[i+1]} - shape {y_preds[i].shape}")

    for feature_name in feature_names:
        print(feature_name)
        idx_feature = feature_idx_dict[feature_name]
        target_feat = y_true[:, :, :, idx_feature]
        list_preds_feat = [pred[:, :, :, idx_feature] for pred in y_preds]
        make_gif(
            feature_name,
            args.date,
            target_feat,
            list_preds_feat,
            models_names,
            domain_info,
        )
