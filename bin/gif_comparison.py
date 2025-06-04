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

import datetime as dt
import math
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import List, Tuple

import einops
import gif
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from skimage.transform import resize
from tqdm import trange

from py4cast.datasets import get_datasets
from py4cast.datasets.base import DatasetInfo, Item, collate_fn
from py4cast.datasets.titan.settings import AROME_PATH, METADATA
from py4cast.lightning import AutoRegressiveLightning
from py4cast.plots import DomainInfo

PARAMS_INFO = {
    "t2m": {
        "grib_name": "AROME_1S100_ECH0_2M.grib",
        "titan_name": "aro_t2m_2m",
        "cmap": "Spectral_r",
        "vmin": 0,
        "vmax": 40,
        "label": "Température à 2m (C°)",
    },
    "r2": {
        "grib_name": "AROME_1S100_ECH0_2M.grib",
        "titan_name": "aro_r2_2m",
        "cmap": "Spectral",
        "vmin": 0,
        "vmax": 100,
        "label": "Humidité à 2m (%)",
    },
    "tp": {
        "grib_name": "AROME_1S100_ECH1_SOL.grib",
        "titan_name": "aro_tp_0m",
        "cmap": "Spectral_r",
        "vmin": 0.5,
        "vmax": 60,
        "label": "Précipitations (mm)",
    },
    "u10": {
        "grib_name": "AROME_1S100_ECH0_10M.grib",
        "titan_name": "aro_u10_10m",
        "cmap": "RdBu",
        "vmin": -20,
        "vmax": 20,
        "label": "Composante U du vent à 10m (m/s)",
    },
    "v10": {
        "grib_name": "AROME_1S100_ECH0_10M.grib",
        "titan_name": "aro_v10_10m",
        "cmap": "RdBu",
        "vmin": -20,
        "vmax": 20,
        "label": "Composante V du vent à 10m (m/s)",
    },
}


def titan_to_arome_names(titan_name: str) -> str:
    """Converts Titan feature name to Arome feature name."""
    return titan_name.split("_")[1]


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


def get_param(path: Path, param: str, num_steps: int) -> np.ndarray:
    """Extracts a weather param from an AROME forecast in grib."""
    ds = xr.open_dataset(path, engine="cfgrib")
    array = ds[param].values
    if array.shape[0] < num_steps:
        raise ValueError(
            f"The requested leadtimes ({num_steps}h) are not available in the AROME forecast {path}."
        )
    arr_list = [downscale(array[t]) for t in range(num_steps)]
    array = np.stack(arr_list)[:, ::-1]
    return array


def post_process_tp_arome(array: np.ndarray, num_steps: int) -> np.ndarray:
    """Converts AROME precip forecast in mm/h.
    By default, AROME accumulates mm starting from t0."""
    diff_arrs = [array[t + 1] - array[t] for t in range(num_steps)]
    return np.stack(diff_arrs)


def read_arome(date: str, params: List[str], num_steps: int) -> np.ndarray:
    """Extracts several parameters of an AROME forecast."""
    list_arrays = []
    for param in params:
        # For precipitation, we need to extract one more leadtime
        extract_steps = num_steps + 1 if param == "tp" else num_steps
        array = get_param(
            AROME_PATH / date / PARAMS_INFO[param]["grib_name"], param, extract_steps
        )
        if param == "tp":
            array = post_process_tp_arome(array, num_steps)
        list_arrays.append(array)
    return np.stack(list_arrays, axis=-1)


def get_model(ckpt: Path, num_pred_steps: int) -> AutoRegressiveLightning:
    """Loads a model from its checkpoint and changes the nb of forecast steps."""
    model = AutoRegressiveLightning.load_from_checkpoint(ckpt)
    model.num_pred_steps_val_test = num_pred_steps
    model.eval()
    return model


def get_item_for_date(date: str, model: AutoRegressiveLightning) -> Item:
    """Returns an Item containing one sample for a chosen date.
    Date should be in format YYYYMMDDHH.
    """
    config_override = {"periods": {"test": {"start": date, "end": date}}}
    _, _, test_ds = get_datasets(
        model.dataset_name,
        model.num_input_steps,
        model.num_pred_steps_train,
        model.num_pred_steps_val_test,
        model.dataset_conf,
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
    y: torch.tensor,
    feature_names: List[str],
    feature_names_to_idx: dict,
    dataset_info: DatasetInfo,
) -> np.ndarray:
    """Post-processes one forecast by de-normalizing the values of each feature."""
    arrays = []
    for feature_name in feature_names:
        idx_feature = feature_names_to_idx[feature_name]
        mean = dataset_info.stats[feature_name]["mean"]
        std = dataset_info.stats[feature_name]["std"]
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
) -> None:
    """Plots one frame of the animation.
    DEPRECATED: This function has been moved to utils.py.
    To update this script, one should use the imported plot_frame function.
    """

    nb_preds = len(predictions) + 1 if target is not None else len(predictions)
    lines = int(math.sqrt(nb_preds))
    cols = nb_preds // lines
    if nb_preds % lines != 0:
        cols += 1

    param = feature_name.split("_")[1]
    if param in PARAMS_INFO.keys():
        cmap = PARAMS_INFO[param]["cmap"]
        vmin = PARAMS_INFO[param]["vmin"]
        vmax = PARAMS_INFO[param]["vmax"]
        colorbar_label = PARAMS_INFO[param]["label"]
    else:
        cmap = "plasma"
        vmin, vmax = None, None
        short_name = "_".join(feature_name.split("_")[:2])
        feature_str = METADATA["WEATHER_PARAMS"][short_name]["long_name"][6:]
        colorbar_label = (
            f"{feature_str}"  # Units: ({dataset_info.units[feature_name]})"
        )

    if (lines, cols) == (1, 3):
        figsize = (12, 5)
    elif (lines, cols) == (2, 2):
        figsize = (4 * cols, 4 * lines)
    else:
        figsize = (4 * cols, 5 * lines)

    fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=200)
    subfig = fig.subfigures(nrows=1, ncols=1)
    axes = subfig.subplots(
        nrows=lines, ncols=cols, subplot_kw={"projection": domain_info.projection}
    )
    extent = domain_info.grid_limits

    axs = axes.flat
    data_list = [target] + predictions if target is not None else predictions

    for i, data in enumerate(data_list):
        axs[i].coastlines()
        if param == "tp":  # threshold precipitations
            data = np.where(data < 0.5, np.nan, data)
        im = axs[i].imshow(
            data,
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        if models_names:
            axs[i].set_title(models_names[i], size=15)

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
    """Plots a gifs comparing multiple forecasts of one feature.
    DEPRECATED: This function has been moved to utils.py.
    To update this script, one should use the imported make_gif function.
    """
    date = dt.datetime.strptime(date, "%Y%m%d%H")
    date_str = date.strftime("%Y-%m-%d %Hh UTC")

    frames = []
    for t in trange(preds[0].shape[0]):
        title = f"{date_str} +{t+1}h"
        preds_t = [pred[t] for pred in preds]
        target_t = target[t] if target is not None else None
        if feature == "aro_t2m_2m":  # Convert to °C
            if target_t is not None:
                target_t = target_t - 273.15
            preds_t = [pred - 273.15 for pred in preds_t]
        frame = plot_frame(
            feature,
            target_t,
            preds_t,
            domain_info,
            title,
            models_names,
        )
        frames.append(frame)
    filename = f"{date.strftime('%Y-%m-%d_%Hh%M')}_{feature}.gif"
    gif.save(frames, filename, duration=500)


if __name__ == "__main__":
    """
    This script needs to be updated since the refacto.
    """
    parser = ArgumentParser("Plot animations")
    parser.add_argument(
        "--ckpt",
        type=str,
        action="append",
        help="Paths to the model checkpoint or AROME",
        required=True,
    )
    parser.add_argument(
        "--plot_analysis",
        "-pa",
        action=BooleanOptionalAction,
        default=False,
        help="Plot the ground truth ie Arome analysis",
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

    feature_names = [param["titan_name"] for param in PARAMS_INFO.values()]

    y_preds = []
    models_names = []
    for ckpt in args.ckpt:
        if ckpt == "AROME":
            arome_features = [titan_to_arome_names(feat) for feat in feature_names]
            forecast = read_arome(args.date, arome_features, args.num_pred_steps)
            models_names.append("AROME Oper")
        else:
            model = get_model(ckpt, args.num_pred_steps)
            item = get_item_for_date(args.date, model)
            forecast = make_forecast(model, item)
            feature_idx_dict = item.inputs.feature_names_to_idx
            forecast = post_process_outputs(
                forecast, feature_names, feature_idx_dict, model.dataset_info
            )
            models_names.append(f"{model.model_name}\n{model.save_path.name}")
        y_preds.append(forecast)

    y_true = item.outputs.tensor
    y_true = post_process_outputs(
        y_true, feature_names, feature_idx_dict, model.dataset_info
    )
    domain_info = model.dataset_info.domain_info
    if args.plot_analysis:
        models_names = ["AROME Analysis"] + models_names

    for feature_name in feature_names:
        print(feature_name)
        idx_feature = feature_idx_dict[feature_name]
        target_feat = y_true[:, :, :, idx_feature] if args.plot_analysis else None
        list_preds_feat = [pred[:, :, :, idx_feature] for pred in y_preds]
        make_gif(
            feature_name,
            args.date,
            target_feat,
            list_preds_feat,
            models_names,
            domain_info,
        )
