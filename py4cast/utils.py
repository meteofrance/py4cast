import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import gif
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def nullable_string(val: str):
    if val == "None":
        return None
    return val


def torch_save(data, path: Path):
    """Saving files with torch to be writeable by anyone"""
    if path.exists():
        path.unlink()
    torch.save(data, path)
    path.chmod(0o666)


def torch_load(path: Path, device: str):
    return torch.load(path, map_location=device)


class RegisterFieldsMixin:
    """
    Mixin class to register
    a dataclass fields
    as a lightning buffer.
    See https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
    """

    def register_buffers(self, lm: type[pl.LightningModule], persistent: bool = False):
        """
        Register the dataclass fields which are torch.Tensor as lightning buffers.
        """
        for field in fields(self):
            field_instance = getattr(self, field.name)
            if isinstance(field_instance, torch.Tensor):
                lm.register_buffer(field.name, field_instance, persistent=persistent)


def merge_dicts(d1: Dict, d2: Dict) -> Dict:
    """
    Recursively merge two nested dictionaries.
    """
    for key in d2:
        if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
            merge_dicts(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


str_to_dtype = {
    "bf16-true": torch.bfloat16,
    "16-true": torch.float16,
    "32-true": torch.float32,
    "64-true": torch.float64,
}


@gif.frame
def plot_frame(
    feature_name: str,
    target: np.ndarray,
    predictions: List[np.ndarray],
    proj_name: str,
    subdomain: List[int],
    metadata: dict[str, Any],
    title: str = None,
    models_names: List[str] = None,
) -> None:
    """Plots one frame of the animation."""

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
        feature_str = metadata["WEATHER_PARAMS"][short_name]["long_name"][6:]
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
        nrows=lines, ncols=cols, subplot_kw={"projection": proj_name}
    )
    extent = subdomain
    if isinstance(axes, np.ndarray):
        pass
    else:
        axes = np.array([axes])

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
    runtime: str,
    target: np.ndarray,
    preds: List[np.ndarray],
    models_names: List[str],
    proj_name: str,
    subdomain: List[int],
    metadata: dict[str, Any],
):
    """Make a gifs comparing multiple forecasts of one feature."""

    frames = []
    for t in range(preds[0].shape[0]):
        title = f"{runtime} +{t+1}h"
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
            proj_name,
            subdomain,
            metadata,
            title,
            models_names,
        )
        frames.append(frame)
    return frames
