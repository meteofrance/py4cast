import datetime as dt
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import xarray as xr
from cfgrib import xarray_to_grib as xtg
from dataclasses_json import dataclass_json

from py4cast.datasets.base import DatasetABC, NamedTensor
from py4cast.forcingutils import compute_hours_of_day


@dataclass_json
@dataclass
class GribSavingSettings:
    template_grib: str
    output_fmt: str
    output_kwargs: list[str]
    directory: str


def saveNamedTensorToGrib(
    pred: NamedTensor,
    ds: DatasetABC,
    sample: Any,
    date: dt.datetime,
    saving_settings: dict,
) -> None:
    """
    Write a named tensor (pred) to grib files, using a prefilled grib file as template.
    The pred data should already be on cpu.
    The template grib should contain all keys necessary to write new data as values.
    Args:
        pred (NamedTensor): the output tensor (pred)
        params (list): list of Param objects used to reference parameters description for writing grib
        leadtimes (list) : list of lead times, as a multiple of prediction steps by the size of a time step (in hours)
        date (dt.datetime) : the date of the initial state
        saving_settings (dict) : settings for the writing process, containing :
            - a template grib path
            - the saving directory
            - the name format of the grib to be written (as fillable f-string with empty placeholders)
            - the keywords to insert in the given name format.
            There should be N-2 keywords, where N is the number of placeholders in the name format.
            The last placeholders are reserved for timestamp.
    """
    if hasattr(ds, "grib_keys_converter"):
        grib_keys, typesOflevel = ds.grib_keys_converter
    else:
        print(
            "Found no custom grib keys converter implemented for dataset, resorting to default grib keys getter"
        )
        params = ds.params
        grib_keys, typesOflevel = get_grib_keys(pred, params)

    grib_groups = get_grib_groups(grib_keys, typesOflevel)
    validtimes = compute_hours_of_day(date, sample.output_terms)
    init_term = compute_hours_of_day(date, [sample.input_terms[-1]])[0]
    leadtimes = validtimes - init_term
    predicted_time_steps = len(leadtimes)
    model_ds = {
        c: xr.open_dataset(
            Path(saving_settings.directory) / saving_settings.template_grib,
            backend_kwargs={
                "indexpath": "",
                "read_keys": [
                    "level",
                    "shortname",
                    "centre",
                    "typeOfGeneratingProcess",
                    "generatingProcessIdentifier",
                    "typeOfLevel",
                    "discipline",
                    "parameterCategory",
                    "parameterNumber",
                    "unit",
                ],
                "filter_by_keys": grib_groups[c],
            },
        )
        for c in grib_groups.keys()
    }

    for t_idx in range(predicted_time_steps)[:1]:
        for group in model_ds.keys():
            target_ds = deepcopy(model_ds[group])

            # if the shape of the dataset grid doesn't match grib template : fill the rest of the data with NaNs

            nanmask, latlon = make_nan_mask(ds, target_ds)
            (
                latmin,
                latmax,
                longmin,
                longmax,
            ) = latlon

            target_ds["time"] = date
            ns_step = np.timedelta64(
                int(leadtimes[t_idx] * 3600 * 1000000000),
                "ns",
            )
            ns_valid = np.timedelta64(
                int(validtimes[t_idx] * 3600 * 1000000000),
                "ns",
            )
            target_ds["step"] = ns_step
            target_ds["valid_time"] = np.datetime64(date) + ns_valid

            for feature_name in pred.feature_names_to_idx.keys():
                name, level, tol = (
                    grib_keys[feature_name]["name"],
                    grib_keys[feature_name]["level"],
                    grib_keys[feature_name]["typeOfLevel"],
                )
                if (name == group) or (level == group) or (tol == group):
                    data = (
                        (
                            pred.tensor[
                                0, t_idx, :, :, pred.feature_names_to_idx[feature_name]
                            ]
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        # TODO : correctly reshape spatial dims in the 1D-catch-all case
                        if pred.num_spatial_dims == 2
                        else (
                            pred.tensor[
                                0, t_idx, :, pred.feature_names_to_idx[feature_name]
                            ]
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    )

                    if nanmask is None:
                        data2grib = data
                    else:
                        data2grib = nanmask
                        data2grib[latmax : latmin + 1, longmin : longmax + 1] = data

                    dims = model_ds[group][name].dims
                    target_ds[name] = (dims, data2grib)
                    target_ds[name] = target_ds[name].assign_attrs(
                        **model_ds[group][name].attrs
                    )

            filename = saving_settings.output_fmt.format(
                *saving_settings.output_kwargs, date, leadtimes[t_idx]
            )
            option = (
                "wb"
                if not os.path.exists(f"{saving_settings.directory}/{filename}")
                else "ab"
            )
            xtg.to_grib(
                target_ds,
                filename,
                option,
            )


def get_grib_keys(pred: NamedTensor, params: list) -> Tuple[dict, list, list, dict]:

    """Match feature names of the pred named tensor to grid-readable levels and parameter names.
    Throw warnings if feature names are found that do not match any parameter (these features will not be saved).
    Args:
        pred (NamedTensor): input named tensor
        params (list): list of parameters, implementing typical Param instances used to describe datasets.


    Returns:
        Dict[str : Dict[str : str]] : dictionary with feature names as keys and levels / names as values
        List[float] : list of levels
        List[str] : list of variable names
    """

    pred_feature_names = pred.feature_names
    grib_keys = {}
    levels = []
    names = []
    typesOflevel = {}
    unmatched_feature_names = set(pred.feature_names)
    for param in params:

        trial_names = param.parameter_short_name
        for ft_idx, ftname in enumerate(trial_names):
            if ftname in pred_feature_names:
                level, name, typeoflevel = (
                    param.levels[ft_idx],
                    param.name,
                    param.level_type,
                )
                grib_keys[ftname] = {
                    "level": level,
                    "name": name,
                    "typeOfLevel": typeoflevel,
                }
                unmatched_feature_names.remove(ftname)
                if level not in levels:
                    levels.append(level)
                if name not in names:
                    names.append(name)
                if typeoflevel not in typesOflevel:
                    typesOflevel[typeoflevel] = {"levels": [level], "names": [name]}
                else:
                    typesOflevel[typeoflevel]["levels"].append(level)
                    typesOflevel[typeoflevel]["names"].append(name)

    if len(unmatched_feature_names) != 0:
        raise UserWarning(
            f"There where unmatched features in pred tensor (no associated param found) : {unmatched_feature_names}"
        )

    return grib_keys, typesOflevel


def get_grib_groups(grib_keys: dict, typesOflevel: dict) -> dict:

    grib_groups = {
        "surface": {
            "cfVarName": [
                grib_keys[ftname]["name"]
                for ftname in grib_keys.keys()
                if grib_keys[ftname]["typeOfLevel"] == "surface"
            ],
            "typeOfLevel": "surface",
        },
    }

    for tol in typesOflevel.keys():
        if tol != "surface":
            levels_tol = typesOflevel[tol]["levels"]
            names_tol = typesOflevel[tol]["names"]

            cribler = levels_tol if len(levels_tol) < len(names_tol) else names_tol
            cribler_flag = "level" if len(levels_tol) < len(names_tol) else names_tol
            for c in cribler:
                if cribler_flag == "level":
                    filter_keys = {
                        "level": [c],
                        "cfVarName": names_tol,
                        "typeOfLevel": tol,
                    }
                else:
                    filter_keys = {
                        "level": levels_tol,
                        "cfVarName": c,
                        "typeOfLevel": tol,
                    }
                grib_groups[c] = filter_keys

    return grib_groups


def make_nan_mask(
    infer_dataset: DatasetABC, template_dataset: xr.Dataset
) -> Tuple[Union[np.ndarray, None], Tuple]:
    if (template_dataset.latitude.values != infer_dataset.grid.lat[0, :]) or (
        template_dataset.longitude.values != infer_dataset.grid.lon[:, 0]
    ):
        nanmask = np.empty(
            (len(template_dataset.latitude), len(template_dataset.longitude))
        )
        nanmask[:] = np.nan
        latmin, latmax = (
            np.where(
                np.round(template_dataset.latitude.values, 5)
                == round(infer_dataset.grid.lat.min(), 5)
            )[0].item(),
            np.where(
                np.round(template_dataset.latitude.values, 5)
                == round(infer_dataset.grid.lat.max(), 5)
            )[0].item(),
        )
        longmin, longmax = (
            np.where(
                np.round(template_dataset.longitude.values, 5)
                == round(infer_dataset.grid.lon.min(), 5)
            )[0].item(),
            np.where(
                np.round(template_dataset.longitude.values, 5)
                == round(infer_dataset.grid.lon.max(), 5)
            )[0].item(),
        )

    else:
        nanmask = None
        longmin, longmax, latmin, latmax = None, None, None, None

    return nanmask, (latmin, latmax, longmin, longmax)
