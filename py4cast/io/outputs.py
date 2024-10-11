import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from cfgrib import xarray_to_grib as xtg
from dataclasses_json import dataclass_json

from py4cast.datasets.base import DatasetABC, NamedTensor
from py4cast.forcingutils import compute_hours_of_day


@dataclass_json
@dataclass
class GribSavingSettings:
    """
    - a template grib path
    - the saving directory
    - the name format of the grib to be written (as fillable f-string with empty placeholders)
    - the keywords to insert in the given name format.
    - the sample identifiers to insert in the given name format.
    There should be p-i keywords, where p is the number of placeholders and i the number of sample identifiers.
    """

    template_grib: str
    directory: str
    output_kwargs: tuple[str, ...] = ()
    sample_identifiers: tuple[str, ...] = ("date", "leadtime")
    output_fmt: str = "grid.forecast_ai_date_{}_ech_{}.json"


def save_named_tensors_to_grib(
    pred: NamedTensor,
    ds: DatasetABC,
    sample: Any,
    saving_settings: dict,
) -> None:
    """
    Write a named tensor (pred) to grib files, using a prefilled grib file as template.
    The pred data should already be on cpu.
    The template grib should contain all keys necessary to write new data as values.
    Args:
        pred (NamedTensor): the output tensor (pred)
        ds (DatasetABC): dataset containing all information's needed to describe samples and terms
        sample (Any) : an instance of the DataSet Sample, containing informations on parameters, date, leadtimes
        date (dt.datetime) : the date of the initial state
        saving_settings (GribaSavingSettings) : settings for the writing process

    """

    params = ds.params

    grib_features = get_grib_param_dataframe(pred, params)
    grib_groups = get_grib_groups(grib_features)
    validtimes = compute_hours_of_day(sample.date, sample.output_terms)
    init_term = compute_hours_of_day(sample.date, [sample.input_terms[-1]])[0]
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
            raw_data = pred.select_dim("timestep", t_idx, bare_tensor=False)
            storable = write_storable_dataset(
                pred,
                ds,
                model_ds[group],
                group,
                sample,
                validtimes[t_idx],
                leadtimes[t_idx],
                raw_data,
                grib_features,
            )
            filename = get_output_filename(saving_settings, sample, leadtimes[t_idx])
            option = (
                "wb"
                if not os.path.exists(f"{saving_settings.directory}/{filename}")
                else "ab"
            )
            xtg.to_grib(
                storable,
                Path(saving_settings.directory) / filename,
                option,
            )


def write_storable_dataset(
    pred: NamedTensor,
    ds: DatasetABC,
    template_ds: xr.Dataset,
    group: str,
    sample: Any,
    validtime: float,
    leadtime: float,
    raw_data: NamedTensor,
    grib_features: pd.DataFrame,
) -> xr.Dataset:
    """Write the template xarray dataset with raw data tensor from inference

    Args:
        pred (NamedTensor): complete namedtensor, containing feature names
        ds (DatasetABC): inference dataset
        template_ds (xr.Dataset): xarray dataset extracted from the template grib
        group (str): index of the template_ds in the template dict containing coherent groups
        sample (Any): the inference sample to be saved
        validtime (float): time of validity of the current sample
        leadtime (float): lead time of the current sample
        raw_data (NamedTensor): extraction from pred at current timestep
        grib_features (pd.DataFrame): complete description of feature names and definition

    Returns:
        xr.Dataset: the template dataset, filled with data from raw_data, in the correct format
    """
    receiver_ds = deepcopy(template_ds)

    # if the shape of the dataset grid doesn't match grib template, fill the rest of the data with NaNs
    nanmask, latlon = make_nan_mask(ds, receiver_ds)
    (
        latmin,
        latmax,
        longmin,
        longmax,
    ) = latlon

    receiver_ds["time"] = sample.date
    ns_step = np.timedelta64(
        int(leadtime * 3600 * 1000000000),
        "ns",
    )
    ns_valid = np.timedelta64(
        int(validtime * 3600 * 1000000000),
        "ns",
    )
    receiver_ds["step"] = ns_step
    receiver_ds["valid_time"] = np.datetime64(sample.date) + ns_valid

    # retrieving key metadata to be able to parallelize writing
    namelist = list(receiver_ds.keys())
    used_grib_feat = grib_features[(grib_features["name"].isin(namelist))]

    # will only be used if namelist has a single item
    name = namelist[0]
    feature_names = used_grib_feat["feature_name"].tolist()
    tol = used_grib_feat["typeOfLevel"].drop_duplicates().tolist()[0]
    feature_idx = torch.tensor([pred.feature_names_to_idx[f] for f in feature_names])

    data = raw_data.index_select_dim("features", feature_idx).squeeze().cpu().numpy()

    if f"{name}_{tol}" == group:
        # there might be a third dimension (eg isobaricInhPa) : basis for nanmask duplication
        dims = template_ds.dims

        try:
            # supplementary dim
            maybe_repeat = template_ds.sizes[tol]
            data2grib = np.repeat(nanmask[np.newaxis], maybe_repeat, axis=0)
            data2grib[:, latmax : latmin + 1, longmin : longmax + 1] = data
            receiver_ds[name] = (dims, data2grib.astype(np.float32))
            receiver_ds[name] = receiver_ds[name].assign_attrs(
                **template_ds[name].attrs
            )

        except KeyError:

            maybe_repeat = len(feature_idx) if len(feature_idx) > 1 else 0
            if maybe_repeat:
                # no suplementary dim but several variables
                data2grib = np.repeat(nanmask[np.newaxis], maybe_repeat, axis=0)
                data2grib[:, latmax : latmin + 1, longmin : longmax + 1] = data
                if set(namelist).issubset(set(receiver_ds.keys())):
                    receiver_ds.update(
                        {
                            f: (dims, data2grib[pred.feature_names_to_idx[f]])
                            for f in feature_names
                        }
                    )
                else:
                    receiver_ds.assign(
                        {
                            f: (dims, data2grib[pred.feature_names_to_idx[f]])
                            for f in feature_names
                        }
                    )
            else:
                "only one variable"
                data2grib = nanmask
                data2grib[latmax : latmin + 1, longmin : longmax + 1] = data
                receiver_ds[name] = (dims, data2grib.astype(np.float32))
                receiver_ds[name] = receiver_ds[name].assign_attrs(
                    **template_ds[name].attrs
                )
    elif tol == group:
        # in this case, there might be several variables : basis for nanmask duplication
        dims = template_ds.dims
        maybe_repeat = len(feature_idx) if len(feature_idx) > 1 else 0
        if maybe_repeat:
            data2grib = np.repeat(nanmask[np.newaxis], maybe_repeat, axis=0)
            data2grib[:, latmax : latmin + 1, longmin : longmax + 1] = data
        else:
            data2grib = nanmask
            data2grib[latmax : latmin + 1, longmin : longmax + 1] = data

        receiver_ds.update(
            {f: (dims, data2grib[pred.feature_names_to_idx[f]]) for f in feature_names}
        )
        receiver_ds[name] = receiver_ds[name].assign_attrs(**template_ds[name].attrs)

    return receiver_ds


def get_output_filename(
    saving_settings: GribSavingSettings, sample: Any, leadtime: float
) -> str:
    identifiers = []
    for ident in saving_settings.sample_identifiers:
        if ident != "leadtime":
            identifiers.append(getattr(sample, ident))
        else:
            identifiers.append(leadtime)
    filename = saving_settings.output_fmt.format(
        *saving_settings.output_kwargs, *identifiers
    )
    return filename


def get_grib_param_dataframe(pred: NamedTensor, params: list) -> pd.DataFrame:
    """Match feature names of the pred named tensor to grid-readable levels and parameter names.
    Throw warnings if feature names are found that do not match any parameter (these features will not be saved).
    Args:
        pred (NamedTensor): input named tensor
        params (list): list of parameters, implementing typical Param instances used to describe datasets.


    Returns:
        pd.Dataframe : relating all feature names with name, level, typeOfLevel
    """

    pred_feature_names = pred.feature_names
    unmatched_feature_names = set(pred.feature_names)
    list_features = []
    for param in params:
        trial_names = param.parameter_short_name
        for ft_idx, ftname in enumerate(trial_names):
            if ftname in pred_feature_names:
                level, name, tol = (
                    param.levels[ft_idx],
                    param.shortname,
                    param.level_type,
                )
                list_features.append(
                    pd.DataFrame(
                        {
                            "feature_name": [ftname],
                            "level": [level],
                            "name": [name],
                            "typeOfLevel": [tol],
                        },
                        index=[ftname],
                    )
                )
                unmatched_feature_names.remove(ftname)
    # pd.concat is more efficient than df.append so we concat at the end
    grib_features = pd.concat(list_features)

    if len(unmatched_feature_names) != 0:
        raise UserWarning(
            f"There where unmatched features in pred tensor (no associated param found) : {unmatched_feature_names}"
        )
    return grib_features


def get_grib_groups(grib_features: pd.DataFrame) -> dict[str:dict]:
    """Identify the largest possible groups of keys to open the grib file a the least possible number of times.

    This has a strong impact on the program's I/O performance, as reading a (possibly large) grib file
    is expensive. Unfortunately, the fields stored in the grib can have incompatible dimensions when viewed
    as xarray Datasets. Therefore, we are forced to open the grib at least as many times as there are incompatible
    fields in the pred tensor feature names.

    This function parses the grib_features to get groups of compatible fields. The current strategy is to identify
    either variables or levels as group identifiers, and gather all fields that share this identifiers.
    Also, different typeOfLevels are easy-to-use group identifiers / separators.

    Notes :
        -> the main incompatibilities postulated in the current strategy are typeOfLevels and variable names/levels.
        other types of incompatibilities might include typeOfStatisticalProcessing, thresholding,
        vertical-wise integration, etc.
        -> more efficient strategies likely exist, but solving the problem in a general manner is NP-hard,
        even if you know all compatibility relations (bipartite dimension problem).
        -> for a low number of fields, this does not provide a large amount of acceleration, but should be
        more efficient.

    Args:
        grib_features (pd.DataFrame): output of get_grib_param_dataframe function

    Returns:
        dict[str : dict]: mapping of group identifiers to grib-style key filtering
    """
    if "surface" in grib_features["typeOfLevel"]:
        grib_groups = {
            "surface": {
                "cfVarName": grib_features["name"]
                .loc[(grib_features["typeOfLevel"] == "surface")]
                .tolist(),
                "typeOfLevel": "surface",
            },
        }
    else:
        grib_groups = {}

    typesOflevel = grib_features["typeOfLevel"].drop_duplicates().tolist()

    for tol in typesOflevel:
        if tol != "surface":
            levels_tol = grib_features["level"].loc[
                (grib_features["typeOfLevel"] == tol)
            ]
            names_tol = grib_features["name"].loc[(grib_features["typeOfLevel"] == tol)]

            cribler = levels_tol if len(levels_tol) < len(names_tol) else names_tol
            cribler_flag = "level" if len(levels_tol) < len(names_tol) else "name"
            for c in cribler:
                if cribler_flag == "level":
                    filter_keys = {
                        "level": c,
                        "cfVarName": names_tol[(grib_features["level"] == c)].tolist(),
                        "typeOfLevel": tol,
                    }
                else:
                    filter_keys = {
                        "level": levels_tol[(grib_features["name"] == c)].tolist(),
                        "cfVarName": c,
                        "typeOfLevel": tol,
                    }
                grib_groups[f"{c}_{tol}"] = filter_keys

    return grib_groups


def make_nan_mask(
    infer_dataset: DatasetABC, template_dataset: xr.Dataset
) -> Tuple[Union[np.ndarray, None], Tuple]:
    """This is to ensure that the infer_dataset grid can be matched with the grib template's grid
    (at this point grib data is passed to an xarray Dataset).
    If the inference grid is stricly embeddable in the template one, then it will be padded with NaNs.
    If the two grids exactly match, nothing is done.
    If the two grids do not match, this function raises an exception.

    Args:
        infer_dataset (DatasetABC): the inference dataset, should possess a grid attribute.
        template_dataset (xr.Dataset): the xarray Dataset created from the template grib file.

    Returns:
        Tuple[Union[np.ndarray, None], Tuple]:
            nanmask : np.ndarray of the shape of template_dataset grid, filled with nans, or None if the grids match.
            (latmin, latmax, longmin, longmax) : int, indices of the infer grid frontier in the template grid.
    """
    try:
        assert hasattr(infer_dataset, "grid")
    except AssertionError:
        raise NotImplementedError(
            f"The dataset {infer_dataset} has no grid attribute, cannot write grib."
        )

    if (
        (
            np.array(template_dataset.latitude.values.min())
            <= infer_dataset.grid.lat[:, 0].min()
        )
        and (
            np.array(template_dataset.latitude.values.max())
            >= infer_dataset.grid.lat[:, 0].max()
        )
        and (
            np.array(template_dataset.longitude.values.min())
            <= infer_dataset.grid.lon[:, 0].min()
        )
        and (
            np.array(template_dataset.longitude.values.max())
            >= infer_dataset.grid.lon[:, 0].max()
        )
    ):
        nanmask = np.empty(
            (len(template_dataset.latitude), len(template_dataset.longitude))
        )
        nanmask[:] = np.nan
        # matching latitudes
        latmin, latmax = (
            np.where(
                np.round(template_dataset.latitude.values, 5)
                == round(infer_dataset.grid.lat.min(), 5)
            )[0],
            np.where(
                np.round(template_dataset.latitude.values, 5)
                == round(infer_dataset.grid.lat.max(), 5)
            )[0],
        )

        # matching longitudes
        longmin, longmax = (
            np.where(
                np.round(template_dataset.longitude.values, 5)
                == round(infer_dataset.grid.lon.min(), 5)
            )[0],
            np.where(
                np.round(template_dataset.longitude.values, 5)
                == round(infer_dataset.grid.lon.max(), 5)
            )[0],
        )

        latmin, latmax, longmin, longmax = (
            latmin.item(),
            latmax.item(),
            longmin.item(),
            longmax.item(),
        )

    else:

        raise ValueError(
            f"Lat/Lon dims of the {infer_dataset} do not fit in template grid, cannot write grib."
        )

    return nanmask, (latmin, latmax, longmin, longmax)
