from py4cast.datasets.base import NamedTensor
from cfgrib import xarray_to_grib as xtg
import xarray as xr
from copy import deepcopy
import datetime as dt
from typing import List, Tuple, Dict
import numpy as np

def saveNamedTensorToGrib(
    pred: NamedTensor, params: list, leadtimes: list, date: dt.datetime, saving_settings: dict
) -> None:
    """
    Write a named tensor (pred) to grib files, using a prefilled grib file as template.
    The pred data should already be on cpu.

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
    grib_keys, levels, names = get_grib_keys(pred, params)

    predicted_time_steps = len(leadtimes)

    model_ds = xr.open_dataset(
        saving_settings["template_grib"],
        backend_kwargs={
            "indexpath": "",
            "read_keys": [
                "level",
                "shortname",
                "centre",
                "typeOfGeneratingProcess",
                "generatingProcessIdentifier",
            ],
            "filter_by_keys": {"level": levels, "cfVarName": names},
        },
    )

    for t_idx in range(predicted_time_steps):
        ds_temp = model_ds.copy()
        ds_temp["time"] = date
        ds_temp["step"] = np.timedelta64(leadtimes[t_idx], "h")
        for feature_name in grib_keys.keys():
            name, level = (
                grib_keys[feature_name]["name"],
                grib_keys[feature_name]["level"],
            )

            data = (
                (
                    pred.tensor[t_idx, :, :, pred.feature_names_to_idx[feature_name]]
                    .cpu()
                    .numpy()
                )
                if pred.num_spatial_dims == 2
                else (
                    pred.tensor[t_idx, :, pred.feature_names_to_idx[feature_name]]
                    .cpu()
                    .numpy()
                )
            )
            # TODO : filter loc key depending on the nature of the output
            dims = ds_temp[name].loc[{"isobaricInhpa": level}].dims
            ds_temp[name].loc[{"isobaricInhpa": level}] = (dims, data)
            ds_temp[name] = ds_temp[name].assign_attrs(model_ds[name].attrs)
            
        xtg.to_grib(
            ds_temp,
            saving_settings["directory"]
            / saving_settings["output_fmt"].format(
                *saving_settings["output_kwargs"], date, leadtimes[t_idx]
            ),
            'wb'
        )


def get_grib_keys(
    pred: NamedTensor, params: list
) -> Tuple[Dict[str : Dict[str:str]], List[float], List[str]]:

    """Match feature names of the pred named tensor to grid-readable levels and parameter names.
    Will throw warnings if feature names are found that do not match any parameter : these features will then not be saved.
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
    unmatched_feature_names = set(pred.feature_names)
    for param in params:
        trial_names = param.parameter_short_names
        for ftname in trial_names:
            if ftname in pred_feature_names:
                grib_keys[ftname] = {"level": param.level, "name": param.grib_name}
                unmatched_feature_names.pop(ftname)
                if param.level not in levels:
                    levels.append(param.level)
                if param.grib_name not in names:
                    names.append(param.grib_name)
    if len(unmatched_feature_names) != 0:
        raise UserWarning(
            f"There where unmatched features in pred tensor : {unmatched_feature_names}"
        )

    return grib_keys, levels, names
