from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings, NamedTensor
from py4cast.lightning import AutoRegressiveLightning
import cfgrib as cf
from cfgrib import xarray_to_grib as xtg
import xarray as xr
from copy import deepcopy
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict


def saveNamedTensorToGrib(
    pred: NamedTensor,
    params: list,
    directory: Path,
    template_grib: Path,
    output_fmt: str,
) -> None:
    """
    Write a named (pred) to grib files, using a prefilled grib file as template.
    The pred data should already be on cpu.

    Args:
        pred (NamedTensor): the output tensor (pred)
        params : list of Param objects used to reference parameters description for writing grib
        directory (Path): the directory where outputs should be savec
        template_grib (Path): the path to the template grib file
        output_fmt (str): the name format to save the newly formed grib
    """
    grib_keys, levels, names = get_grib_keys(pred, params)

    predicted_time_steps = pred.tensor.shape[1]

    model_grib = xr.open_dataset(
        template_grib,
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
