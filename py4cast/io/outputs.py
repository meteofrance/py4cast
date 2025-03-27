import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import datetime as dt

import epygram
import xarray as xr
from cfgrib import xarray_to_grib as xtg
from dataclasses_json import dataclass_json
from mfai.torch.namedtensor import NamedTensor

from py4cast.datasets.base import DatasetABC
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

    # Get number of prediction
    predicted_time_steps = len(sample.output_timestamps.validity_times)
    # Open template grib
    tmplt_ds = epygram.formats.resource(Path(saving_settings.directory) / saving_settings.template_grib, 'r')

    datetime = sample.output_timestamps.datetime 
    
    # Get time between 2 consecutive steps in hours
    time_step = int((sample.timestamps.timedeltas[1] - sample.timestamps.timedeltas[0]).total_seconds())

    for step_idx in range(predicted_time_steps):

        # Get data
        raw_data = pred.select_dim("timestep", step_idx, bare_tensor=False)
        # Define leadtime 
        leadtime = int(sample.output_timestamps.timedeltas[step_idx].total_seconds() / 60**2)
        
        # Get timedelta & validity time from initial datetime
        timedelta = sample.output_timestamps.timedeltas[step_idx]
        validity_time = sample.output_timestamps.validity_times[step_idx]

        # Get the name of the file
        path_to_file = get_output_filename(saving_settings, sample, leadtime)
        full_path = Path(saving_settings.directory) / path_to_file
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a GRIB
        grib_final = epygram.formats.resource(full_path, 'w',fmt='GRIB')

        feature_not_accepted = []
        for feature in pred.feature_names:

            # validity
            dict_val = {
                "date_time" : validity_time,
                "basis": datetime,
                "term": timedelta,  
                }
    
            # Get FID of the current feature
            fid = feature2fid(feature, dict_val, time_step)
            # if feature2fid return None continue the for loop
            if not fid:
                feature_not_accepted.append(feature)
                continue
            
            f = tmplt_ds.readfield(fid).clone()

            lon, lat = f.geometry.get_lonlat_grid()
            nanmask, latlon = make_nan_mask(ds, lat[:, 0], lon[0])
            (
                latmin,
                latmax,
                longmin,
                longmax,
            ) = latlon
            
             # Create mask for masked array in the field
            mask = np.full(nanmask.shape, True, dtype=bool)
            mask[latmin-1:latmax, longmin-1:longmax] = False

            # Select correct feature in py4cast prediction
            data = raw_data.tensor[:,:, raw_data.feature_names_to_idx[feature]].cpu().numpy()
            # Create data for masked array filled with default value
            _data = np.full(nanmask.shape,f.data.data[0][0], dtype=np.float64)
            _data[latmin-1:latmax, longmin-1:longmax] = data
            
            f.setdata(np.ma.MaskedArray(_data, mask, fill_value =f.data.fill_value))
            f.validity[0].set(**dict_val)
            grib_final.writefield(f)

        print(f"Leadtime {leadtime} has been written in {full_path}")
        print(f"Following features were not accepted : {feature_not_accepted}")

def get_output_filename(
    saving_settings: GribSavingSettings, sample: Any, leadtime: float
) -> str:
    identifiers = []
    for ident in saving_settings.sample_identifiers:
        if ident == "runtime":
            # Get the max of the input timestamps which is t0
            runtime = sample.timestamps.datetime.strftime("%Y%m%dT%H%MP")
            identifiers.append(runtime)
        elif ident == "leadtime":
            identifiers.append(leadtime)
        elif ident == "member":
            # Offset of 1. To start at 1 instead of 0.
            member = getattr(sample, ident) + 1
            # format string
            mb = str(member).zfill(3)
            identifiers.append(mb)

    path_to_file = saving_settings.output_fmt.format(
        *saving_settings.output_kwargs, *identifiers
    )
    return path_to_file

def make_nan_mask(
    infer_dataset: DatasetABC, lat: np.ndarray, lon: np.ndarray
) -> Tuple[Union[np.ndarray, None], Tuple]:
    """This is to ensure that the infer_dataset grid can be matched with the given lat lon.
    If the inference grid is stricly embeddable in the given coordinates, then it will be padded with NaNs.
    If it's exactly match, nothing is done.
    If the two grids do not match, this function raises an exception.

    Args:
        infer_dataset (DatasetABC): the inference dataset, should possess a grid attribute.
        lat (numpy.ndarray): the latitude of reference as numpy vector
        lon (numpy.ndarray): the longitude of reference as numpy vector

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
            np.array(lat.min())
            <= infer_dataset.grid.lat[:, 0].min()
        )
        and (
            np.array(lat.max())
            >= infer_dataset.grid.lat[:, 0].max()
        )
        and (
            np.array(lon.min())
            <= infer_dataset.grid.lon[:, 0].min()
        )
        and (
            np.array(lon.max())
            >= infer_dataset.grid.lon[:, 0].max()
        )
    ):
        nanmask = np.empty(
            [len(lat), len(lon)]
        )
        nanmask[:,:] = np.nan
        # matching latitudes
        latmin, latmax = (
            np.where(
                np.round(lat, 5)
                == round(infer_dataset.grid.lat.min(), 5)
            )[0],
            np.where(
                np.round(lat, 5)
                == round(infer_dataset.grid.lat.max(), 5)
            )[0],
        )
        # matching longitudes
        longmin, longmax = (
            np.where(
                np.round(lon, 5)
                == round(infer_dataset.grid.lon.min(), 5)
            )[0],
            np.where(
                np.round(lon, 5)
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

def feature2fid(feature, dict_val, time_step):
    name2fid = {}
    name2fid.update({
        "temperature": {
            'editionNumber': 2,
            'name': '2 metre temperature',
            'shortName': '2t',
            'discipline': 0,
            'parameterCategory': 0,
            'parameterNumber': 0,
            'typeOfFirstFixedSurface': 103,
            'level': 2,
            'typeOfSecondFixedSurface': 255,
            'tablesVersion': 15,
            'productDefinitionTemplateNumber': 0
        },
        "u10": {
                "editionNumber": 2,
                "name": '10 metre U wind component',
                "shortName": '10u',
                "discipline": 0,
                "parameterCategory": 2,
                "parameterNumber": 2,
                "typeOfFirstFixedSurface": 103,
                "level": 10,
                "typeOfSecondFixedSurface": 255,
                "tablesVersion": 15,
                "productDefinitionTemplateNumber": 0,
        },    
    "v10": { 
                "editionNumber": 2,
                "name": '10 metre V wind component',
                "shortName": '10v',
                "discipline": 0,
                "parameterCategory": 2,
                "parameterNumber": 3,
                "typeOfFirstFixedSurface": 103,
                "level": 10,
                "typeOfSecondFixedSurface": 255,
                "tablesVersion": 15,
                "productDefinitionTemplateNumber": 0,
        },
    "r2": {
                "editionNumber": 2,
                "name": '2 metre relative humidity',
                "shortName": '2r',
                "discipline": 0,
                "parameterCategory": 1,
                "parameterNumber": 1,
                "typeOfFirstFixedSurface": 103,
                "level": 2,
                "typeOfSecondFixedSurface": 255,
                "tablesVersion": 15,
                "productDefinitionTemplateNumber": 0,
        },
    "pmer": {
                "editionNumber": 2,
                "name": 'Pressure reduced to MSL',
                "shortName": 'prmsl',
                "discipline": 0,
                "parameterCategory": 3,
                "parameterNumber": 1,
                "typeOfFirstFixedSurface": 101,
                "level": 0,
                "typeOfSecondFixedSurface": 255,
                "tablesVersion": 15,
                "productDefinitionTemplateNumber": 0,
        },
    "tp": {
                "editionNumber": 2,
                "name": 'Time integral of rain flux',
                "shortName": 'tirf',
                "discipline": 0,
                "parameterCategory": 1,
                "parameterNumber": 65,
                "typeOfFirstFixedSurface": 1,
                "level": 0,
                "typeOfSecondFixedSurface": 255,
                "tablesVersion": 15,
                "productDefinitionTemplateNumber": 8,
                "lengthOfTimeRange": 1,
                "typeOfStatisticalProcessing": 1,
        },
    })
    if feature in ["aro_t2m_2m", "t2m_2_heightAboveGround"]:
        fid = name2fid["temperature"]
    elif feature in ['u10_10_heightAboveGround', "aro_u10_10m"]:
        fid = name2fid["u10"]
    elif feature in ['v10_10_heightAboveGround', "aro_v10_10m"]:
        fid  = name2fid["v10"]
    # Pressure reduced to MSL
    elif feature == 'aro_prmsl_0hpa':
        fid  = name2fid["pmer"]
        # 2 metre relative humidity
    elif feature == 'aro_r2_2m':
        fid  = name2fid["r2"]
    # Time integral of rain flux
    elif feature == 'aro_tp_0m':
        fid  = name2fid["tp"]
        dict_val["cumulativeduration"] = dt.timedelta(seconds=time_step)
    else:
        return None
    return fid