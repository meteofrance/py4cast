import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import epygram
import gif
import numpy as np
from dataclasses_json import dataclass_json
from mfai.torch.namedtensor import NamedTensor

from py4cast.datasets.base import DatasetABC
from py4cast.datasets.titan.settings import METADATA
from py4cast.utils import make_gif


@dataclass_json
@dataclass
class OutputSavingSettings:
    """
    Class to hold data about saving settings. Return the path to save gif and gribs
    - a template grib path
    - the saving grib directory
    - the saving gif directory
    - path_to_runtime : The path to the the runtime to concatenate to the directory
    - output_kwargs : fill the placeholders of path_to_runtime
    - grib_fmt : the name format of the grib to be written (as fillable f-string with empty placeholders)
    - grib_identifiers : the keywords to insert in the given grib_fmt.
    - gif_fmt : the name format of the gif to be written (as fillable f-string with empty placeholders)
    - gif_identifiers : the keywords to insert in the given gif_fmt
    There should be p-i keywords, where p is the number of placeholders and i the number of sample identifiers.
    """

    template_grib: str
    dir_grib: str
    dir_gif: str
    path_to_runtime: str
    grib_fmt: str = "grid.forecast_ai_date_{}_ech_{}.json"
    output_kwargs: tuple[str, ...] = ()
    grib_identifiers: tuple[str, ...] = ("date", "leadtime")
    gif_fmt: str = "{}_feature_{}.gif"
    gif_identifiers: tuple[str, ...] = ("runtime", "feature")

    def get_path(self, dir_path, runtime, idents, idents_dict, fmt):
        """
        return the dir_path in arg concatenated with the path to the runtime and the format given.
        """
        # Tests if number of placeholders are matching numer of id.
        ph = len((fmt).split("{}")) - 1
        fi = len(idents)
        if ph != fi:
            raise ValueError(
                f"fmt : {fmt} has {ph} placeholders,\
                but {fi} identifiers."
            )
        ph2 = (
            len((self.path_to_runtime).split("{}")) - 2
        )  # minus 2 because of the runtime mandatory
        kw = len(self.output_kwargs)
        if ph2 != kw:
            raise ValueError(
                f"fmt : {self.path_to_runtime} has {ph2} placeholders,\
                but {kw} identifiers."
            )

        identifiers = []
        for ident in idents:
            identifiers.append(idents_dict[ident])

        full_path = (
            dir_path
            / self.path_to_runtime.format(*self.output_kwargs, runtime)
            / fmt.format(*identifiers)
        )

        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def get_gif_path(self, runtime, feature):

        idents_dict = {}
        idents_dict["runtime"] = runtime
        idents_dict["feature"] = feature
        path = self.get_path(
            self._dir_gif, runtime, self.gif_identifiers, idents_dict, self.gif_fmt
        )
        return path

    def get_grib_path(self, runtime, member, leadtime):

        idents_dict = {}
        idents_dict["leadtime"] = leadtime
        # format string
        mb = str(member).zfill(3)
        idents_dict["member"] = mb
        path = self.get_path(
            self._dir_grib, runtime, self.grib_identifiers, idents_dict, self.grib_fmt
        )
        return path

    @property
    def _dir_grib(self):
        path = Path(self.dir_grib)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _dir_gif(self):
        path = Path(self.dir_gif)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _template_grib(self):
        return self._dir_grib / self.template_grib


def save_named_tensors_to_grib(
    pred: NamedTensor, ds: DatasetABC, sample: Any, saving_settings: dict, runtime: str
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
        runtime (str) :
    """

    # Get number of prediction
    predicted_time_steps = len(sample.output_timestamps.validity_times)
    # Open template grib
    tmplt_ds = epygram.formats.resource(saving_settings._template_grib, "r")

    datetime = sample.output_timestamps.datetime

    # Get time between 2 consecutive steps in hours
    time_step = int(
        (
            sample.timestamps.timedeltas[1] - sample.timestamps.timedeltas[0]
        ).total_seconds()
    )

    for step_idx in range(predicted_time_steps):

        # Get data
        raw_data = pred.select_dim("timestep", step_idx)
        # Define leadtime
        leadtime = int(
            sample.output_timestamps.timedeltas[step_idx].total_seconds() / 60**2
        )

        # Get timedelta & validity time from initial datetime
        timedelta = sample.output_timestamps.timedeltas[step_idx]
        validity_time = sample.output_timestamps.validity_times[step_idx]

        # Get the name of the file
        member = getattr(sample, "member") + 1
        full_path = saving_settings.get_grib_path(runtime, member, leadtime)

        # Create a GRIB
        grib_final = epygram.formats.resource(full_path, "w", fmt="GRIB")

        feature_not_accepted = []
        for feature in pred.feature_names:

            # validity
            dict_val = {
                "date_time": validity_time,
                "basis": datetime,
                "term": timedelta,
            }

            # Get FID of the current feature
            fid = feature2fid(feature, dict_val, time_step)
            # if fid is None continue the for loop
            if not fid:
                feature_not_accepted.append(feature)
                continue

            # Get template grib coordinates
            f = tmplt_ds.readfield(fid).clone()
            lon_grib, lat_grib = f.geometry.get_lonlat_grid()
            _lat_grib = lat_grib[:, 0]
            _lon_grib = lon_grib[0, :]
            shape_grib_latlon = (len(_lat_grib), len(_lon_grib))

            # Select correct feature in py4cast prediction
            data = (
                raw_data.tensor[:, :, raw_data.feature_names_to_idx[feature]]
                .cpu()
                .numpy()
            )

            # Position the infer dataset into the latlon grib, return latlon idx in the grib.
            idxs_latlon_grib = match_latlon(ds, _lat_grib, _lon_grib)

            # Create data for masked array filled with default value
            mask = fill_tensor_with(
                embedded_data=False,
                embedded_idxs=idxs_latlon_grib,
                shape=shape_grib_latlon,
                default_v=True,
                _dtype=bool,
            )
            _data = fill_tensor_with(
                embedded_data=data,
                embedded_idxs=idxs_latlon_grib,
                shape=shape_grib_latlon,
                default_v=f.data.data[0][0],
                _dtype=np.float64,
            )

            m_arr = np.ma.MaskedArray(_data, mask, fill_value=f.data.fill_value)

            f.setdata(m_arr)
            f.validity[0].set(**dict_val)
            grib_final.writefield(f)

        print(f"Leadtime {leadtime} has been written in {full_path}")


def save_gifs(pred, runtime, grid, save_settings):

    for feature_name in pred.feature_names:
        # Make gif
        feat = [pred.tensor[:, :, :, pred.feature_names_to_idx[feature_name]].cpu()]
        frames = make_gif(
            feature_name,
            runtime,
            None,
            feat,
            "Py4cast",
            grid.projection,
            grid.grid_limits,
            METADATA,
        )

        # Save gifs
        gif_path = save_settings.get_gif_path(runtime, feature_name)
        gif.save(frames, str(gif_path), duration=500)


def fill_tensor_with(embedded_data, embedded_idxs, shape, default_v, _dtype):
    """
    This function creates a numpy array of shape shape with a value defined by default_v.
    The embedded data should be embeddable in this array, and are positions at the index in embedded_idxs.
    Args:
        embedded_data (Union[numpy.ndarray, bool]): the data to position in the grib
        embedded_idxs (Tuple): A tuple of 4 idx ; represents the position in the array.
        shape (Tuple): shape of the returned array,
        default_v (Union[bool, float64]): fill the rest of the array with this default value
        _dtype: dtype of the array
    Return :
        a numpy.ndarray
    """
    (
        latmin,
        latmax,
        longmin,
        longmax,
    ) = embedded_idxs

    _tensor = np.full(shape, default_v, dtype=_dtype)
    _tensor[latmin : latmax + 1, longmin : longmax + 1] = embedded_data

    return _tensor


def match_latlon(
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
        Tuple
            (latmin, latmax, longmin, longmax) : int, indices of the infer grid frontier in the template grid.
    """
    try:
        assert hasattr(infer_dataset, "grid")
    except AssertionError:
        raise NotImplementedError(
            f"The dataset {infer_dataset} has no grid attribute, cannot write grib."
        )

    if (
        (np.array(lat.min()) <= infer_dataset.grid.lat[:, 0].min())
        and (np.array(lat.max()) >= infer_dataset.grid.lat[:, 0].max())
        and (np.array(lon.min()) <= infer_dataset.grid.lon[:, 0].min())
        and (np.array(lon.max()) >= infer_dataset.grid.lon[:, 0].max())
    ):

        # matching latitudes
        latmin, latmax = (
            np.where(np.round(lat, 5) == round(infer_dataset.grid.lat.min(), 5))[0],
            np.where(np.round(lat, 5) == round(infer_dataset.grid.lat.max(), 5))[0],
        )
        # matching longitudes
        longmin, longmax = (
            np.where(np.round(lon, 5) == round(infer_dataset.grid.lon.min(), 5))[0],
            np.where(np.round(lon, 5) == round(infer_dataset.grid.lon.max(), 5))[0],
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

    return (latmin, latmax, longmin, longmax)


def feature2fid(feature: str, dict_val: Dict[str, dt.datetime], time_step: int):
    """
    Return fid from the feature name.
    TODO should be automatic.
    """
    name2fid = {}
    name2fid.update(
        {
            "temperature": {
                "editionNumber": 2,
                "name": "2 metre temperature",
                "shortName": "2t",
                "discipline": 0,
                "parameterCategory": 0,
                "parameterNumber": 0,
                "typeOfFirstFixedSurface": 103,
                "level": 2,
                "typeOfSecondFixedSurface": 255,
                "tablesVersion": 15,
                "productDefinitionTemplateNumber": 0,
            },
            "u10": {
                "editionNumber": 2,
                "name": "10 metre U wind component",
                "shortName": "10u",
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
                "name": "10 metre V wind component",
                "shortName": "10v",
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
                "name": "2 metre relative humidity",
                "shortName": "2r",
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
                "name": "Pressure reduced to MSL",
                "shortName": "prmsl",
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
                "name": "Time integral of rain flux",
                "shortName": "tirf",
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
        }
    )
    if feature in ["aro_t2m_2m", "t2m_2_heightAboveGround"]:
        fid = name2fid["temperature"]
    elif feature in ["u10_10_heightAboveGround", "aro_u10_10m"]:
        fid = name2fid["u10"]
    elif feature in ["v10_10_heightAboveGround", "aro_v10_10m"]:
        fid = name2fid["v10"]
    # Pressure reduced to MSL
    elif feature == "aro_prmsl_0hpa":
        fid = name2fid["pmer"]
    # 2 metre relative humidity
    elif feature == "aro_r2_2m":
        fid = name2fid["r2"]
    # Total precipitation
    elif feature == "aro_tp_0m":
        fid = name2fid["tp"]
        dict_val["cumulativeduration"] = dt.timedelta(seconds=time_step)
    else:
        return None
    return fid
