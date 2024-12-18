import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Literal

import numpy as np
import torch
import xarray as xr
from skimage.transform import resize

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    ParamConfig,
    SamplePreprocSettings,
    Stats,
    WeatherParam,
)
from py4cast.datasets.base import DatasetABC, Period, Timestamps
from py4cast.datasets.titan.settings import FORMATSTR, METADATA, SCRATCH_PATH


class TitanAccessor(DataAccessor):

    @staticmethod
    def get_weight_per_level(
        level: int,
        level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
    ):
        if level_type == "isobaricInhPa":
            return 1 + (level) / (1000)
        else:
            return 2.0

    #############################################################
    #                            GRID                           #
    #############################################################
    @staticmethod
    def load_grid_info(name: str) -> GridConfig:
        if name not in ["PAAROME_1S100", "PAAROME_1S40"]:
            raise NotImplementedError(
                "Grid must be in ['PAAROME_1S100', 'PAAROME_1S40']"
            )
        path = SCRATCH_PATH / f"conf_{name}.grib"
        conf_ds = xr.open_dataset(path)
        grid_info = METADATA["GRIDS"][name]
        full_size = grid_info["size"]
        landsea_mask = None
        grid_conf = GridConfig(
            full_size,
            conf_ds.latitude.values,
            conf_ds.longitude.values,
            conf_ds.h.values,
            landsea_mask,
        )
        return grid_conf

    @staticmethod
    def get_grid_coords(param: WeatherParam) -> List[int]:
        return METADATA["GRIDS"][param.grid.name]["extent"]

    #############################################################
    #                              PARAMS                       #
    #############################################################
    @staticmethod
    def load_param_info(name: str) -> ParamConfig:
        info = METADATA["WEATHER_PARAMS"][name]
        grib_name = info["grib"]
        grib_param = info["param"]
        unit = info["unit"]
        level_type = info["type_level"]
        long_name = info["long_name"]
        grid = info["grid"]
        if grid not in ["PAAROME_1S100", "PAAROME_1S40", "PA_01D"]:
            raise NotImplementedError(
                "Parameter native grid must be in ['PAAROME_1S100', 'PAAROME_1S40', 'PA_01D']"
            )
        return ParamConfig(unit, level_type, long_name, grid, grib_name, grib_param)

    #############################################################
    #                              LOADING                      #
    #############################################################

    @staticmethod
    def get_dataset_path(name: str, grid: Grid):
        str_subdomain = "-".join([str(i) for i in grid.subdomain])
        subdataset_name = f"{name}_{grid.name}_{str_subdomain}"
        return SCRATCH_PATH / "subdatasets" / subdataset_name

    def get_filepath(
        self,
        ds_name: str,
        param: WeatherParam,
        date: dt.datetime,
        file_format: Literal["npy", "grib"],
    ) -> Path:
        """
        Returns the path of the file containing the parameter data.
        - in grib format, data is grouped by level type.
        - in npy format, data is saved as npy, rescaled to the wanted grid, and each
        2D array is saved as one file to optimize IO during training."""
        if file_format == "grib":
            folder = SCRATCH_PATH / "grib" / date.strftime(FORMATSTR)
            return folder / param.grib_name
        else:
            npy_path = self.get_dataset_path(ds_name, param.grid) / "data"
            filename = f"{param.name}_{param.level}_{param.level_type}.npy"
            return npy_path / date.strftime(FORMATSTR) / filename

    def load_data_from_disk(
        self,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        # the member parameter is not accessed if irrelevant
        member: int = 0,
        file_format: Literal["npy", "grib"] = "grib",
    ):
        """
        Function to load invidiual parameter and lead time from a file stored in disk
        """
        dates = timestamps.validity_times
        arr_list = []
        for d in dates:
            data_path = self.get_filepath(ds_name, param, date, file_format)
            if file_format == "grib":
                arr, lons, lats = load_data_grib(param, data_path)
                arr = fit_to_grid(arr, lons, lats)
            else:
                arr = np.load(data_path)

            subdomain = param.grid.subdomain
            arr = arr[subdomain[0] : subdomain[1], subdomain[2] : subdomain[3]]
            if file_format == "grib":
                arr = arr[::-1]
            arr_list.append(arr)
        return np.stack(arr_list)  # invert latitude

    def exists(
        self,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: Literal["npy", "grib"] = "grib",
    ) -> bool:
        for date in timestamps.validity_times:
            filepath = self.get_filepath(ds_name, param, date, file_format)
            if not filepath.exists():
                return False
        return True

    def valid_timestamp(n_inputs: int, timestamps: Timestamps) -> bool:
        """
        Verification function called after the creation of each timestamps.
        Check if computed terms respect the dataset convention.
        Reminder:
        Titan terms are between +0h lead time and +23h lead time wrt to the day:00h00UTC reference
        Allowing larger terms would double-sample some samples (day+00h00 <-> (day+1)+24h00)
        """
        term_0 = timestamps.terms[n_inputs - 1]
        if term_0 > np.timedelta64(23, "h"):
            return False
        return True


############################################################
#                   HELPER FUNCTIONS for TITAN             #
############################################################


def fit_to_grid(
    param: WeatherParam,
    arr: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    get_grid_coords: Callable[[WeatherParam], List[str]],
) -> np.ndarray:
    # already on good grid, nothing to do:
    if param.grid.name == param.native_grid:
        return arr

    # crop arpege data to arome domain:
    if param.native_grid == "PA_01D" and param.grid.name in [
        "PAAROME_1S100",
        "PAAROME_1S40",
    ]:
        grid_coords = get_grid_coords(param)
        # Mask ARPEGE data to AROME bounding box
        mask_lon = (lons >= grid_coords[2]) & (lons <= grid_coords[3])
        mask_lat = (lats >= grid_coords[1]) & (lats <= grid_coords[0])
        arr = arr[mask_lat, :][:, mask_lon]

    anti_aliasing = param.grid.name == "PAAROME_1S40"  # True if downsampling
    # upscale or downscale to grid resolution:
    return resize(arr, param.grid.full_size, anti_aliasing=anti_aliasing)


@lru_cache(maxsize=50)
def read_grib(path_grib: Path) -> xr.Dataset:
    return xr.load_dataset(path_grib, engine="cfgrib", backend_kwargs={"indexpath": ""})


def load_data_grib(param: WeatherParam, path: Path) -> np.ndarray:
    ds = read_grib(path)
    assert param.grib_param is not None
    level_type = ds[param.grib_param].attrs["GRIB_typeOfLevel"]
    lats = ds.latitude.values
    lons = ds.longitude.values
    if level_type != "isobaricInhPa":  # Only one level
        arr = ds[param.grib_param].values
    else:
        arr = ds[param.grib_param].sel(isobaricInhPa=param.level).values
    return arr, lons, lats


#############################################################
#                            DATASET                        #
#############################################################


class TitanDataset(DatasetABC):
    # Si on doit travailler avec plusieurs grilles, on fera un super dataset qui contient
    # plusieurs datasets chacun sur une seule grille
    def __init__(
        self,
        name: str,
        grid: Grid,
        period: Period,
        params: List[WeatherParam],
        settings: SamplePreprocSettings,
        accessor_kls: TitanAccessor,
    ):
        super().__init__(name, grid, period, params, settings, accessor=accessor_kls())

    def __len__(self):
        return len(self.sample_list)