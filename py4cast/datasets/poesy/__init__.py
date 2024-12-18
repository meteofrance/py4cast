import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    ParamConfig,
    SamplePreprocSettings,
    Timtestamps,
    WeatherParam,
)
from py4cast.datasets.base import DatasetABC
from py4cast.datasets.poesy.settings import (
    LATLON_FNAME,
    METADATA,
    OROGRAPHY_FNAME,
    SCRATCH_PATH,
)
from py4cast.settings import CACHE_DIR


@dataclass
class PoesyAccessor(DataAccessor):

    @staticmethod
    def get_dataset_path(name: str, grid: Grid) -> Path:
        complete_name = str(name) + "_" + grid.name
        return CACHE_DIR / complete_name

    @staticmethod
    def get_weight_per_level(
        level: float,
        level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
    ) -> float:
        if level_type == "isobaricInHpa":
            return 1.0 + (level) / (90)
        elif level_type == "heightAboveGround":
            return 2.0
        elif level_type == "surface":
            return 1.0
        else:
            raise Exception(f"unknown level_type:{level_type}")

    @staticmethod
    def load_grid_info(grid: Grid) -> GridConfig:
        geopotential = np.load(SCRATCH_PATH / OROGRAPHY_FNAME)
        latlon = np.load(SCRATCH_PATH / LATLON_FNAME)
        full_size = geopotential.shape
        latitude = latlon[1, :, 0]
        longitude = latlon[0, 0]
        landsea_mask = np.where(geopotential > 0, 1.0, 0.0).astype(np.float32)
        return GridConfig(full_size, latitude, longitude, geopotential, landsea_mask)

    @staticmethod
    def load_param_info(name: str) -> ParamConfig:
        info = METADATA["WEATHER_PARAMS"][name]
        unit = info["unit"]
        long_name = info["long_name"]
        grid = info["grid"]
        level_type = info["level_type"]
        grib_name = None
        grib_param = None
        return ParamConfig(unit, level_type, long_name, grid, grib_name, grib_param)

    @staticmethod
    def get_grid_coords(param: WeatherParam) -> List[float]:
        raise NotImplementedError("Poesy does not require get_grid_coords")

    @staticmethod
    def get_filepath(
        ds_name: str,
        param: WeatherParam,
        date: dt.datetime,
        file_format: str = "npy",
    ) -> str:
        """
        Return the filename.
        """
        var_file_name = METADATA["WEATHER_PARAMS"][param.name]["file_name"]
        return (
            SCRATCH_PATH
            / f"{date.strftime('%Y-%m-%dT%H:%M:%SZ')}_{var_file_name}_lt1-45_crop.npy"
        )

    def load_data_from_disk(
        self,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        member: int,
        file_format: str = "npy",
    ) -> np.array:
        """
        date : Date of file.
        term : Position of leadtimes in file.
        """
        data_array = np.load(
            self.get_filepath(ds_name, param, timestamps.datetime), mmap_mode="r"
        )

        arr = data_array[
            param.grid.subdomain[0] : param.grid.subdomain[1],
            param.grid.subdomain[2] : param.grid.subdomain[3],
            (timestamps.terms / dt.timedelta(hours=1)).astype(int) - 1,
            member,
        ].transpose([2, 0, 1])

        return np.expand_dims(arr, -1)

    def exists(
        self,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: str = "npy",
    ) -> bool:

        filepath = self.get_filepath(ds_name, param, timestamps.datetime, file_format)
        if not filepath.exists():
            return False
        return True

    def valid_timestamp(n_inputs: int, timestamps: Timestamps) -> bool:
        """
        Verification function called after the creation of each timestamps.
        Check if computed terms respect the dataset convention.
        Reminder:
        Poesy terms are between +1h lead time and +45h lead time.
        """
        limits = METADATA["TERMS"]
        for t in timestamps.terms:

            if (t > dt.timedelta(hours=int(limits["end"]))) or (
                t < dt.timedelta(hours=int(limits["start"]))
            ):
                return False
        return True
