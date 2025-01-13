import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union

import numpy as np

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    ParamConfig,
    Timestamps,
    WeatherParam,
)
from py4cast.datasets.poesy.settings import (
    LATLON_FNAME,
    METADATA,
    OROGRAPHY_FNAME,
    SCRATCH_PATH,
)
from py4cast.settings import CACHE_DIR


@dataclass
class PoesyAccessor(DataAccessor):
    def cache_dir(self, name: str, grid: Grid):
        complete_name = str(name) + "_" + grid.name
        return CACHE_DIR / complete_name

    @staticmethod
    def get_dataset_path(name: str, grid: Grid) -> Path:
        return SCRATCH_PATH

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

    @classmethod
    def get_filepath(
        cls,
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

    @classmethod
    def load_data_from_disk(
        cls,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        member: int,
        file_format: str = "npy",
    ) -> np.array:
        data_array = np.load(
            cls.get_filepath(ds_name, param, timestamps.datetime), mmap_mode="r"
        )

        arr = data_array[
            param.grid.subdomain[0] : param.grid.subdomain[1],
            param.grid.subdomain[2] : param.grid.subdomain[3],
            (np.array(timestamps.timedeltas) / dt.timedelta(hours=1)).astype(int) - 1,
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

    @staticmethod
    def optional_check_before_exists(
        t0: dt.datetime,
        num_input_steps: int,
        num_pred_steps: int,
        pred_step: dt.timedelta,
        leadtime: Union[dt.timedelta, None],
    ) -> bool:
        """
        Return True if the dataset contains the data for t0 + leadtime. Else return False.

        Args:
            - t0 (dt.datetime): valid time of the observation or run date (in case of dataset that contain
            multiple forecasts).
            - num_input_steps (int,): number of input steps.
            - num_pred_steps (int,): number of prediction steps.
            - pred_step (dt.timedelta): duration of the prediction step.
            - leadtime (dt.timedelta): leadtime for wich we want to know if it is a valid timestamp.

        Reminder:
            Poesy leadtimes are between +1h and +45h.
        """
        limits = METADATA["TERMS"]

        validtime = t0 + leadtime

        min_validtime = validtime - (num_input_steps - 1) * pred_step
        max_validtime = validtime + (num_pred_steps) * pred_step
        if min_validtime - t0 < dt.timedelta(hours=int(limits["start"])):
            return False
        if max_validtime - t0 > dt.timedelta(hours=int(limits["end"])):
            return False

        return True
