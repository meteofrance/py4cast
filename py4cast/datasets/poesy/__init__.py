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
    @staticmethod
    def cache_dir(name: str, grid: Grid):
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


def valid_timestamp(
    t0: dt.datetime,
    num_input_steps: int,
    num_pred_steps: int,
    step_duration: dt.timedelta,
    leadtimes: List[dt.timedelta],
) -> List[Timestamps]:
    """
    Return the list of all avalaible Timestamps for t0.
    Reminder:
    Poesy leadtimes are between +1h and +45h.
    """
    limits = METADATA["TERMS"]

    valid_times_one_run = [t0 + leadtime for leadtime in leadtimes]
    timesteps = [
        delta * step_duration
        for delta in range(-num_input_steps + 1, num_pred_steps + 1)
    ]

    timestamps = []
    for t in valid_times_one_run:
        min_validtime, max_validtime = t - timesteps[0], t + timesteps[-1]
        if min_validtime - t0 < dt.timedelta(hours=int(limits["start"])):
            continue
        if max_validtime - t0 > dt.timedelta(hours=int(limits["end"])):
            continue
        timestamps.append(Timestamps(datetime=t, timesteps=timesteps))

    return timestamps
