import os
from functools import cached_property
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
from py4cast.settings import CACHE_DIR


class DummyAccessor(DataAccessor):

    @cached_property
    def get_dataset_path(name: str, grid: Grid) -> Path:
        if not os.path.exists(CACHE_DIR / f"{name}_{grid.name}"):
            os.mkdir(CACHE_DIR / f"{name}_{grid.name}")
        return CACHE_DIR / f"{name}_{grid.name}"

    def get_weight_per_level(
        level: int,
        level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
    ) -> float:

        return 1.0

    def load_grid_info(name: str) -> GridConfig:
        lat, _ = (np.indices((64, 64)) - 16) * 0.5
        _, lon = (np.indices((64, 64)) + 30) * 0.5

        return GridConfig(
            full_size=(64, 64),
            latitude=lat,
            longitude=lon,
            geopotential=np.ones(64, 64),
            landsea_mask=None,
        )

    def get_grid_coords(param: WeatherParam) -> List[float]:
        return [-8.0, 24.0, 15.0, 47.0]

    def load_param_info(name: str) -> ParamConfig:
        return ParamConfig(
            unit="adimensional",
            level_type="isobaricInhPa",
            long_name="dummy_parameter",
            grid="dummygrid",
            grib_name=None,
            grib_param=None,
        )

    def get_filepath(
        self,
        dataset_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: str = "npy",
    ) -> Path:
        if not os.path.exists(
            self.get_dataset_path(dataset_name, "dummygrid") / "dummy_data.npy"
        ):
            arr = np.random.randn(len(timestamps.terms), 64, 64, 1).clip(-3, 3)
            np.save(
                self.get_dataset_path(dataset_name, "dummygrid") / "dummy_data.npy", arr
            )
        return self.get_dataset_path(dataset_name, "dummygrid") / "dummy_data.npy"

    def load_data_from_disk(
        self,
        dataset_name: str,  # name of the dataset or dataset version
        param: WeatherParam,  # specific parameter (2D field associated to a grid)
        timestamps: Timestamps,  # specific timestamp at which to load the field
        member: int = 0,  # optional members id. when dealing with ensembles
        file_format: Literal["npy", "grib"] = "npy",  # format of the base file on disk
    ) -> np.array:
        """
        Main loading function to fetch actual data on disk.
        loads a given parameter on a given timestamp
        """
        arr = np.load(self.get_filepath(dataset_name, param, timestamps))
        return arr
