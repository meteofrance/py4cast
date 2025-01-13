import json
import os
from pathlib import Path
from typing import List, Literal

import numpy as np
from torch import save, tensor

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    ParamConfig,
    Timestamps,
    WeatherParam,
)
from py4cast.settings import CACHE_DIR, DEFAULT_CONFIG_DIR


class DummyAccessor(DataAccessor):
    print("importing DummyAccessor automatically creates directory and dummy config")
    config = {
        "periods": {
            "train": {"start": 20230101, "end": 20230101, "obs_step": 3600},
            "valid": {"start": 20230101, "end": 20230101, "obs_step": 3600},
            "test": {"start": 20230101, "end": 20230101, "obs_step": 3600},
        },
        "settings": {"standardize": "true", "file_format": "npy"},
        "grid": {
            "name": "dummygrid",
            "border_size": 0,
            "subdomain": [0, 64, 0, 64],
            "proj_name": "PlateCarree",
            "projection_kwargs": {},
        },
        "params": {
            "dummy_parameter": {"levels": [500], "kind": "input_output"},
        },
    }

    jsonconfig = json.dumps(config, sort_keys=True, indent=4)

    with open(DEFAULT_CONFIG_DIR / "datasets" / "dummy_config.json", "w") as outfile:
        outfile.write(jsonconfig)

    def cache_dir(self, name: str, grid: Grid) -> Path:
        path = CACHE_DIR / f"{name}_{grid.name}"
        os.makedirs(path, mode=0o777, exist_ok=True)

        if not os.path.exists(path / "parameters_stats.pt"):
            stats = {
                "dummy_parameter_500_isobaricInhPa": {
                    "mean": tensor(0.0),
                    "std": tensor(1.0),
                    "max": tensor(3.0),
                    "min": tensor(-3.0),
                }
            }
            save(stats, path / "parameters_stats.pt")
        if not os.path.exists(path / "diff_stats.pt"):
            dstats = {
                "dummy_parameter_500_isobaricInhPa": {
                    "mean": tensor(0.0),
                    "std": tensor(1.42),
                }
            }

            save(dstats, path / "diff_stats.pt")

        return path

    @staticmethod
    def get_dataset_path(name: str, grid: Grid) -> Path:
        path = CACHE_DIR / f"{name}_{grid.name}"
        os.makedirs(path, mode=0o777, exist_ok=True)

        return path

    @staticmethod
    def get_weight_per_level(
        level: int,
        level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
    ) -> float:
        return 1.0

    @staticmethod
    def load_grid_info(name: str) -> GridConfig:
        lat = (np.indices((64,)) - 16) * 0.5
        lon = (np.indices((64,)) + 30) * 0.5

        return GridConfig(
            full_size=(64, 64),
            latitude=lat.squeeze(),
            longitude=lon.squeeze(),
            geopotential=np.ones((64, 64)),
            landsea_mask=None,
        )

    @staticmethod
    def get_grid_coords(param: WeatherParam) -> List[float]:
        return [-8.0, 24.0, 15.0, 47.0]

    @staticmethod
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
            self.get_dataset_path(dataset_name, param.grid) / "dummy_data.npy"
        ):
            arr = np.random.randn(len(timestamps.timedeltas), 64, 64, 1).clip(-3, 3)
            np.save(
                self.get_dataset_path(dataset_name, param.grid) / "dummy_data.npy", arr
            )
        return self.get_dataset_path(dataset_name, param.grid) / "dummy_data.npy"

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

    def exists(
        self,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: Literal["npy", "grib"] = "grib",
    ) -> bool:
        return True
