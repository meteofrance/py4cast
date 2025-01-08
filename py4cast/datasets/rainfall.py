import datetime as dt
from pathlib import Path
from typing import List, Literal

import numpy as np
import xarray as xr

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    Period,
    GridConfig,
    ParamConfig,
    SamplePreprocSettings,
    Timestamps,
    WeatherParam,
)
from py4cast.datasets.base import DatasetABC


#############################################################
#                          SETTINGS                         #
#############################################################
FORMATSTR = "%Y%m%d%H%M"
SCRATCH_PATH = Path("/scratch/shared/RADAR_DATA/antilope_5min")


class RainfallAccessor(DataAccessor):

    @staticmethod
    def get_weight_per_level():
        return 1.0

    #############################################################
    #                            GRID                           #
    #############################################################
    @staticmethod
    def load_grid_info(name: str) -> GridConfig:
        if name not in ["FRANXL1S100"]:
            raise NotImplementedError("Grid must be in ['FRANXL1S100'].")

        path = SCRATCH_PATH / f"conf_{name}.grib"
        conf_ds = xr.open_dataset(path)
        landsea_mask = None
        grid_conf = GridConfig(
            conf_ds.prec.shape,
            conf_ds.latitude.values,
            conf_ds.longitude.values,
            conf_ds.h.values,
            landsea_mask,
        )
        return grid_conf

    @staticmethod
    def get_grid_coords(param: WeatherParam) -> List[int]:
        return [51.5, 41.0, -6.0, 10.5]

    #############################################################
    #                              PARAMS                       #
    #############################################################
    @staticmethod
    def load_param_info(name: str = "precip") -> ParamConfig:
        if name not in ["precip"]:
            raise NotImplementedError("Param must be in ['precip'].") 
        return ParamConfig(
            unit="kg m**-2",
            level_type="surface",
            long_name="antilope_precipitation",
            grid=name,
            grib_name=None,
            grib_param="prec",
        )
    
    #############################################################
    #                              LOADING                      #
    #############################################################

    @classmethod
    def cache_dir(cls, name: str, grid: Grid):
        return cls.get_dataset_path(name, grid)

    @staticmethod
    def get_dataset_path(name: str, grid: Grid):
        return SCRATCH_PATH / "cache"

    @classmethod
    def get_filepath(
        cls,
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
        return SCRATCH_PATH / file_format / "Hexagone" / f"{date.strftime(FORMATSTR)}.grib"

    @classmethod
    def load_data_from_disk(
        cls,
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
        for date in dates:
            data_path = cls.get_filepath(ds_name, param, date, file_format)
            if file_format == "grib":
                arr = xr.open_dataset(data_path)
                arr = arr.prec.values
            else:
                arr = np.load(data_path)
                arr = arr["arr_0"]
            
            arr_list.append(np.expand_dims(arr, axis=-1))
        return np.stack(arr_list)

    @classmethod
    def exists(
        cls,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: Literal["npy", "grib"] = "grib",
    ) -> bool:
        for date in timestamps.validity_times:
            filepath = cls.get_filepath(ds_name, param, date, file_format)
            if not filepath.exists():
                return False
        return True

    def valid_timestamp(n_inputs: int, timestamps: Timestamps) -> bool:
        return True

    def parameter_namer(param: WeatherParam) -> str:
        if param.level_type in ["surface", "heightAboveGround"]:
            level_type = "m"
        else:
            level_type = "hpa"
        return f"{param.name}_{param.level}{level_type}"



##############################################################################
#  Describe functions
##############################################################################
def describe():
    """Describes Rainfall."""
    train_ds = DatasetABC(
        name="rainfall",
        grid="FRANXL1S100",
        period=Period(
            name="train",
            start="2024010100",
            end="2025010100",
            step_duration=dt.timedelta(minutes=5),
            term_start=dt.timedelta(hours=-1),
            term_end=dt.timedelta(hours=3)
        ),
        params=WeatherParam(
            name="precip",
            level="surface",
            grid="FRANXL1S100",
            kind="input_output",
            ),
        settings=SamplePreprocSettings(
            dataset_name="rainfall",
            num_input_steps=12,
            num_pred_steps=36,
            step_duration=dt.timedelta(minutes=5),
        ),
        accessor=RainfallAccessor(),
    ),
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


if __name__ == "__main__":
    describe()