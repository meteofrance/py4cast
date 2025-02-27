import datetime as dt
import time
from pathlib import Path
from typing import List, Literal

import numpy as np
import xarray as xr
import yaml
from cartopy.crs import PlateCarree, Stereographic
from tqdm import trange
from typer import Typer

from py4cast.datasets import compute_dataset_stats as cds
from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    ParamConfig,
    Timestamps,
    WeatherParam,
)
from py4cast.datasets.base import DatasetABC

#############################################################
#                          SETTINGS                         #
#############################################################
FORMATSTR = "%Y%m%d%H%M"
SCRATCH_PATH = Path("/scratch/shared/RADAR_DATA/lame_eau_npz")
DEFAULT_CONFIG = Path(__file__).parents[2] / "config/CLI/dataset/rainfall.yaml"

app = Typer()


def domain_to_extent(domain): # Cartopy needs regular lat lon data so conversion is needed
    crs = Stereographic(central_latitude=45)
    lower_right = crs.transform_point(*domain["lower_right"], PlateCarree())
    upper_right = crs.transform_point(*domain["upper_right"], PlateCarree())
    lower_left = crs.transform_point(*domain["lower_left"], PlateCarree())
    maxy, miny = upper_right[1], lower_left[1]
    minx, maxx = lower_left[0], lower_right[0]
    return (minx, maxx, miny, maxy)


class RainfallAccessor(DataAccessor):
    @staticmethod
    def get_weight_per_level(
        level: int,
        level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
    ):
        return 1.0

    #############################################################
    #                            GRID                           #
    #############################################################
    @staticmethod
    def load_grid_info(name: str) -> GridConfig:
        DOMAIN = {
            "upper_left": (-9.965, 53.670),
            "lower_right": (10.259217, 39.46785),
            "upper_right": (14.564706, 53.071644),
            "lower_left": (-6.977881, 39.852361),
        }
        shape = [1536, 1536]
        startlon, endlon, endlat, startlat = domain_to_extent(DOMAIN)
        lat = np.linspace(startlat, endlat, 1536)
        lon = np.linspace(startlon, endlon, 1536)
        altitude = np.zeros(shape) # Dummy topography mask
        landsea_mask = None
        grid_conf = GridConfig(
            shape,
            lat,
            lon,
            altitude,
            landsea_mask,
        )
        return grid_conf

    @property
    def dataset_name(self):
        return "rainfall"

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
            unit="mm/h",
            level_type="surface",
            long_name="lame d'eau Serval",
            grid=name,
            grib_name=None,
            grib_param="prec",
        )

    #############################################################
    #                              LOADING                      #
    #############################################################

    def cache_dir(self, name: str, grid: Grid):
        path = self.get_dataset_path(name, grid)
        path.mkdir(mode=0o777, exist_ok=True)
        return path

    @staticmethod
    def get_dataset_path(name: str, grid: Grid):
        return SCRATCH_PATH / "cache"

    @classmethod
    def get_filepath(
        cls,
        ds_name: str,
        param: WeatherParam,
        date: dt.datetime,
        file_format: Literal["npz"],
    ) -> Path:
        """
        Returns the path of the file containing the parameter data.
        - in grib format, data is grouped by level type.
        - in npz format, data is saved as npz, rescaled to the wanted grid, and each
        2D array is saved as one file to optimize IO during training."""
        return (
            SCRATCH_PATH
            / "Hexagone"
            / f"{date.year}"
            / f"{date.strftime(FORMATSTR)}.{file_format}"
        )

    @classmethod
    def load_data_from_disk(
        cls,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        # the member parameter is not accessed if irrelevant
        member: int = 0,
        file_format: Literal["npz"] = "npz",
    ):
        """
        Function to load invidiual parameter and lead time from a file stored in disk
        """
        dates = timestamps.validity_times
        arr_list = []
        for i, date in enumerate(dates):
            data_path = cls.get_filepath(ds_name, param, date, file_format)
            if file_format == "grib":
                arr = xr.open_dataset(data_path)
                arr = arr.prec.values
            else:
                arr = np.load(data_path)
                arr = arr["arr_0"]
                arr = np.where(arr < 0, 0, arr)  # Put 0 outside of radar field
                arr = arr / 100  # convert from mm10-2 to mm in 5 minutes
                arr = arr * 12  # convert to mm/h
            arr = arr[::-1]
            arr_list.append(np.expand_dims(arr, axis=-1))
        return np.stack(arr_list)

    @classmethod
    def exists(
        cls,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: Literal["npz"] = "npz",
    ) -> bool:
        for date in timestamps.validity_times:
            filepath = cls.get_filepath(ds_name, param, date, file_format)
            if not filepath.exists():
                return False
        return True

    @staticmethod
    def parameter_namer(param: WeatherParam) -> str:
        return param.name


##############################################################################
#  Describe and prepare functions
##############################################################################
@app.command()
def prepare(
    path_config: Path = DEFAULT_CONFIG,
    num_input_steps: int = 2,
    num_pred_steps_train: int = 1,
    num_pred_steps_val_test: int = 3,
    compute_stats: bool = True,
):
    """
    Prepares Rainfall dataset for training.
    This command will:
        - create all needed folders
        - computes statistics on all weather parameters.
    """

    print("--> Preparing Rainfall Dataset...")

    print("Load train dataset configuration...")
    with open(path_config, "r") as fp:
        conf = yaml.safe_load(fp)

    print("instantiate train dataset configuration...")
    train_ds, _, _ = DatasetABC.from_dict(
        RainfallAccessor,
        name=path_config.stem,
        conf=conf,
        num_input_steps=num_input_steps,
        num_pred_steps_train=num_pred_steps_train,
        num_pred_steps_val_test=num_pred_steps_val_test,
    )

    print("Creating cache folder")
    print(train_ds.cache_dir)
    train_ds.cache_dir.mkdir(exist_ok=True)

    if compute_stats:
        print(f"Dataset stats will be saved in {train_ds.cache_dir}")

        print("Computing stats on each parameter...")
        train_ds.settings.standardize = False

        cds.compute_parameters_stats(train_ds)

        print("Computing time stats on each parameters, between 2 timesteps...")
        train_ds.settings.standardize = True
        if hasattr(train_ds, "sample_list"):
            del train_ds.sample_list
        cds.compute_time_step_stats(train_ds)

    return train_ds


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG):
    """Describes Rainfall DataSet."""
    train_ds, _, _ = DatasetABC.from_dict(
        RainfallAccessor,
        fname=path_config.stem,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_tests=3,
    )
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


@app.command()
def plot(path_config: Path = DEFAULT_CONFIG):
    """Plots a png and a gif for one sample."""
    train_ds, _, _ = DatasetABC.from_dict(
        RainfallAccessor,
        fname=path_config.stem,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_tests=3,
    )
    print("Plot gif of one sample...")
    sample = train_ds.sample_list[0]
    sample.plot_gif("test.gif")
    print("Plot png for one step of sample...")
    item = sample.load(no_standardize=True)
    sample.plot(item, 0, "test.png")


@app.command()
def speedtest(path_config: Path = DEFAULT_CONFIG, n_iter: int = 5):
    print("Speed test:")
    train_ds, _, _ = DatasetABC.from_dict(
        RainfallAccessor,
        fname=path_config.stem,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_tests=3,
    )
    data_iter = iter(train_ds.torch_dataloader())
    start_time = time.time()
    for i in trange(n_iter, desc="Loading samples"):
        _ = next(data_iter)
    delta = time.time() - start_time
    speed = n_iter / delta
    print(f"Loading speed: {round(speed, 3)} sample(s)/sec")


if __name__ == "__main__":
    app()
