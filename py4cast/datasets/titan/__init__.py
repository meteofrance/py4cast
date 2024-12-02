import datetime as dt
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union

import gif
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import typer
import xarray as xr
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

from py4cast.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Item,
    NamedTensor,
    Period,
    TorchDataloaderSettings,
    collate_fn,
    get_param_list,
)
from py4cast.datasets.titan.settings import (
    DEFAULT_CONFIG,
    FORMATSTR,
    METADATA,
    SCRATCH_PATH,
)
from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing
from py4cast.plots import DomainInfo
from py4cast.utils import merge_dicts

app = typer.Typer()

class TitanAccessor(DataAccessor):

    def get_weight_per_lvl(
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

    def get_grid_coords(param: Param) -> List[int]:
       return METADATA["GRIDS"][param.grid.name]["extent"]

    #############################################################
    #                              PARAMS                       #
    #############################################################


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

    def get_filepath(
        ds_name: str,
        param: Param,
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
            npy_path = get_dataset_path(ds_name, param.grid) / "data"
            filename = f"{param.name}_{param.level}_{param.level_type}.npy"
            return npy_path / date.strftime(FORMATSTR) / filename


def process_sample_dataset(ds_name: str, date: dt.datetime, params: List[Param]):
    """Saves each 2D parameter data of the given date as one NPY file."""
    for param in params:
        dest_file = get_filepath(ds_name, param, date, "npy")
        dest_file.parent.mkdir(exist_ok=True)
        if not dest_file.exists():
            try:
                arr = load_data_from_disk(ds_name, param, date, "grib")
                np.save(dest_file, arr)
            except Exception as e:
                print(e)
                print(
                    f"WARNING: Could not load grib data {param.name} {param.level} {date}. Skipping sample."
                )
                break


def fit_to_grid(
    param: Param,
    arr: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    get_grid_coords: Callable[[Param], List[str]],
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


def load_data_grib(param: Param, path: Path) -> np.ndarray:
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


def load_data_from_disk(
    ds_name: str,
    param: Param,
    date: dt.datetime,
    file_format: Literal["npy", "grib"] = "grib",
) -> np.ndarray :
    """
    Function to load invidiual parameter and lead time from a file stored in disk
    """
    data_path = get_filepath(ds_name, param, date, file_format)
    if file_format == "grib":
        arr, lons, lats = load_data_grib(param, data_path)
        arr = fit_to_grid(arr, lons, lats)
    else:
        arr = np.load(data_path)

    subdomain = param.grid.subdomain
    arr = arr[subdomain[0] : subdomain[1], subdomain[2] : subdomain[3]]
    if file_format == "grib":
        arr = arr[::-1]
    return arr  # invert latitude


def exists(
    ds_name: str,
    param: Param,
    date: dt.datetime,
    file_format: Literal["npy", "grib"] = "grib",
) -> bool:
    filepath = get_filepath(ds_name, param, date, file_format)
    return filepath.exists()


def get_param_tensor(
    param: Param,
    stats: Stats,
    dates: List[dt.datetime],
    settings: Settings,
    no_standardize: bool = False,
) -> torch.tensor:
    """
    Fetch data on disk fo the given parameter and all involved dates
    Unless specified, normalize the samples with parameter-specific constants
    returns a tensor
    """
    arrays = [
        load_data_from_disk(settings.dataset_name, param, date, settings.file_format)
        for date in dates
    ]
    arr = np.stack(arrays)
    # Extend dimension to match 3D (level dimension)
    if len(arr.shape) != 4:
        arr = np.expand_dims(arr, axis=1)
    arr = np.transpose(arr, axes=[0, 2, 3, 1])  # shape = (steps, lvl, x, y)
    if settings.standardize and not no_standardize:
        name = param.parameter_short_name
        means = np.asarray(stats[name]["mean"])
        std = np.asarray(stats[name]["std"])
        arr = (arr - means) / std
    return torch.from_numpy(arr)


def generate_forcings(
    date: dt.datetime, output_terms: Tuple[float], grid: Grid
) -> List[NamedTensor]:
    """
    Generate all the forcing in this function.
    Return a list of NamedTensor.
    """
    lforcings = []
    time_forcing = NamedTensor(  # doy : day_of_year
        feature_names=["cos_hour", "sin_hour", "cos_doy", "sin_doy"],
        tensor=get_year_hour_forcing(date, output_terms).type(torch.float32),
        names=["timestep", "features"],
    )
    solar_forcing = NamedTensor(
        feature_names=["toa_radiation"],
        tensor=generate_toa_radiation_forcing(
            grid.lat, grid.lon, date, output_terms
        ).type(torch.float32),
        names=["timestep", "lat", "lon", "features"],
    )
    lforcings.append(time_forcing)
    lforcings.append(solar_forcing)

    return lforcings


#############################################################
#                            DATASET                        #
#############################################################


def get_dataset_path(name: str, grid: Grid):
    str_subdomain = "-".join([str(i) for i in grid.subdomain])
    subdataset_name = f"{name}_{grid.name}_{str_subdomain}"
    return SCRATCH_PATH / "subdatasets" / subdataset_name


class TitanDataset(DatasetABC, Dataset):
    # Si on doit travailler avec plusieurs grilles, on fera un super dataset qui contient
    # plusieurs datasets chacun sur une seule grille
    def __init__(
        self,
        name: str,
        grid: Grid,
        period: Period,
        params: List[Param],
        settings: Settings,
    ):
        self.name = name
        self.grid = grid
        if grid.name not in ["PAAROME_1S100", "PAAROME_1S40"]:
            raise NotImplementedError(
                "Grid must be in ['PAAROME_1S100', 'PAAROME_1S40']"
            )
        self.period = period
        self.params = params
        self.settings = settings
        self.shuffle = self.period.name == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        n_input, n_pred = self.settings.num_input_steps, self.settings.num_pred_steps
        filename = f"valid_samples_{self.period.name}_{n_input}_{n_pred}.txt"
        self.valid_samples_file = self.cache_dir / filename


    @cached_property
    def sample_list(self):
        """Creates the list of samples."""
        print("Start creating samples...")
        stats = self.stats if self.settings.standardize else None
        if self.valid_samples_file.exists():
            print(f"Retrieving valid samples from file {self.valid_samples_file}")
            with open(self.valid_samples_file, "r") as f:
                dates_str = [line[:-1] for line in f.readlines()]
                dateformat = "%Y-%m-%d_%Hh%M"
                dates = [dt.datetime.strptime(ds, dateformat) for ds in dates_str]
                dates = list(set(dates).intersection(set(self.period.date_list)))
                samples = [
                    Sample(date, self.settings, self.params, stats, self.grid)
                    for date in dates
                ]
        else:
            print(
                f"Valid samples file {self.valid_samples_file} does not exist. Computing samples list..."
            )
            samples = []
            for date in tqdm.tqdm(self.period.date_list):
                sample = Sample(date, self.settings, self.params, stats, self.grid)
                if sample.is_valid():
                    samples.append(sample)
        print(f"--> All {len(samples)} {self.period.name} samples are now defined")
        return samples


@app.command()
def prepare(
    path_config: Path = DEFAULT_CONFIG,
    num_input_steps: int = 1,
    num_pred_steps_train: int = 1,
    num_pred_steps_val_test: int = 1,
    convert_grib2npy: bool = False,
    compute_stats: bool = True,
    write_valid_samples_list: bool = True,
):
    """Prepares Titan dataset for training.
    This command will:
        - create all needed folders
        - convert gribs to npy and rescale data to the wanted grid
        - establish a list of valid samples for each set
        - computes statistics on all weather parameters."""
    print("--> Preparing Titan Dataset...")

    print("Load dataset configuration...")
    with open(path_config, "r") as fp:
        conf = json.load(fp)

    print("Creating folders...")
    train_ds, valid_ds, test_ds = TitanDataset.from_dict(
        path_config.stem,
        conf,
        num_input_steps,
        num_pred_steps_train,
        num_pred_steps_val_test,
    )
    train_ds.cache_dir.mkdir(exist_ok=True)
    data_dir = train_ds.cache_dir / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Dataset will be saved in {train_ds.cache_dir}")

    if convert_grib2npy:
        print("Converting gribs to npy...")
        param_list = get_param_list(
            conf, train_ds.grid, load_param_info, get_weight_per_lvl
        )
        sum_dates = (
            list(train_ds.period.date_list)
            + list(valid_ds.period.date_list)
            + list(test_ds.period.date_list)
        )
        dates = sorted(list(set(sum_dates)))
        for date in tqdm.tqdm(dates):
            process_sample_dataset(date, param_list)
        print("Done!")

    conf["settings"]["standardize"] = False
    train_ds, valid_ds, test_ds = TitanDataset.from_dict(
        path_config.stem,
        conf,
        num_input_steps,
        num_pred_steps_train,
        num_pred_steps_val_test,
    )
    if compute_stats:
        print("Computing stats on each parameter...")
        train_ds.compute_parameters_stats()
    if write_valid_samples_list:
        train_ds.write_list_valid_samples()
        valid_ds.write_list_valid_samples()
        test_ds.write_list_valid_samples()

    if compute_stats:
        print("Computing time stats on each parameters, between 2 timesteps...")
        conf["settings"]["standardize"] = True
        train_ds, valid_ds, test_ds = TitanDataset.from_dict(
            path_config.stem,
            conf,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
        )
        train_ds.compute_time_step_stats()


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG):
    """Describes Titan."""
    train_ds, _, _ = TitanDataset.from_json(path_config, 2, 1, 5)
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


@app.command()
def plot(path_config: Path = DEFAULT_CONFIG):
    """Plots a png and a gif for one sample."""
    train_ds, _, _ = TitanDataset.from_json(path_config, 2, 1, 5)
    print("Plot gif of one sample...")
    sample = train_ds.sample_list[0]
    sample.plot_gif("test.gif")
    print("Plot png for one step of sample...")
    item = sample.load(no_standardize=True)
    sample.plot(item, 0, "test.png")


@app.command()
def speedtest(path_config: Path = DEFAULT_CONFIG, n_iter: int = 5):
    """Makes a loading speed test."""
    train_ds, _, _ = TitanDataset.from_json(path_config, 2, 1, 5)
    data_iter = iter(train_ds.torch_dataloader())
    print("Dataset file_format: ", train_ds.settings.file_format)
    print("Speed test:")
    start_time = time.time()
    for _ in tqdm.trange(n_iter, desc="Loading samples"):
        next(data_iter)
    delta = time.time() - start_time
    print("Elapsed time : ", delta)
    speed = n_iter / delta
    print(f"Loading speed: {round(speed, 3)} batch(s)/sec")


if __name__ == "__main__":
    app()
