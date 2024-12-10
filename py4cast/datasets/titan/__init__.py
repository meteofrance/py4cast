import datetime as dt
import json
import time
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union

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
    Grid,
    GridConfig,
    Item,
    NamedTensor,
    ParamConfig,
    Period,
    Sample,
    SamplePreprocSettings,
    Stats,
    Timestamps,
    TorchDataloaderSettings,
    WeatherParam,
    collate_fn,
    get_param_list,
)
from py4cast.datasets.titan.settings import (
    DEFAULT_CONFIG,
    FORMATSTR,
    METADATA,
    SCRATCH_PATH,
)
from py4cast.plots import DomainInfo
from py4cast.utils import merge_dicts

app = typer.Typer()


def get_weight_per_lvl(
    level: int,
    level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
):
    if level_type == "isobaricInhPa":
        return 1 + (level) / (1000)
    else:
        return 2


#############################################################
#                            GRID                           #
#############################################################


def load_grid_info(name: str) -> GridConfig:
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


def get_grid_coords(param: WeatherParam) -> List[int]:
    return METADATA["GRIDS"][param.grid.name]["extent"]


def get_filepath(
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
        npy_path = get_dataset_path(ds_name, param.grid) / "data"
        filename = f"{param.name}_{param.level}_{param.level_type}.npy"
        return npy_path / date.strftime(FORMATSTR) / filename


def process_sample_dataset(ds_name: str, date: dt.datetime, params: List[WeatherParam]):
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


def load_data_from_disk(
    ds_name: str,
    param: WeatherParam,
    date: dt.datetime,
    # the member parameter is not accessed if irrelevant
    member: int = 0,
    file_format: Literal["npy", "grib"] = "grib",
):
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
    param: WeatherParam,
    timestamps: Timestamps,
    file_format: Literal["npy", "grib"] = "grib",
) -> bool:
    for date in timestamps.validity_times:
        filepath = get_filepath(ds_name, param, date, file_format)
        if not filepath.exists():
            return False
    return True


def valid_timestamp(n_inputs: int, timestamps: Timestamps) -> bool:
    # avoiding duplicating samples on days border (00h00 +/ -1 <-> 24h00 +/- 1)
    term_0 = timestamps.terms[n_inputs - 1]
    if term_0 > np.timedelta64(23, "h"):
        return False
    return True


def get_param_tensor(
    param: WeatherParam,
    stats: Stats,
    timestamps: Timestamps,
    settings: SamplePreprocSettings,
    standardize: bool = True,
    member: int = 0,
) -> torch.tensor:
    """
    Fetch data on disk fo the given parameter and all involved dates
    Unless specified, normalize the samples with parameter-specific constants
    returns a tensor
    """
    dates = timestamps.validity_times
    arrays = [
        load_data_from_disk(
            settings.dataset_name, param, date, member, settings.file_format
        )
        for date in dates
    ]
    arr = np.stack(arrays)
    # Extend dimension to match 3D (level dimension)
    if len(arr.shape) != 4:
        arr = np.expand_dims(arr, axis=1)
    arr = np.transpose(arr, axes=[0, 2, 3, 1])  # shape = (steps, lvl, x, y)
    if standardize:
        name = param.parameter_short_name
        means = np.asarray(stats[name]["mean"])
        std = np.asarray(stats[name]["std"])
        arr = (arr - means) / std
    return torch.from_numpy(arr)


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
        params: List[WeatherParam],
        settings: SamplePreprocSettings,
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
    def dataset_info(self) -> DatasetInfo:
        """Returns a DatasetInfo object describing the dataset.

        Returns:
            DatasetInfo: _description_
        """
        shortnames = {
            "input": self.shortnames("input"),
            "input_output": self.shortnames("input_output"),
            "output": self.shortnames("output"),
        }
        return DatasetInfo(
            name=str(self),
            domain_info=self.domain_info,
            shortnames=shortnames,
            units=self.units,
            weather_dim=self.weather_dim,
            forcing_dim=self.forcing_dim,
            step_duration=self.period.step_duration,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    @cached_property
    def sample_list(self):
        """Creates the list of samples."""
        print("Start creating samples...")
        stats = self.stats if self.settings.standardize else None

        n_inputs, n_preds, step_duration = (
            self.settings.num_input_steps,
            self.settings.num_pred_steps,
            self.period.step_duration,
        )

        sample_timesteps = [
            step_duration * step for step in range(-n_inputs + 1, n_preds + 1)
        ]
        all_timestamps = []
        for date in tqdm.tqdm(self.period.date_list):
            for term in self.period.terms_list:
                t0 = date + dt.timedelta(hours=int(term))
                validity_times = [
                    t0 + dt.timedelta(hours=ts) for ts in sample_timesteps
                ]
                terms = [dt.timedelta(hours=int(t + term)) for t in sample_timesteps]

                timestamps = Timestamps(
                    datetime=date,
                    terms=np.array(terms),
                    validity_times=validity_times,
                )
                if valid_timestamp(n_inputs, timestamps):
                    all_timestamps.append(timestamps)

        samples = []
        for ts in all_timestamps:
            for member in self.settings.members:
                sample = Sample(
                    ts,
                    self.settings,
                    self.params,
                    stats,
                    self.grid,
                    exists,
                    get_param_tensor,
                    member,
                )
                if sample.is_valid():
                    samples.append(sample)

        print(f"--> All {len(samples)} {self.period.name} samples are now defined")
        return samples

    @cached_property
    def dataset_extra_statics(self):
        """Add the LandSea Mask to the statics."""
        return [
            NamedTensor(
                feature_names=["LandSeaMask"],
                tensor=torch.from_numpy(self.grid.landsea_mask)
                .type(torch.float32)
                .unsqueeze(2),
                names=["lat", "lon", "features"],
            )
        ]

    def __len__(self):
        return len(self.sample_list)

    @cached_property
    def forcing_dim(self) -> int:
        """Return the number of forcings."""
        res = 4  # For date (hour and year)
        res += 1  # For solar forcing
        for param in self.params:
            if param.kind == "input":
                res += 1
        return res

    @cached_property
    def weather_dim(self) -> int:
        """Return the dimension of pronostic variable."""
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += 1
        return res

    def __getitem__(self, index: int) -> Item:
        sample = self.sample_list[index]
        item = sample.load()
        return item

    @classmethod
    def from_dict(
        cls,
        name: str,
        conf: dict,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
    ) -> Tuple["TitanDataset", "TitanDataset", "TitanDataset"]:

        conf["grid"]["load_grid_info_func"] = load_grid_info
        grid = Grid(**conf["grid"])

        param_list = get_param_list(conf, grid, load_param_info, get_weight_per_lvl)

        train_settings = SamplePreprocSettings(
            name, num_input_steps, num_pred_steps_train, **conf["settings"]
        )
        train_period = Period(**conf["periods"]["train"], name="train")
        train_ds = TitanDataset(name, grid, train_period, param_list, train_settings)

        valid_settings = SamplePreprocSettings(
            name, num_input_steps, num_pred_steps_val_test, **conf["settings"]
        )
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        valid_ds = TitanDataset(name, grid, valid_period, param_list, valid_settings)

        test_period = Period(**conf["periods"]["test"], name="test")
        test_ds = TitanDataset(name, grid, test_period, param_list, valid_settings)
        return train_ds, valid_ds, test_ds

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple["TitanDataset", "TitanDataset", "TitanDataset"]:
        with open(fname, "r") as fp:
            conf = json.load(fp)
            if config_override is not None:
                conf = merge_dicts(conf, config_override)
        return cls.from_dict(
            fname.stem,
            conf,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
        )

    def __str__(self) -> str:
        return f"titan_{self.grid.name}"

    def torch_dataloader(
        self, tl_settings: TorchDataloaderSettings = TorchDataloaderSettings()
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=tl_settings.batch_size,
            num_workers=tl_settings.num_workers,
            shuffle=self.shuffle,
            prefetch_factor=tl_settings.prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=tl_settings.pin_memory,
        )

    @property
    def meshgrid(self) -> np.array:
        """array of shape (2, num_lat, num_lon) of (X, Y) values"""
        return self.grid.meshgrid

    @property
    def geopotential_info(self) -> np.array:
        """array of shape (num_lat, num_lon) with geopotential value for each datapoint"""
        return self.grid.geopotential

    @property
    def border_mask(self) -> np.array:
        return self.grid.border_mask

    def shortnames(
        self,
        kind: List[Literal["input", "output", "input_output"]] = [
            "input",
            "output",
            "input_output",
        ],
    ) -> List[str]:
        """
        List of readable names for the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return [p.parameter_short_name for p in self.params if p.kind == kind]

    @cached_property
    def units(self) -> Dict[str, str]:
        """
        Return a dictionnary with name and units
        """
        return {p.parameter_short_name: p.unit for p in self.params}

    @cached_property
    def state_weights(self):
        """Weights used in the loss function."""
        kinds = ["output", "input_output"]
        return {
            p.parameter_short_name: p.state_weight
            for p in self.params
            if p.kind in kinds
        }

    @property
    def cache_dir(self) -> Path:
        return get_dataset_path(self.name, self.grid)

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered. Usefull information for plotting."""
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )


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
