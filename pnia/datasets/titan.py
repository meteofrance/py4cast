import datetime as dt
import json
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import cartopy
import numpy as np
import pandas as pd
import torch
import tqdm
import xarray as xr
import yaml
from cyeccodes import nested_dd_iterator
from cyeccodes.eccodes import get_multi_messages_from_file
from torch.utils.data import DataLoader, Dataset

from pnia.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Item,
    NamedTensor,
    TorchDataloaderSettings,
    collate_fn,
)
from pnia.plots import DomainInfo
from pnia.settings import CACHE_DIR

SCRATCH_PATH = Path("/scratch/shared/Titan")
FORMATSTR = "%Y-%m-%d_%Hh%M"
# Assuming no leap years in dataset (2024 is next)
SECONDS_IN_YEAR = (
    365 * 24 * 60 * 60
)  # TODO: coder une fonction qui le calcule selon l'année

with open(SCRATCH_PATH / "metadata.yaml", "r") as file:
    METADATA = yaml.safe_load(file)

DEFAULT_CONFIG = Path(__file__).parents[2] / "config" / "titan.json"


def get_weight_per_lvl(level: int, kind: Literal["hPa", "m"]):
    if kind == "hPa":
        return 1 + (level) / (90)
    else:
        return 2


@lru_cache(256)
def read_grib(path_grib: Path, names=None, levels=None):
    if names or levels:
        include_filters = {
            k: v for k, v in [("cfVarName", names), ("level", levels)] if v is not None
        }
    else:
        include_filters = None
    _, results = get_multi_messages_from_file(
        path_grib,
        storage_keys=("cfVarName", "level"),
        include_filters=include_filters,
        metadata_keys=("missingValue", "Ni", "Nj"),
        include_latlon=False,
    )

    grib_dict = {}
    for metakey, result in nested_dd_iterator(results):
        array = result["values"]
        grid = (result["metadata"]["Nj"], result["metadata"]["Ni"])
        mv = result["metadata"]["missingValue"]
        array = np.reshape(array, grid)
        array = np.where(array == mv, np.nan, array)
        name, level = metakey.split("-")
        level = int(level)
        grib_dict[name] = {}
        grib_dict[name][level] = array
    return grib_dict


#############################################################
#                            PERIOD                         #
#############################################################


@dataclass(slots=True)
class Period:
    start: dt.datetime
    end: dt.datetime
    step: int  # In hours
    name: str

    def __init__(self, start: int, end: int, step: int, name: str):
        self.start = dt.datetime.strptime(str(start), "%Y%m%d%H")
        self.end = dt.datetime.strptime(str(end), "%Y%m%d%H")
        self.step = step
        self.name = name

    @property
    def date_list(self):
        return pd.date_range(
            start=self.start, end=self.end, freq=f"{self.step}H"
        ).to_pydatetime()


#############################################################
#                            GRID                           #
#############################################################


@dataclass
class Grid:
    name: Literal["ANTJP7CLIM_1S100", "PAAROME_1S100", "PAAROME_1S40", "PA_01D"]
    border_size: int = 10
    # Subgrid selection: If (0,0,0,0) the whole grid is kept.
    subgrid: Tuple[int] = (0, 0, 0, 0)
    x: int = field(init=False)  # X dimension
    y: int = field(init=False)  # Y dimension

    def __post_init__(self):
        grid_info = METADATA["GRIDS"][self.name]
        x, y = grid_info["size"]

        # Setting correct subgrid if no subgrid is selected.
        if sum(self.subgrid) == 0:
            self.subgrid = (0, x, 0, y)
        self.x = self.subgrid[1] - self.subgrid[0]
        self.y = self.subgrid[3] - self.subgrid[2]

    @cached_property
    def lat(self) -> np.array:
        conf_ds = xr.open_dataset(SCRATCH_PATH / "conf.grib")
        latitudes = conf_ds.latitude[self.subgrid[0] : self.subgrid[1]]
        return np.transpose(np.tile(latitudes, (self.y, 1)))

    @cached_property
    def lon(self) -> np.array:
        conf_ds = xr.open_dataset(SCRATCH_PATH / "conf.grib")
        longitudes = conf_ds.longitude[self.subgrid[2] : self.subgrid[3]]
        return np.tile(longitudes, (self.x, 1))

    # TODO : à partir du grib, enregistrer un fichier npy lat lon h mask pour chacune des grilles

    @property
    def geopotential(self) -> np.array:
        conf_ds = xr.open_dataset(SCRATCH_PATH / "conf.grib")
        return conf_ds.h.values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @property
    def landsea_mask(self) -> np.array:
        return np.zeros((self.x, self.y))  # TODO

    @property
    def border_mask(self) -> np.array:
        if self.border_size > 0:
            border_mask = np.ones((self.x, self.y)).astype(bool)
            size = self.border_size
            border_mask[size:-size, size:-size] *= False
        elif self.border_size == 0:
            border_mask = np.ones((self.x, self.y)).astype(bool) * False
        else:
            raise ValueError(f"Bordersize should be positive. Get {self.border_size}")
        return border_mask

    @property
    def N_grid(self) -> int:
        return self.x * self.y

    @cached_property
    def grid_limits(self):
        conf_ds = xr.open_dataset(SCRATCH_PATH / "conf.grib")
        grid_limits = [  # In projection
            float(conf_ds.latitude[self.subgrid[0]].values),  # min x
            float(conf_ds.latitude[self.subgrid[1] - 1].values),  # max x
            float(conf_ds.longitude[self.subgrid[2]].values),  # min y
            float(conf_ds.longitude[self.subgrid[3] - 1].values),  # max y
        ]
        return grid_limits

    @cached_property
    def meshgrid(self) -> np.array:
        """Build a meshgrid from coordinates position."""
        conf_ds = xr.open_dataset(SCRATCH_PATH / "conf.grib")
        latitudes = conf_ds.latitude[self.subgrid[0] : self.subgrid[1]]
        longitudes = conf_ds.longitude[self.subgrid[2] : self.subgrid[3]]
        meshgrid = np.array(np.meshgrid(longitudes, latitudes))
        return meshgrid  # shape (2, x, y)

    @cached_property
    def projection(self):
        return cartopy.crs.PlateCarree()


#############################################################
#                            PARAM                          #
#############################################################


@dataclass(slots=True)
class Param:
    name: str
    levels: Tuple[int]  # desired levels
    grid: Grid
    # Parameter status :
    # input = forcings, output = diagnostic, input_output = classical weather var
    kind: Literal["input", "output", "input_output"]
    level_type: str = field(init=False)
    long_name: str = field(init=False)
    unit: str = field(init=False)
    native_grid: str = field(init=False)
    grib_name: str = field(init=False)
    grib_param: str = field(init=False)

    def __post_init__(self):
        param_info = METADATA["WEATHER_PARAMS"][self.name]
        self.grib_name = param_info["grib"]
        self.grib_param = param_info["param"]
        self.unit = param_info["unit"]
        if param_info["type_level"] in ["heightAboveGround", "meanSea", "surface"]:
            self.level_type = "m"
        else:
            self.level_type = "hPa"
        self.long_name = param_info["long_name"]
        self.native_grid = param_info["grid"]
        if self.grid.name != "PAAROME_1S100":
            raise NotImplementedError

    @property
    def number(self) -> int:
        """Get the number of parameters."""
        return len(self.levels)

    @property  # Does not accept a cached property with slots=True
    def ndims(self) -> str:
        if len(self.levels) > 1:
            return 3
        else:
            return 2

    @property
    def state_weights(self) -> list:
        return [get_weight_per_lvl(level, self.level_type) for level in self.levels]

    @property
    def parameter_name(self) -> list:
        return [f"{self.long_name} {level}{self.level_type}" for level in self.levels]

    @property
    def parameter_short_names(self) -> list:
        # TODO : add type of lvl in name
        return [f"{self.name}_{level}{self.level_type}" for level in self.levels]

    @property
    def units(self) -> list:
        """For a given variable, the unit is the same accross all levels."""
        return [self.unit for _ in self.levels]

    def get_filepath(
        self, date: dt.datetime, file_format: Literal["npy", "grib"]
    ) -> Path:
        """Return the file path."""
        folder = SCRATCH_PATH / file_format / date.strftime(FORMATSTR)
        if file_format == "npy":
            return folder / f"{self.name}.npy"
        else:
            return folder / self.grib_name

    def load_data_npy(self, date: dt.datetime) -> np.ndarray:
        array = np.load(self.get_filepath(date, "npy"))
        subgrid = self.grid.subgrid
        array = array[subgrid[0] : subgrid[1], subgrid[2] : subgrid[3]]
        if self.native_grid != self.grid.name:
            # TODO : interpolation d'un champ sur une grille différente
            raise NotImplementedError(
                f"Unable to load data with grid {self.native_grid}"
            )
        # TODO : select only desired levels -> save dataset as npz with dico level
        return array

    def load_data_grib(self, date: dt.datetime) -> np.ndarray:
        path_grib = self.get_filepath(date, "grib")
        param_dict = read_grib(path_grib)[self.grib_param]
        array = param_dict[
            self.levels[0]
        ]  # warning : doesn't work with multiple levels for now
        subgrid = self.grid.subgrid
        return array[subgrid[0] : subgrid[1], subgrid[2] : subgrid[3]]

    def load_data(
        self, date: dt.datetime, file_format: Literal["npy", "grib"] = "grib"
    ):
        if file_format == "npy":
            return self.load_data_npy(date)
        else:
            return self.load_data_grib(date)

    def exist(self, date: dt.datetime, file_format: Literal["npy", "grib"] = "grib"):
        filepath = self.get_filepath(date, file_format)
        return filepath.exists()


#############################################################
#                            SETTINGS                       #
#############################################################


@dataclass(slots=True)
class TitanSettings:
    num_input_steps: int  # Number of input timesteps
    num_pred_steps: int  # Number of output timesteps
    step_duration: float  # duration in hour
    standardize: bool = True
    file_format: Literal["npy", "grib"] = "grib"


#############################################################
#                            SAMPLE                         #
#############################################################


@dataclass(slots=True)
class Sample:
    """Describes a sample"""

    date_t0: dt.datetime
    settings: TitanSettings
    terms: Tuple[float] = field(init=False)  # gap in hours btw step and t0
    dates: Tuple[dt.datetime] = field(init=False)  # date of each step

    def __post_init__(self):
        n_inputs, n_preds = self.settings.num_input_steps, self.settings.num_pred_steps
        steps = list(range(-n_inputs + 1, n_preds + 1))
        self.terms = [self.settings.step_duration * step for step in steps]
        self.dates = [self.date_t0 + dt.timedelta(hours=term) for term in self.terms]

    def __repr__(self):
        return f"Date T0 {self.date}, terms {self.terms}"

    @property
    def hours_of_day(self) -> np.array:
        """Hour of the day for each step. This is a float."""
        hours = [date.hour + date.minute / 60 for date in self.dates]
        return np.asarray(hours)

    @property
    def seconds_from_start_of_year(self) -> np.array:
        """Second from the start of the year for each step."""
        start_of_year = dt.datetime(self.date_t0.year, 1, 1)
        seconds = [(date - start_of_year).total_seconds() for date in self.dates]
        return np.asarray(seconds)

    def is_valid(self, param_list: List[Param]) -> bool:
        """Check that all the files necessary for this sample exist.

        Args:
            param_list (List): List of parameters
        Returns:
            Boolean: Whether the sample is available or not
        """
        for date in self.dates:
            for param in param_list:
                if not param.exist(date, self.settings.file_format):
                    return False
        return True


#############################################################
#                            DATASET                        #
#############################################################


class TitanDataset(DatasetABC, Dataset):
    # Si on doit travailler avec plusieurs grilles, on fera un super dataset qui contient
    # plusieurs datasets chacun sur une seule grille
    def __init__(
        self, grid: Grid, period: Period, params: List[Param], settings: TitanSettings
    ):
        self.grid = grid
        if grid.name != "PAAROME_1S100":
            raise NotImplementedError
        self.period = period
        self.params = params
        self.settings = settings
        self._cache_dir = CACHE_DIR / "datasets" / str(self)
        self.shuffle = self.split == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def dataset_info(self) -> DatasetInfo:
        """Returns a DatasetInfo object describing the dataset.

        Returns:
            DatasetInfo: _description_
        """
        return DatasetInfo(
            name=str(self),
            domain_info=self.domain_info,
            shortnames=self.shortnames,
            units=self.units,
            weather_dim=self.weather_dim,
            forcing_dim=self.forcing_dim,
            step_duration=self.settings.step_duration,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    @cached_property
    def sample_list(self):
        """Creates the list of samples."""
        print("Start forming samples...")
        samples = [Sample(date, self.settings) for date in self.period.date_list]
        samples = [sample for sample in samples if sample.is_valid(self.params)]
        print("All samples are now defined")
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

    def get_year_hour_forcing(self, sample: Sample):
        """Get the forcing term dependent of the sample time"""
        hour_angle = (torch.Tensor(sample.hours_of_day) / 12) * torch.pi  # (sample_len)
        seconds_from_start_year = torch.Tensor(sample.seconds_from_start_of_year)
        # Keep only pred steps for forcing
        hour_angle = hour_angle[-self.settings.num_pred_steps :]
        seconds_from_start_year = seconds_from_start_year[
            -self.settings.num_pred_steps :
        ]
        year_angle = (seconds_from_start_year / SECONDS_IN_YEAR) * 2 * torch.pi
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (sample_len, 4)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        return datetime_forcing

    @cached_property
    def forcing_dim(self) -> int:
        """Return the number of forcings."""
        res = 4  # For date (hour and year)
        for param in self.params:
            if param.kind == "input":
                res += param.number
        return res

    @cached_property
    def weather_dim(self) -> int:
        """Return the dimension of pronostic variable."""
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += param.number
        return res

    @cached_property
    def diagnostic_dim(self):
        """
        Return dimensions of output variable only
        Not used yet
        """
        res = 0
        for param in self.params:
            if param.kind == "output":
                res += param.number
        return res

    def get_param_tensor(self, param: Param, dates: List[dt.datetime]) -> torch.tensor:
        if self.settings.standardize:
            names = param.parameter_short_names
            means = np.asarray([self.stats[name]["mean"] for name in names])
            std = np.asarray([self.stats[name]["std"] for name in names])
        arrays = [param.load_data(date, self.settings.file_format) for date in dates]
        array = np.stack(arrays)
        # Extend dimension to match 3D (level dimension)
        if len(array.shape) != 4:
            array = np.expand_dims(array, axis=1)
        array = np.transpose(array, axes=[0, 2, 3, 1])  # shape = (steps, lvl, x, y)
        if self.settings.standardize:
            array = (array - means) / std
        return torch.from_numpy(array)

    def __getitem__(self, index):
        sample = self.sample_list[index]

        # Datetime Forcing
        lforcings = [
            NamedTensor(
                feature_names=[
                    "cos_hour",
                    "sin_hour",
                    "cos_doy",
                    "sin_doy",
                ],  # doy : day_of_year
                tensor=self.get_year_hour_forcing(sample).type(torch.float32),
                names=["out_step", "features"],
            )
        ]
        linputs = []
        loutputs = []

        # Reading parameters from files
        for param in self.params:
            state_kwargs = {
                "feature_names": param.parameter_short_names,
                "names": ["out_step", "lat", "lon", "features"],
            }

            if param.kind == "input":
                # forcing is taken for every predicted step
                dates = sample.dates[-self.settings.num_pred_steps :]
                tensor = self.get_param_tensor(param, dates)
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))
                lforcings.append(tmp_state)

            elif param.kind == "output":
                dates = sample.dates[-self.settings.num_pred_steps :]
                tensor = self.get_param_tensor(param, dates)
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))
                loutputs.append(tmp_state)

            else:  # input_output
                tensor = self.get_param_tensor(param, sample.dates)
                state_kwargs["names"][0] = "out_step"
                tmp_state = NamedTensor(
                    tensor=tensor[-self.settings.num_pred_steps :],
                    **deepcopy(state_kwargs),
                )

                loutputs.append(tmp_state)
                state_kwargs["names"][0] = "in_step"
                tmp_state = NamedTensor(
                    tensor=tensor[: self.settings.num_input_steps],
                    **deepcopy(state_kwargs),
                )
                linputs.append(tmp_state)
        return Item(inputs=linputs, outputs=loutputs, forcing=lforcings)

    @classmethod
    def from_dict(
        cls,
        conf: dict,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
    ) -> Tuple["TitanDataset", "TitanDataset", "TitanDataset"]:
        grid = Grid(**conf["grid"])
        param_list = [
            Param(name=name, levels=values["levels"], kind=values["kind"], grid=grid)
            for name, values in conf["params"].items()
        ]

        train_settings = TitanSettings(
            num_input_steps, num_pred_steps_train, **conf["settings"]
        )
        train_period = Period(**conf["periods"]["train"], name="train")
        train_ds = TitanDataset(grid, train_period, param_list, train_settings)

        valid_settings = TitanSettings(
            num_input_steps, num_pred_steps_val_test, **conf["settings"]
        )
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        valid_ds = TitanDataset(grid, valid_period, param_list, valid_settings)

        test_period = Period(**conf["periods"]["test"], name="test")
        test_ds = TitanDataset(grid, test_period, param_list, valid_settings)
        return train_ds, valid_ds, test_ds

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
    ) -> Tuple["TitanDataset", "TitanDataset", "TitanDataset"]:
        with open(fname, "r") as fp:
            conf = json.load(fp)
        return cls.from_dict(
            conf, num_input_steps, num_pred_steps_train, num_pred_steps_val_test
        )

    def __str__(self) -> str:
        return f"titan_{self.grid.name}"

    def torch_dataloader(
        self, tl_settings: TorchDataloaderSettings = TorchDataloaderSettings()
    ) -> DataLoader:
        return DataLoader(
            self,
            tl_settings.batch_size,
            num_workers=tl_settings.num_workers,
            shuffle=self.shuffle,
            prefetch_factor=tl_settings.prefetch_factor,
            collate_fn=collate_fn,
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
    def limited_area(self) -> bool:
        """Returns True if the dataset is compatible with Limited area models"""
        return True

    @property
    def border_mask(self) -> np.array:
        return self.grid.border_mask

    @property
    def split(self) -> Literal["train", "valid", "test"]:
        return self.period.name

    def shortnames(
        self,
        kind: List[Literal["input", "output", "input_output"]] = [
            "input",
            "output",
            "input_output",
        ],
    ) -> List[str]:
        """
        Return the name of the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        names = []
        for param in self.params:
            if param.kind in kind:
                names += param.parameter_short_names
        return names

    @cached_property
    def units(self) -> Dict[str, int]:
        """
        Return a dictionnary with name and units
        """
        dout = {}
        for param in self.params:
            names = param.parameter_short_names
            units = param.units
            for name, unit in zip(names, units):
                dout[name] = unit
        return dout

    @cached_property
    def state_weights(self):
        """Weights used in the loss function."""
        w_dict = {}
        for param in self.params:
            if param.kind in ["output", "input_output"]:
                for name, weight in zip(
                    param.parameter_short_names, param.state_weights
                ):
                    w_dict[name] = weight

        return w_dict

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered. Usefull information for plotting."""
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )

    @classmethod
    def prepare(cls, path_config: Path):
        print("--> Preparing Titan Dataset...")

        print("Load train dataset configuration...")
        with open(path_config, "r") as fp:
            conf = json.load(fp)

        print("Computing stats on each parameter...")
        conf["settings"]["standardize"] = False
        train_ds, _, _ = TitanDataset.from_dict(conf, 2, 3, 3)
        train_ds.compute_parameters_stats()

        print("Computing time stats on each parameters, between 2 timesteps...")
        conf["settings"]["standardize"] = True
        train_ds, _, _ = TitanDataset.from_dict(conf, 2, 3, 3)
        train_ds.compute_time_step_stats()


# TODO :
# - pouvoir gérer plusieurs niveaux


if __name__ == "__main__":
    import time
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Prepare Titan dataset.")
    parser.add_argument(
        "--path_config",
        default=DEFAULT_CONFIG,
        type=Path,
        help="Configuration file for the dataset.",
    )
    args = parser.parse_args()

    TitanDataset.prepare(args.path_config)

    print("Dataset info : ")
    train_ds, _, _ = TitanDataset.from_json(args.path_config, 2, 3, 3)
    train_ds.dataset_info.summary()

    print("Test __get_item__")
    print("Len dataset : ", len(train_ds))

    beg = time.time()
    for i in tqdm.tqdm(range(5)):
        item = train_ds[i]
    print("Time to load 5 sample : ", time.time() - beg)
    print("Last Item description :")
    print(item)
