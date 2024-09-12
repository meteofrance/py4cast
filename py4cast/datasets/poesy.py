import datetime as dt
import json
import time
from argparse import ArgumentParser
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import cartopy
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from py4cast.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Item,
    NamedTensor,
    TorchDataloaderSettings,
    collate_fn,
)
from py4cast.plots import DomainInfo
from py4cast.settings import CACHE_DIR
from py4cast.utils import merge_dicts

SCRATCH_PATH = Path("/scratch/shared/poesy/poesy_crop")
OROGRAPHY_FNAME = "PEARO_EURW1S40_Orography_crop.npy"
LATLON_FNAME = "latlon_crop.npy"

# Shape of cropped poesy data (lon x lat x leadtimes x members)
DATA_SHAPE = (600, 600, 45, 16)

# Assuming no leap years in dataset (2024 is next)
SECONDS_IN_YEAR = 365 * 24 * 60 * 60


def poesy_forecast_namer(date: dt.datetime, shortname, **kwargs):
    """
    use to find local files
    """
    return f"{date.strftime('%Y-%m-%dT%H:%M:%SZ')}_{shortname}_lt1-45_crop.npy"


def get_weight(level: int, kind: str):
    if kind == "hPa":
        return 1 + (level) / (90)
    elif kind == "m":
        return 2
    else:
        raise Exception(f"unknown kind:{kind}, must be hPa or m right now")


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


# Define static attributes to add -> see Grid class in smeagol.py
@dataclass
class Grid:
    domain: str
    model: str
    geometry: str = "EURW1S40"
    border_size: int = 0
    # Subgrid selection. If (0,0,0,0) the whole grid is kept.
    subgrid: Tuple[int] = (0, 0, 0, 0)
    x: int = field(init=False)  # X dimension
    y: int = field(init=False)  # Y dimension

    def __post_init__(self):
        ds = np.load(self.grid_filepath)
        x, y = ds.shape
        # Setting correct subgrid if no subgrid is selected.
        if self.subgrid == (0, 0, 0, 0):
            self.subgrid = (0, x, 0, y)
        self.x = self.subgrid[1] - self.subgrid[0]
        self.y = self.subgrid[3] - self.subgrid[2]

    @property
    def grid_filepath(self):
        return SCRATCH_PATH / OROGRAPHY_FNAME

    @cached_property
    def lat(self) -> np.array:
        filename = SCRATCH_PATH / LATLON_FNAME
        with open(filename, "rb") as f:
            lonlat = np.load(f)
        return lonlat[
            0, self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @cached_property
    def lon(self) -> np.array:
        filename = SCRATCH_PATH / LATLON_FNAME
        with open(filename, "rb") as f:
            lonlat = np.load(f)
        return lonlat[
            1, self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @cached_property
    def meshgrid(self) -> np.array:
        """
        Build a meshgrid from coordinates position.
        """
        return np.asarray([self.lat, self.lon])

    @property
    def geopotential(self) -> np.array:
        ds = np.load(self.grid_filepath)
        return ds[self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]]

    @property
    def landsea_mask(self) -> np.array:
        # TO DO : add landsea mask instead of orography
        ds = np.load(self.grid_filepath)
        return ds[self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]]

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
        return [  # In projection
            self.lon[0, 0],  # min x
            self.lon[-1, -1],  # max x
            self.lat[0, 0],  # min y
            self.lat[-1, -1],  # max y
        ]

    @cached_property
    def projection(self):
        # Create projection
        return cartopy.crs.LambertConformal(central_longitude=2, central_latitude=46.7)


@dataclass(slots=True)
class Param:
    name: str
    shortname: str
    levels: Tuple[int]
    grid: Grid  # Parameter grid.
    # It is not necessarly the same as the model grid.
    # Function which can return the filenames.
    # It should accept member and date as argument (as well as term).
    fnamer: Callable[[], [str]]  # VSCode doesn't like this, is it ok ?
    level_type: str = "hPa"  # To be read in nc file ?
    kind: Literal["input", "output", "input_output"] = "input_output"
    unit: str = "FakeUnit"  # To be read in nc FIle  ?
    ndims: int = 2

    @property
    def number(self) -> int:
        """
        Get the number of parameters.
        """
        return len(self.levels)

    @property
    def state_weights(self) -> list:
        return [get_weight(level, self.level_type) for level in self.levels]

    @property
    def parameter_name(self) -> list:
        return [f"{self.name}_{level}" for level in self.levels]

    @property
    def parameter_short_name(self) -> list:
        return [f"{self.shortname}_{level}" for level in self.levels]

    @property
    def units(self) -> list:
        """
        For a given variable, the unit is the
        same accross all levels.
        """
        return [self.unit for _ in self.levels]

    def filename(self, date: dt.datetime) -> str:
        """
        Return the filename.
        """

        return SCRATCH_PATH / self.fnamer(date=date, shortname=self.shortname)

    def load_data(self, date: dt.datetime, term: List, member: int) -> np.array:
        """
        date : Date of file.
        term : Position of leadtimes in file.
        """
        data_array = np.memmap(
            self.filename(date=date), dtype="float32", mode="r", shape=DATA_SHAPE
        )

        return data_array[
            self.grid.subgrid[0] : self.grid.subgrid[1],
            self.grid.subgrid[2] : self.grid.subgrid[3],
            term,
            member,
        ]

    def exist(self, date: dt.datetime) -> bool:
        flist = self.filename(date=date)
        return flist.exists()


@dataclass(slots=True)
class PoesySettings:
    term: dict
    num_input_steps: int  # = 2  # Number of input timesteps
    num_output_steps: int  # = 1  # Number of output timesteps (= 0 for inference)
    num_inference_pred_steps: int = 0  # 0 in training config ; else used to provide future information about forcings
    standardize: bool = False
    members: Tuple[int] = (0,)

    @property
    def num_total_steps(self) -> int:
        """
        Total number of timesteps
        for one sample.
        """
        # Nb of step in one sample
        return self.num_input_steps + self.num_output_steps


@dataclass(slots=True)
class Sample:
    # Describe a sample
    # TODO consider members
    member: int
    settings: PoesySettings
    date: dt.datetime
    input_terms: Tuple[float]
    output_terms: Tuple[float]

    # Term wrt to the date {date}. Gives validity
    terms: Tuple[float] = field(init=False)

    def __post_init__(self):
        self.terms = self.input_terms + self.output_terms

    @property
    def hours_of_day(self) -> np.array:
        """
        Hour of the day.
        For output terms only. This is a float.
        """
        hours = []
        for term in self.output_terms:
            date_tmp = self.date + dt.timedelta(hours=float(term))
            hours.append(date_tmp.hour + date_tmp.minute / 60)
        return np.asarray(hours)

    @property
    def seconds_from_start_of_year(self) -> np.array:
        """
        Second from the start of the year.
        For output terms only
        """
        start_of_year = dt.datetime(self.date.year, 1, 1)
        return np.asarray(
            [
                (
                    self.date + dt.timedelta(hours=float(term)) - start_of_year
                ).total_seconds()
                for term in self.output_terms
            ]
        )

    def is_valid(self, param_list: List) -> bool:
        """
        Check that all the files necessary for this samples exists.

        Args:
            param_list (List): List of parameters
        Returns:
            Boolean:  Whether the sample exist or not
        """

        for param in param_list:

            if not param.exist(self.date):
                return False

        return True


class InferSample(Sample):
    """
    Sample dedicated to inference. No outputs terms, only inputs.
    """

    def __post_init__(self):
        self.terms = self.input_terms


class PoesyDataset(DatasetABC, Dataset):
    def __init__(
        self, grid: Grid, period: Period, params: List[Param], settings: PoesySettings
    ):
        self.grid = grid
        self.period = period
        self.params = params
        self.settings = settings
        self._cache_dir = CACHE_DIR / str(self)
        self.shuffle = self.split == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.step_duration = self.settings.term["timestep"]

    @cached_property
    def cache_dir(self):
        return self._cache_dir

    def __str__(self) -> str:
        return f"Poesy_{self.grid.geometry}"

    def __len__(self):
        return len(self.sample_list)

    @cached_property
    def dataset_info(self) -> DatasetInfo:
        """
        Return a DatasetInfo object.
        This object describes the dataset.

        Returns:
            DatasetInfo: _description_
        """

        shortnames = {
            "forcing": self.shortnames("forcing"),
            "input_output": self.shortnames("input_output"),
            "diagnostic": self.shortnames("diagnostic"),
        }

        return DatasetInfo(
            name=str(self),
            domain_info=self.domain_info,
            units=self.units,
            shortnames=shortnames,
            weather_dim=self.weather_dim,
            forcing_dim=self.forcing_dim,
            step_duration=self.step_duration,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    @cached_property
    def sample_list(self):
        """
        Create a list of sample from information
        """
        print("Start forming samples")
        terms = list(
            np.arange(
                self.settings.term["start"],
                self.settings.term["end"],
                self.settings.term["timestep"],
            )
        )

        sample_by_date = len(terms) // self.settings.num_total_steps

        samples = []
        number = 0

        for date in self.period.date_list:
            for member in self.settings.members:
                for sample in range(0, sample_by_date):

                    input_terms = terms[
                        sample
                        * self.settings.num_total_steps : sample
                        * self.settings.num_total_steps
                        + self.settings.num_input_steps
                    ]
                    output_terms = terms[
                        sample * self.settings.num_total_steps
                        + self.settings.num_input_steps : sample
                        * self.settings.num_total_steps
                        + self.settings.num_input_steps
                        + self.settings.num_output_steps
                    ]
                    samp = Sample(
                        settings=self.settings,
                        date=date,
                        member=member,
                        input_terms=input_terms,
                        output_terms=output_terms,
                    )

                    if samp.is_valid(self.params):

                        samples.append(samp)
                        number += 1
        print("All samples are now defined")

        return samples

    @cached_property
    def dataset_extra_statics(self):
        """
        We add the LandSea Mask to the statics.
        """
        return [
            NamedTensor(
                feature_names=["LandSeaMask"],
                tensor=torch.from_numpy(self.grid.landsea_mask)
                .type(torch.float32)
                .unsqueeze(2),
                names=["lat", "lon", "features"],
            )
        ]

    def get_year_hour_forcing(self, sample: Sample):
        """
        Get the forcing term dependent of the sample time
        """
        hour_angle = (
            torch.Tensor(sample.hours_of_day) / 12
        ) * torch.pi  # (sample_len,)
        year_angle = (
            (torch.Tensor(sample.seconds_from_start_of_year) / SECONDS_IN_YEAR)
            * 2
            * torch.pi
        )  # (sample_len,)
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (N_t, 4)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        return datetime_forcing

    @cached_property
    def forcing_dim(self) -> int:
        """
        Return the number of forcings.
        """
        res = 4  # For date
        for param in self.params:
            if param.kind == "input":
                res += param.number
        return res

    @cached_property
    def weather_dim(self) -> int:
        """
        Return the dimension of pronostic variable.
        """
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

    def get_param_tensor(
        self,
        param: Param,
        date: dt.datetime,
        terms: List,
        member: int = 1,
        inference_steps: int = 0,
    ) -> torch.tensor:

        if self.settings.standardize:
            names = param.parameter_short_name
            means = np.asarray([self.stats[name]["mean"] for name in names])
            std = np.asarray([self.stats[name]["std"] for name in names])

        array = param.load_data(date, terms, member)
        # Extend dimension to match 3D (level dimension)
        if len(array.shape) != 4:
            array = np.expand_dims(array, axis=-1)
        array = np.transpose(array, axes=[2, 0, 1, 3])  # shape = (steps, lvl, x, y)

        if self.settings.standardize:
            array = (array - means) / std

        # Define which value is considered invalid
        tensor_data = torch.from_numpy(array)

        if inference_steps:
            empty_data = torch.empty((inference_steps, *array.shape[1:]))
            tensor_data = torch.cat((tensor_data, empty_data), dim=0)
        return tensor_data

    def __getitem__(self, index):
        sample = self.sample_list[index]

        # Datetime Forcing
        datetime_forcing = self.get_year_hour_forcing(sample).type(torch.float32)
        lforcings = [
            NamedTensor(
                feature_names=[
                    "cos_hour",
                    "sin_hour",
                ],  # doy : day_of_year
                tensor=datetime_forcing[:, :2],
                names=["timestep", "features"],
            ),
            NamedTensor(
                feature_names=[
                    "cos_doy",
                    "sin_doy",
                ],  # doy : day_of_year
                tensor=datetime_forcing[:, 2:],
                names=["timestep", "features"],
            ),
        ]
        linputs = []
        loutputs = []

        # Reading parameters from files
        for param in self.params:
            state_kwargs = {
                "feature_names": param.parameter_short_name,
                "names": ["timestep", "lat", "lon", "features"],
            }
            try:

                if param.kind == "input_output":
                    # Search data for date sample.date and terms sample.terms
                    tensor = self.get_param_tensor(
                        param,
                        sample.date,
                        terms=sample.terms,
                        inference_steps=self.settings.num_inference_pred_steps,
                    )
                    state_kwargs["names"][0] = "timestep"
                    # Save outputs
                    tmp_state = NamedTensor(
                        tensor=tensor[self.settings.num_input_steps :],
                        **deepcopy(state_kwargs),
                    )
                    loutputs.append(tmp_state)
                    # Save inputs
                    tmp_state = NamedTensor(
                        tensor=tensor[: self.settings.num_input_steps],
                        **deepcopy(state_kwargs),
                    )
                    linputs.append(tmp_state)

            except KeyError as e:
                print(f"Error for param {param}")
                raise e

        for lforcing in lforcings:
            lforcing.unsqueeze_and_expand_from_(linputs[0])

        return Item(
            inputs=NamedTensor.concat(linputs),
            outputs=NamedTensor.concat(loutputs),
            forcing=NamedTensor.concat(lforcings),
        )

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple["PoesyDataset", "PoesyDataset", "PoesyDataset"]:
        with open(fname, "r") as fp:
            conf = json.load(fp)

        grid = Grid(**conf["grid"])
        param_list = []
        for data_source in conf["dataset"]:
            data = conf["dataset"][data_source]
            members = conf["dataset"][data_source].get("members", [0])
            term = conf["dataset"][data_source]["term"]
            for var in data["var"]:
                vard = data["var"][var]
                # Change grid definition
                if "level" in vard:
                    level_type = "hPa"
                else:
                    level_type = "m"
                param = Param(
                    name=var,
                    shortname=vard.pop("shortname", "t2m"),
                    levels=vard.pop("level", [0]),
                    grid=grid,
                    level_type=level_type,
                    fnamer=poesy_forecast_namer,
                    **vard,
                )
                param_list.append(param)
        train_period = Period(**conf["periods"]["train"], name="train")
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        test_period = Period(**conf["periods"]["test"], name="test")
        train_ds = PoesyDataset(
            grid,
            train_period,
            param_list,
            PoesySettings(
                members=members,
                term=term,
                num_output_steps=num_pred_steps_train,
                num_input_steps=num_input_steps,
            ),
        )
        valid_ds = PoesyDataset(
            grid,
            valid_period,
            param_list,
            PoesySettings(
                members=members,
                term=term,
                num_output_steps=num_pred_steps_val_test,
                num_input_steps=num_input_steps,
            ),
        )
        test_ds = PoesyDataset(
            grid,
            test_period,
            param_list,
            PoesySettings(
                members=members,
                term=term,
                num_output_steps=num_pred_steps_val_test,
                num_input_steps=num_input_steps,
            ),
        )
        return train_ds, valid_ds, test_ds

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
        """
        array of shape (2, num_lat, num_lon)
        of (X, Y) values
        """
        return self.grid.meshgrid

    @property
    def geopotential_info(self) -> np.array:
        """
        array of shape (num_lat, num_lon)
        with geopotential value for each datapoint
        """
        return self.grid.geopotential

    @property
    def limited_area(self) -> bool:
        """
        Returns True if the dataset is
        compatible with Limited area models
        """
        return True

    @property
    def border_mask(self) -> np.array:
        return self.grid.border_mask

    @property
    def split(self) -> Literal["train", "valid", "test"]:
        return self.period.name

    def __get_params_attr(
        self,
        attribute: str,
        kind: Literal[
            "all", "input", "output", "forcing", "diagnostic", "input_output"
        ] = "all",
    ) -> List[str]:
        out_list = []
        valid_params = (
            ("output", ("output", "input_output")),
            ("forcing", ("input",)),
            ("input", ("input", "input_output")),
            ("diagnostic", ("output",)),
            ("input_output", ("input_output",)),
            ("all", ("input", "input_output", "output")),
        )
        if kind not in [k for k, pk in valid_params]:
            raise NotImplementedError(
                f"{kind} is not known. Possibilites are {[k for k,pk in valid_params]}"
            )
        for param in self.params:
            if any(kind == k and param.kind in pk for k, pk in valid_params):
                out_list += getattr(param, attribute)
        return out_list

    @cached_property
    def units(self) -> Dict[str, int]:
        """
        Return a dictionnary with name and units
        """
        dout = {}
        for param in self.params:
            names = getattr(param, "parameter_short_name")
            units = getattr(param, "units")
            for name, unit in zip(names, units):
                dout[name] = unit
        return dout

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
                names += param.parameter_short_name
        return names

    @cached_property
    def state_weights(self):
        """
        Weights used in the loss function.
        """
        w_dict = {}
        for param in self.params:
            if param.kind in ["output", "input_output"]:
                for name, weight in zip(
                    param.parameter_short_name, param.state_weights
                ):
                    w_dict[name] = weight

        return w_dict

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered.
        Usefull information for plotting.
        """
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )

    @classmethod
    def prepare(cls, path_config: Path):
        print("--> Preparing Poesy Dataset...")

        print("Load train dataset configuration...")
        with open(path_config, "r") as fp:
            conf = json.load(fp)

        print("Computing stats on each parameter...")
        conf["settings"]["standardize"] = True
        train_ds, _, _ = PoesyDataset.from_json(
            fname=path_config,
            num_input_steps=2,
            num_pred_steps_train=1,
            num_pred_steps_val_test=1,
        )
        train_ds.compute_parameters_stats()

        print("Computing time stats on each parameters, between 2 timesteps...")
        conf["settings"]["standardize"] = True
        train_ds, _, _ = PoesyDataset.from_json(
            fname=path_config,
            num_input_steps=2,
            num_pred_steps_train=1,
            num_pred_steps_val_test=1,
        )
        train_ds.compute_time_step_stats()

        return train_ds


class InferPoesyDataset(PoesyDataset):
    """
    Inherite from the PoesyDataset class.
    This class overrides methods sample_list and from_json.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def sample_list(self):
        """
        Create a list of sample from information.
        Outputs terms are computed from the number of prediction steps wanted by the user.
        """
        print("Start forming samples")
        terms = list(
            np.arange(
                self.settings.term["start"],
                self.settings.term["end"],
                self.settings.term["timestep"],
            )
        )

        sample_by_date = len(terms) // self.settings.num_total_steps

        samples = []
        number = 0

        for date in self.period.date_list:
            for member in self.settings.members:
                for sample in range(0, sample_by_date):

                    input_terms = terms[
                        sample
                        * self.settings.num_total_steps : sample
                        * self.settings.num_total_steps
                        + self.settings.num_input_steps
                    ]

                    output_terms = [
                        input_terms[-1] + self.settings.term["timestep"] * (step + 1)
                        for step in range(self.settings.num_inference_pred_steps)
                    ]

                    samp = InferSample(
                        settings=self.settings,
                        date=date,
                        member=member,
                        input_terms=input_terms,
                        output_terms=output_terms,
                    )

                    if samp.is_valid(self.params):

                        samples.append(samp)
                        number += 1
        print("All samples are now defined")

        return samples

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple["InferPoesyDataset", None, None]:
        """
        Return 1 InferPoesyDataset.
        Override configuration file if needed.
        """
        with open(fname, "r") as fp:
            conf = json.load(fp)
            if config_override is not None:
                conf = merge_dicts(conf, config_override)

        grid = Grid(**conf["grid"])
        param_list = []
        for data_source in conf["dataset"]:
            data = conf["dataset"][data_source]
            members = conf["dataset"][data_source].get("members", [0])
            term = conf["dataset"][data_source]["term"]
            for var in data["var"]:
                vard = data["var"][var]
                # Change grid definition
                if "level" in vard:
                    level_type = "hPa"
                else:
                    level_type = "m"
                param = Param(
                    name=var,
                    shortname=vard.pop("shortname", "t2m"),
                    levels=vard.pop("level", [0]),
                    grid=grid,
                    level_type=level_type,
                    fnamer=poesy_forecast_namer,
                    **vard,
                )
                param_list.append(param)
        inference_period = Period(**conf["periods"]["test"], name="test")
        ds = InferPoesyDataset(
            grid,
            inference_period,
            param_list,
            PoesySettings(
                members=members,
                term=term,
                num_input_steps=num_input_steps,
                num_output_steps=0,
                num_inference_pred_steps=conf["num_inference_pred_steps"],
            ),
        )
        return ds, None, None


if __name__ == "__main__":

    path_config = "config/datasets/poesy.json"

    parser = ArgumentParser(description="Prepare Poesy dataset and test loading speed.")
    parser.add_argument(
        "--path_config",
        default=path_config,
        type=Path,
        help="Configuration file for the dataset.",
    )
    parser.add_argument(
        "--n_iter",
        default=10,
        type=int,
        help="Number of samples to test loading speed.",
    )
    args = parser.parse_args()

    # PoesyDataset.prepare(args.path_config)

    print("Dataset info : ")
    train_ds, _, _ = PoesyDataset.from_json(args.path_config, 2, 3, 3)
    train_ds.dataset_info.summary()

    print("Test __get_item__")
    print("Len dataset : ", len(train_ds))

    print("First Item description :")
    data_iter = iter(train_ds.torch_dataloader())
    print(next(data_iter))

    print("Speed test:")
    start_time = time.time()
    for i in tqdm.trange(args.n_iter, desc="Loading samples"):
        _ = next(data_iter)
    delta = time.time() - start_time
    speed = args.n_iter / delta
    print(f"Loading speed: {round(speed, 3)} sample(s)/sec")
