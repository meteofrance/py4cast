import datetime as dt
import json
import os
from collections.abc import Callable
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import cartopy
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from py4cast.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Item,
    NamedTensor,
    TorchDataloaderSettings,
    collate_fn,
)
from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing

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
    scratch_path : str
    # Subgrid selection. If (0,0,0,0) the whole grid is kept.
    subgrid: Tuple[int] = (0, 0, 0, 0)
    x: int = field(init=False)  # X dimension
    y: int = field(init=False)  # Y dimension

    def __post_init__(self):
        ds = self.get_grid_file()
        x, y = ds.shape
        # Setting correct subgrid if no subgrid is selected.
        if self.subgrid == (0, 0, 0, 0):
            self.subgrid = (0, x, 0, y)
        self.x = self.subgrid[1] - self.subgrid[0]
        self.y = self.subgrid[3] - self.subgrid[2]

    @abstractmethod
    def get_grid_file(self):
        return None

    @property
    @abstractmethod
    def grid_filepath(self):
        return Path(self.scratch_path) / OROGRAPHY_FNAME

    @cached_property
    def lat(self) -> np.array:
       return np.empty()

    @cached_property
    def lon(self) -> np.array:
        return np.empty()

    @cached_property
    def meshgrid(self) -> np.array:
        """
        Build a meshgrid from coordinates position.
        """
        return np.asarray([self.lat, self.lon])

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
        data_array = np.load(self.filename(date=date), mmap_mode="r")

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
class Settings:
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
    settings: Settings
    date: dt.datetime
    input_terms: Tuple[float]
    output_terms: Tuple[float]

    # Term wrt to the date {date}. Gives validity
    terms: Tuple[float] = field(init=False)

    def __post_init__(self):
        self.terms = self.input_terms + self.output_terms

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

