import datetime as dt
import os
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple

import cartopy
import einops
import numpy as np
import torch
from mfai.torch.namedtensor import NamedTensor
from py4cast.settings import CACHE_DIR


@dataclass(slots=True)
class Period:
    # first day of the period (included)
    # each day of the period will be separated from start by an integer multiple of 24h
    # note that the start date valid hour ("t0") may not be 00h00
    start: dt.datetime
    # last day of the period (included)
    end: dt.datetime
    # In hours, step btw the t0 of consecutive terms
    step_duration: int
    name: str
    # first term (= time delta wrt to a date t0) that is admissible
    term_start: int = 0
    # last term (= time delta wrt to a date start) that is admissible
    term_end: int = 23

    def __post_init__(self):
        self.start = np.datetime64(dt.datetime.strptime(str(self.start), "%Y%m%d%H"))
        self.end = np.datetime64(dt.datetime.strptime(str(self.end), "%Y%m%d%H"))

    @property
    def terms_list(self) -> np.array:
        return np.arange(self.term_start, self.term_end + 1, self.step_duration)

    @property
    def date_list(self) -> np.array:
        """
        List all dates available for the period, with a 24h leap
        """
        return np.arange(
            self.start,
            self.end + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
            dtype="datetime64[s]",
        ).tolist()


@dataclass
class Timestamps:
    """
    Describe all timestamps in a sample.
    It contains
        datetime, terms, validity times

    If n_inputs = 2, n_preds = 2, terms will be (-1, 0, 1, 2) * step_duration
     where step_duration is typically an integer multiple of 1 hour

    validity times correspond to the addition of terms to the reference datetime
    """

    # date and hour of the reference time
    datetime: dt.datetime
    # terms are time deltas vis-à-vis the reference input time step.
    terms: np.array

    # validity times are complete datetimes
    validity_times: List[dt.datetime]


GridConfig = namedtuple(
    "GridConfig", "full_size latitude longitude geopotential landsea_mask"
)


@dataclass
class Grid:
    name: str
    load_grid_info_func: Callable[
        [Any], GridConfig
    ]  # function to load grid data (customizable)
    border_size: int = 10

    # subdomain selection: If (0,0,0,0) the whole domain is kept.
    subdomain: Tuple[int] = (0, 0, 0, 0)
    # Note : training won't work with the full domain on some NN because the size
    # can't be divided by 2. Minimal domain : [0,1776,0,2800]
    x: int = field(init=False)  # X dimension of the grid (longitudes)
    y: int = field(init=False)  # Y dimension of the grid (latitudes)
    # projection information (e.g for plotting)
    proj_name: str = "PlateCarree"
    projection_kwargs: dict = field(default_factory={})

    def __post_init__(self):
        self.grid_config = self.get_grid_info()
        # Setting correct subdomain if no subdomain is selected.
        if sum(self.subdomain) == 0:
            self.subdomain = (
                0,
                self.grid_config.full_size[0],
                0,
                self.grid_config.full_size[1],
            )
        self.x = self.subdomain[1] - self.subdomain[0]
        self.y = self.subdomain[3] - self.subdomain[2]
        self.full_size = self.grid_config.full_size

    def get_grid_info(self) -> GridConfig:
        return self.load_grid_info_func(self.name)

    @cached_property
    def lat(self) -> np.array:
        latitudes = self.grid_config.latitude[self.subdomain[0] : self.subdomain[1]]
        return np.transpose(np.tile(latitudes, (self.y, 1)))

    @cached_property
    def lon(self) -> np.array:
        longitudes = self.grid_config.longitude[self.subdomain[2] : self.subdomain[3]]
        return np.tile(longitudes, (self.x, 1))

    @property
    def geopotential(self) -> np.array:
        return self.grid_config.geopotential[
            self.subdomain[0] : self.subdomain[1], self.subdomain[2] : self.subdomain[3]
        ]

    @property
    def landsea_mask(self) -> np.array:
        if self.grid_config.landsea_mask is not None:
            return self.grid_config.landsea_mask[
                self.subdomain[0] : self.subdomain[1],
                self.subdomain[2] : self.subdomain[3],
            ]
        return np.zeros((self.x, self.y))  # TODO : add real mask

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
        grid_limits = [  # In projection (llon, ulon, llat, ulat)
            float(self.grid_config.longitude[self.subdomain[2]]),  # min y
            float(self.grid_config.longitude[self.subdomain[3] - 1]),  # max y
            float(self.grid_config.latitude[self.subdomain[1] - 1]),  # max x
            float(self.grid_config.latitude[self.subdomain[0]]),  # min x
        ]
        return grid_limits

    @cached_property
    def meshgrid(self) -> np.array:
        """Build a meshgrid from coordinates position."""
        latitudes = self.grid_config.latitude[self.subdomain[0] : self.subdomain[1]]
        longitudes = self.grid_config.longitude[self.subdomain[2] : self.subdomain[3]]
        meshgrid = np.array(np.meshgrid(longitudes, latitudes))
        return meshgrid  # shape (2, x, y)

    @cached_property
    def projection(self):
        func = getattr(cartopy.crs, self.proj_name)
        return func(**self.projection_kwargs)


def grid_static_features(grid: Grid, extra_statics: List[NamedTensor]):
    """
    Grid static features
    """
    # -- Static grid node features --
    xy = grid.meshgrid  # (2, N_x, N_y)
    grid_xy = torch.tensor(xy)
    # Need to rearange
    pos_max = torch.max(torch.max(grid_xy, dim=1).values, dim=1).values
    pos_min = torch.min(torch.min(grid_xy, dim=1).values, dim=1).values
    grid_xy = (einops.rearrange(grid_xy, ("n x y -> x y n")) - pos_min) / (
        pos_max - pos_min
    )  # Rearange and divide  by maximum coordinate

    # (Nx, Ny, 1)
    geopotential = torch.tensor(grid.geopotential).unsqueeze(2)  # (N_x, N_y, 1)
    gp_min = torch.min(geopotential)
    gp_max = torch.max(geopotential)
    # Rescale geopotential to [0,1]
    if gp_max != gp_min:
        geopotential = (geopotential - gp_min) / (gp_max - gp_min)  # (N_x,N_y, 1)
    else:
        warnings.warn("Geopotential is constant. Set it to 1")
        geopotential = geopotential / gp_max

    grid_border_mask = torch.tensor(grid.border_mask).unsqueeze(2)  # (N_x, N_y,1)

    feature_names = []
    for x in extra_statics:
        feature_names += x.feature_names
    state_var = NamedTensor(
        tensor=torch.cat(
            [grid_xy, geopotential, grid_border_mask]
            + [x.tensor for x in extra_statics],
            dim=-1,
        ),
        feature_names=["x", "y", "geopotential", "border_mask"]
        + feature_names,  # Noms des champs 2D
        names=["lat", "lon", "features"],
    )
    state_var.type_(torch.float32)
    return state_var


ParamConfig = namedtuple(
    "ParamConfig", "unit level_type long_name grid grib_name grib_param"
)


@dataclass(slots=True)
class WeatherParam:
    name: str
    level: int
    grid: Grid
    load_param_info: Callable[[str], ParamConfig]
    # Parameter status :
    # input = forcings, output = diagnostic, input_output = classical weather var
    kind: Literal["input", "output", "input_output"]
    get_weight_per_level: Callable[[int, str], [float]]
    level_type: str = field(init=False)
    long_name: str = field(init=False)
    unit: str = field(init=False)
    native_grid: str = field(init=False)
    grib_name: str = field(init=False)
    grib_param: str = field(init=False)

    def __post_init__(self):
        param_info = self.load_param_info(self.name)
        self.unit = param_info.unit
        if param_info.level_type in ["heightAboveGround", "meanSea", "surface"]:
            self.level_type = param_info.level_type
        else:
            self.level_type = "isobaricInhPa"
        self.long_name = param_info.long_name
        self.native_grid = param_info.grid
        self.grib_name = param_info.grib_name
        self.grib_param = param_info.grib_param

    @property
    def state_weight(self) -> float:
        """Weight to confer to the param in the loss function"""
        return self.get_weight_per_level(self.level, self.level_type)

    @property
    def parameter_name(self) -> str:
        return f"{self.long_name}_{self.level}_{self.level_type}"


@dataclass
class Stats:
    fname: Path

    def __post_init__(self):
        self.stats = torch.load(self.fname, "cpu", weights_only=True)

    def items(self):
        return self.stats.items()

    def __getitem__(self, shortname: str):
        return self.stats[shortname]

    def to_list(
        self,
        stat_name: Literal["mean", "std", "min", "max"],
        shortnames: List[str],
        dtype: torch.dtype = torch.float32,
    ) -> list:
        """
        Get a tensor with the stats inside.
        The order is the one of the shortnames.

        Args:
            stat_name : Statistics wanted
            names (List[str]): Field for which we want stats

        Returns:
            _type_: _description_
        """
        if len(shortnames) > 0:
            return torch.stack(
                [self[name][stat_name] for name in shortnames], dim=0
            ).type(dtype)
        else:
            return []


@dataclass(slots=True)
class SamplePreprocSettings:
    dataset_name: str
    num_input_steps: int  # Number of input timesteps
    num_pred_steps: int  # Number of output timesteps
    step_duration: float  # duration in hour
    standardize: bool = True
    file_format: Literal["npy", "grib"] = "grib"
    members: Optional[Tuple[int]] = None
    add_landsea_mask: bool = False


class DataAccessor(ABC):
    """
    Abstract base class used as interface contract for user-defined data sources.
    If you intend to use a new data source with py4cast, define your own DataAccessor (with customized methods)
    that allows to read and access your data in the way you have organized it.
    When all abstract methods have a concrete implementation with the correct signature, you are ready to go.
    `py4cast` will then handles sample definition, choice of variables, dataloading logic .
    We refer to py4cast.datasets.titan (reanalysis) and py4cast.datasets.poesy (reforecast)
    as two end-to-end examples of DataAccessors.
    """

    def cache_dir(name: str, grid: Grid) -> Path:
        """
        Return the path of cache_dir, where, e.g, stat files can be stored
        """
        path = CACHE_DIR / f"{name}_{grid.name}"
        os.makedirs(path, mode=0o777, exist_ok=True)
        return path

    @abstractmethod
    def get_dataset_path(name: str, grid: Grid) -> Path:
        """
        Return the path that will be used as cache for data during dataset preparation.
        """

    @abstractmethod
    def get_weight_per_level(
        level: int,
        level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
    ) -> float:
        """
        Attribute a weight in the final reconstruction loss depending on the height level and level_type
        """

    @abstractmethod
    def load_grid_info(name: str) -> GridConfig:
        """
        Fetch grid related information on disk (with grid name being given),
        Return an instance of the GridConfig namedtuple.
        Consumed by the 'Grid' interface object.
        """

    @abstractmethod
    def get_grid_coords(param: WeatherParam) -> List[float]:
        """
        Get the extent of the grid related to a given parameter
        (min and max for latitude (2 first positions) and longitude (positions 3 and 4))
        """

    @abstractmethod
    def load_param_info(name: str) -> ParamConfig:
        """
        Fetch the information related to a given parameter related to `name`
        Return a configuration to instantiate parameter data as ParamConfig.
        Consumed by the `Param` interface object.
        """

    @classmethod
    @abstractmethod
    def get_filepath(
        cls,
        dataset_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: Literal["npy", "grib"],
    ) -> Path:
        """
        Return a path to retrieve a given parameter at from a dataset
        """

    @classmethod
    @abstractmethod
    def load_data_from_disk(
        cls,
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

    @classmethod
    @abstractmethod
    def exists(
        cls,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: Literal["npy", "grib"] = "grib",
    ) -> bool:
        """
        Verfication functions to check whether actual data exists,
        corresponding to the given timestamp and WeatherParam.
        Concrete implementations can typically verify that the file where the data is exists.
        """

    @abstractmethod
    def valid_timestamp(n_inputs: int, timestamps: Timestamps) -> bool:
        """
        Verification function called after the creation of each timestamps.
        Check if computed terms respect the dataset convention.
        """

    @abstractmethod
    def parameter_namer(param: WeatherParam) -> str:
        """
        Return a string used to identify parameters names on files and stats metadata
        """
        return f"{param.name}_{param.level}_{param.level_type}"
