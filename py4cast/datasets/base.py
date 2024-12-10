"""
Base classes defining our software components
and their interfaces
"""

import datetime as dt
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections import namedtuple
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import cartopy
import einops
import gif
import matplotlib.pyplot as plt
import numpy as np
import torch
from mfai.torch.namedtensor import NamedTensor
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_tensor_fn
from tqdm import tqdm

from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing
from py4cast.plots import DomainInfo
from py4cast.utils import RegisterFieldsMixin, torch_save


@dataclass(slots=True)
class Item:
    """
    Dataclass holding one Item.
    inputs has shape (timestep, lat, lon, features)
    outputs has shape (timestep, lat, lon, features)
    forcing has shape (timestep, lat, lon, features)
    """

    inputs: NamedTensor
    forcing: NamedTensor
    outputs: NamedTensor

    def unsqueeze_(self, dim_name: str, dim_index: int):
        """
        Insert a new dimension dim_name at dim_index of size 1
        """
        self.inputs.unsqueeze_(dim_name, dim_index)
        self.outputs.unsqueeze_(dim_name, dim_index)
        self.forcing.unsqueeze_(dim_name, dim_index)

    def squeeze_(self, dim_name: Union[List[str], str]):
        """
        Squeeze the underlying tensor along the dimension(s)
        given its/their name(s).
        """
        self.inputs.squeeze_(dim_name)
        self.outputs.squeeze_(dim_name)
        self.forcing.squeeze_(dim_name)

    def to_(self, *args, **kwargs):
        """
        'In place' operation to call torch's 'to' method on the underlying NamedTensors.
        """
        self.inputs.to_(*args, **kwargs)
        self.outputs.to_(*args, **kwargs)
        self.forcing.to_(*args, **kwargs)

    def pin_memory(self):
        """
        Custom Item must implement this method to pin the underlying tensors to memory.
        See https://pytorch.org/docs/stable/data.html#memory-pinning
        """
        self.inputs.pin_memory_()
        self.forcing.pin_memory_()
        self.outputs.pin_memory_()
        return self

    def __post_init__(self):
        """
        Checks that the dimensions of the inputs, outputs are consistent.
        This is necessary for our auto-regressive training.
        """
        if self.inputs.names != self.outputs.names:
            raise ValueError(
                f"Inputs and outputs must have the same dim names, got {self.inputs.names} and {self.outputs.names}"
            )

        # Also check feature names
        if self.inputs.feature_names != self.outputs.feature_names:
            raise ValueError(
                f"Inputs and outputs must have the same feature names, "
                f"got {self.inputs.feature_names} and {self.outputs.feature_names}"
            )

    def __str__(self) -> str:
        """
        Utility method to explore a batch/item shapes and names.
        """
        table = []
        for attr in (f.name for f in fields(self)):
            nt: NamedTensor = getattr(self, attr)
            if nt is not None:
                for feature_name in nt.feature_names:
                    tensor = nt[feature_name]
                    table.append(
                        [
                            attr,
                            nt.names,
                            list(nt[feature_name].shape),
                            feature_name,
                            tensor.min(),
                            tensor.max(),
                        ]
                    )
        headers = [
            "Type",
            "Dimension Names",
            "Torch Shape",
            "feature name",
            "Min",
            "Max",
        ]
        return str(tabulate(table, headers=headers, tablefmt="simple_outline"))


@dataclass
class ItemBatch(Item):
    """
    Dataclass holding a batch of items.
    input has shape (batch, timestep, lat, lon, features)
    output has shape (batch, timestep, lat, lon, features)
    forcing has shape (batch, timestep, lat, lon, features)
    """

    @cached_property
    def batch_size(self):
        return self.inputs.dim_size("batch")

    @cached_property
    def num_input_steps(self):
        return self.inputs.dim_size("timestep")

    @cached_property
    def num_pred_steps(self):
        return self.outputs.dim_size("timestep")


def collate_fn(items: List[Item]) -> ItemBatch:
    """
    Collate a list of item. Add one dimension at index zero to each NamedTensor.
    Necessary to form a batch from a list of items.
    See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
    """
    # Here we postpone that for each batch the same dimension should be present.
    batch_of_items = {}

    # Iterate over inputs, outputs and forcing fields
    for field_name in (f.name for f in fields(Item)):
        batched_tensor = collate_tensor_fn(
            [getattr(item, field_name).tensor for item in items]
        ).type(torch.float32)

        batch_of_items[field_name] = NamedTensor.expand_to_batch_like(
            batched_tensor, getattr(items[0], field_name)
        )

    return ItemBatch(**batch_of_items)


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


@dataclass
class Statics(RegisterFieldsMixin):
    """
    Static fields of the dataset.
    Tensor can be registered as buffer in a lightning module
    using the register_buffers method.
    """

    # border_mask: torch.Tensor
    grid_static_features: NamedTensor
    grid_shape: Tuple[int, int]
    border_mask: torch.Tensor = field(init=False)
    interior_mask: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.border_mask = self.grid_static_features["border_mask"]
        self.interior_mask = 1.0 - self.border_mask

    @cached_property
    def meshgrid(self) -> torch.Tensor:
        """
        Return a tensor concatening X,Y
        """
        return einops.rearrange(
            torch.cat(
                [
                    self.grid_static_features["x"],
                    self.grid_static_features["y"],
                ],
                dim=-1,
            ),
            ("x y n -> n x y"),
        )


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
        aggregate: Literal["mean", "std", "min", "max"],
        shortnames: List[str],
        dtype: torch.dtype = torch.float32,
    ) -> list:
        """
        Get a tensor with the stats inside.
        The order is the one of the shortnames.

        Args:
            aggregate : Statistics wanted
            names (List[str]): Field for which we want stats

        Returns:
            _type_: _description_
        """
        if len(shortnames) > 0:
            return torch.stack(
                [self[name][aggregate] for name in shortnames], dim=0
            ).type(dtype)
        else:
            return []


@dataclass(slots=True)
class DatasetInfo:
    """
    This dataclass holds all the informations
    about the dataset that other classes
    and functions need to interact with it.
    """

    name: str  # Name of the dataset
    domain_info: DomainInfo  # Information used for plotting
    units: Dict[str, str]  # d[shortname] = unit (str)
    weather_dim: int
    forcing_dim: int
    step_duration: (
        float  # Duration (in hour) of one step in the dataset. 0.25 means 15 minutes.
    )
    statics: Statics  # A lot of static variables
    stats: Stats
    diff_stats: Stats
    state_weights: Dict[str, float]
    shortnames: Dict[str, List[str]] = None

    def summary(self):
        """
        Print a table summarizing variables present in the dataset (and their role)
        """
        print(f"\n Summarizing {self.name} \n")
        print(f"Step_duration {self.step_duration}")
        print(f"Static fields {self.statics.grid_static_features.feature_names}")
        print(f"Grid static features {self.statics.grid_static_features}")
        print(f"Features shortnames {self.shortnames}")
        for p in ["input", "input_output", "output"]:
            names = self.shortnames[p]
            print(names)
            mean = self.stats.to_list("mean", names)
            std = self.stats.to_list("std", names)
            mini = self.stats.to_list("min", names)
            maxi = self.stats.to_list("max", names)
            units = [self.units[name] for name in names]
            if p != "input":
                diff_mean = self.diff_stats.to_list("mean", names)
                diff_std = self.diff_stats.to_list("std", names)
                weight = [self.state_weights[name] for name in names]

                data = list(
                    zip(
                        names, units, mean, std, mini, maxi, diff_mean, diff_std, weight
                    )
                )
                table = tabulate(
                    data,
                    headers=[
                        "Name",
                        "Unit",
                        "Mean",
                        "Std",
                        "Minimum",
                        "Maximum",
                        "DiffMean",
                        "DiffStd",
                        "Weight in Loss",
                    ],
                    tablefmt="simple_outline",
                )
            else:
                data = list(zip(names, units, mean, std, mini, maxi))
                table = tabulate(
                    data,
                    headers=["Name", "Unit", "Mean", "Std", "Minimun", "Maximum"],
                    tablefmt="simple_outline",
                )
            if data:
                print(p.upper())  # Print the kind of variable
                print(table)  # Print the table


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


# This namedtuple contains attributes that are used by the Grid class
# These attributes are retrieved from disk in any user-defined manner.
# It is there to define the expected type of the retrieval function.
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

    # TODO : from the grib, save a npy lat lon h mask for each grid
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


def generate_forcings(
    date: dt.datetime, output_terms: np.array, grid: Grid
) -> List[NamedTensor]:
    """
    Generate all the forcing in this function.
    Return a list of NamedTensor.
    """
    lforcings = []
    float_terms = (output_terms / dt.timedelta(hours=1)).astype(float)
    time_forcing = NamedTensor(  # doy : day_of_year
        feature_names=["cos_hour", "sin_hour", "cos_doy", "sin_doy"],
        tensor=get_year_hour_forcing(date, float_terms).type(torch.float32),
        names=["timestep", "features"],
    )
    solar_forcing = NamedTensor(
        feature_names=["toa_radiation"],
        tensor=generate_toa_radiation_forcing(
            grid.lat, grid.lon, date, float_terms
        ).type(torch.float32),
        names=["timestep", "lat", "lon", "features"],
    )
    lforcings.append(time_forcing)
    lforcings.append(solar_forcing)

    return lforcings


@dataclass(slots=True)
class SamplePreprocSettings:
    """
    Main settings defining the timesteps of a data sample (regardless of parameters)
    and additional preprocessing information
    that will be used during training/inference.
    Values can be modified by defining a `settings` field in the configuration json file.

    """

    dataset_name: str
    num_input_steps: int  # Number of input timesteps
    num_pred_steps: int  # Number of output timesteps
    standardize: bool = True
    file_format: Literal["npy", "grib"] = "grib"
    members: Tuple[int] = (0,)


# This namedtuple contains attributes that are used by the WeatherParam class
# These attributes are retrieved from disk in any user-defined manner.
# It is there to define the expected type of the retrieval function.
ParamConfig = namedtuple(
    "ParamConfig", "unit level_type long_name grid grib_name grib_param"
)


@dataclass(slots=True)
class WeatherParam:
    """
    This class represent a single weather parameter (seen as a 2D field)
    with all attributes used to retrieve and manipulate the parameter;
    Used in the construction of the Dataset object.
    """

    name: str
    level: int
    grid: Grid
    load_param_info: Callable[[str], ParamConfig]
    # Parameter status :
    # input = forcings, output = diagnostic, input_output = classical weather var
    kind: Literal["input", "output", "input_output"]
    # function to retrieve the weight given to the parameter in the loss, depending on the level
    get_weight_per_level: Callable[[int, str], float]
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

    @property
    def parameter_short_name(self) -> str:
        return f"{self.name}_{self.level}_{self.level_type}"


def get_param_list(
    conf: dict,
    grid: Grid,
    # function to retrieve all parameters information about the dataset
    load_param_info: Callable[[str], ParamConfig],
    # function to retrieve the weight given to the parameter in the loss
    get_weight_per_level: Callable[[str], float],
) -> List[WeatherParam]:
    param_list = []
    for name, values in conf["params"].items():
        for lvl in values["levels"]:
            param = WeatherParam(
                name=name,
                level=lvl,
                grid=grid,
                load_param_info=load_param_info,
                kind=values["kind"],
                get_weight_per_level=get_weight_per_level,
            )
            param_list.append(param)
    return param_list


#############################################################
#                            SAMPLE                         #
#############################################################


@dataclass(slots=True)
class Sample:
    """Describes a sample"""

    timestamps: Timestamps
    settings: SamplePreprocSettings
    params: List[WeatherParam]
    stats: Stats
    grid: Grid
    exists: Callable[[Any], bool]
    get_param_tensor: Callable[[Any], torch.tensor]
    member: int = 0

    input_timestamps: Timestamps = field(default=None)
    output_timestamps: Timestamps = field(default=None)

    def __post_init__(self):
        """Setups time variables to be able to define a sample.
        For example for n_inputs = 2, n_preds = 3, step_duration = 3h:
        all_steps = [-1, 0, 1, 2, 3]
        all_timesteps = [-3h, 0h, 3h, 6h, 9h]
        pred_timesteps = [3h, 6h, 9h]
        all_dates = [24/10/22 21:00,  24/10/23 00:00, 24/10/23 03:00, 24/10/23 06:00, 24/10/23 09:00]
        """

        if self.settings.num_input_steps + self.settings.num_pred_steps != len(
            self.timestamps.validity_times
        ):
            raise Exception("Length terms does not match inputs + outputs")

        self.input_timestamps = Timestamps(
            self.timestamps.datetime,
            self.timestamps.terms[: self.settings.num_input_steps],
            self.timestamps.validity_times[: self.settings.num_input_steps],
        )
        self.output_timestamps = Timestamps(
            self.timestamps.datetime,
            self.timestamps.terms[self.settings.num_input_steps :],
            self.timestamps.validity_times[self.settings.num_input_steps :],
        )

    def __repr__(self):
        return f"Date {self.timestamps.datetime}, input terms {self.input_terms}, output terms {self.output_terms}"

    def is_valid(self) -> bool:
        for param in self.params:
            if not self.exists(
                self.settings.dataset_name,
                param,
                self.timestamps,
                self.settings.file_format,
            ):
                return False
        return True

    def load(self, no_standardize: bool = False) -> Item:
        """
        Return inputs, outputs, forcings as tensors concatenated into a Item.
        """
        linputs, loutputs = [], []

        # Reading parameters from files
        for param in self.params:
            state_kwargs = {
                "feature_names": [param.parameter_short_name],
                "names": ["timestep", "lat", "lon", "features"],
            }
            if param.kind == "input":
                # forcing is taken for every predicted step
                tensor = self.get_param_tensor(
                    param=param,
                    stats=self.stats,
                    timestamps=self.input_timestamps,
                    settings=self.settings,
                    standardize=(self.settings.standardize and not no_standardize),
                    member=self.member,
                )
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))

            elif param.kind == "output":
                tensor = self.get_param_tensor(
                    param=param,
                    stats=self.stats,
                    timestamps=self.output_timestamps,
                    settings=self.settings,
                    standardize=(self.settings.standardize and not no_standardize),
                    member=self.member,
                )
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))
                loutputs.append(tmp_state)

            else:  # input_output
                tensor = self.get_param_tensor(
                    param=param,
                    stats=self.stats,
                    timestamps=self.timestamps,
                    settings=self.settings,
                    standardize=(self.settings.standardize and not no_standardize),
                    member=self.member,
                )
                state_kwargs["names"][0] = "timestep"
                tmp_state = NamedTensor(
                    tensor=tensor[-self.settings.num_pred_steps :],
                    **deepcopy(state_kwargs),
                )

                loutputs.append(tmp_state)
                tmp_state = NamedTensor(
                    tensor=tensor[: self.settings.num_input_steps],
                    **deepcopy(state_kwargs),
                )
                linputs.append(tmp_state)

        lforcings = generate_forcings(
            date=self.timestamps.datetime,
            output_terms=self.output_timestamps.terms,
            grid=self.grid,
        )

        for forcing in lforcings:
            forcing.unsqueeze_and_expand_from_(linputs[0])

        return Item(
            inputs=NamedTensor.concat(linputs),
            outputs=NamedTensor.concat(loutputs),
            forcing=NamedTensor.concat(lforcings),
        )

    def plot(self, item: Item, step: int, save_path: Path = None) -> None:
        # Retrieve the named tensor
        ntensor = item.inputs if step <= 0 else item.outputs

        # Retrieve the timestep data index
        if step <= 0:  # input step
            index_tensor = step + self.settings.num_input_steps - 1
        else:  # output step
            index_tensor = step - 1

        # Sort parameters by level, to plot each level on one line
        levels = sorted(list(set([p.level for p in self.params])))
        dict_params = {level: [] for level in levels}
        for param in self.params:
            if param.parameter_short_name in ntensor.feature_names:
                dict_params[param.level].append(param)

        # Groups levels 0m, 2m and 10m on one "surf" level
        dict_params["surf"] = []
        for lvl in [0, 2, 10]:
            if lvl in levels:
                dict_params["surf"] += dict_params.pop(lvl)

        # Plot settings
        kwargs = {"projection": self.grid.projection}
        nrows = len(dict_params.keys())
        ncols = max([len(param_list) for param_list in dict_params.values()])
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 15), subplot_kw=kwargs)

        for i, level in enumerate(dict_params.keys()):
            for j, param in enumerate(dict_params[level]):
                pname = param.parameter_short_name
                tensor = ntensor[pname][index_tensor, :, :, 0]
                arr = tensor.numpy()[::-1]  # invert latitude
                vmin, vmax = self.stats[pname]["min"], self.stats[pname]["max"]
                img = axs[i, j].imshow(
                    arr, vmin=vmin, vmax=vmax, extent=self.grid.grid_limits
                )
                axs[i, j].set_title(pname)
                axs[i, j].coastlines(resolution="50m")
                cbar = fig.colorbar(img, ax=axs[i, j], fraction=0.04, pad=0.04)
                cbar.set_label(param.unit)

        plt.suptitle(
            f"Run: {self.timestamps.datetime} - Valid time: {self.timestamps.validity_times[step]}"
        )
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

    @gif.frame
    def plot_frame(self, item: Item, step: int) -> None:
        self.plot(item, step)

    def plot_gif(self, save_path: Path):
        # We don't want to standardize data for plots
        item = self.load(no_standardize=True)
        frames = []
        n_inputs, n_preds = self.settings.num_input_steps, self.settings.num_pred_steps
        steps = list(range(-n_inputs + 1, n_preds + 1))
        for step in tqdm.tqdm(steps, desc="Making gif"):
            frame = self.plot_frame(item, step)
            frames.append(frame)
        gif.save(frames, str(save_path), duration=250)


@dataclass(slots=True)
class TorchDataloaderSettings:
    """
    Settings for the torch dataloader
    """

    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = False
    prefetch_factor: Union[int, None] = None
    persistent_workers: bool = False


class DatasetABC(ABC):
    """
    Abstract Base class defining the mandatory
    properties and methods a dataset MUST
    implement.
    """

    @abstractmethod
    def torch_dataloader(self, tl_settings: TorchDataloaderSettings) -> DataLoader:
        """
        Builds a torch dataloader from self.
        """

    @abstractproperty
    def dataset_info(self) -> DatasetInfo:
        """
        Return the object DatasetInfo
        """

    @abstractproperty
    def meshgrid(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (lat, lon) values
        """

    @cached_property
    def grid_shape(self) -> tuple:
        x, _ = self.meshgrid
        return x.shape

    @abstractproperty
    def geopotential_info(self) -> np.array:
        """
        array of shape (num_lat, num_lon)
        with geopotential value for each datapoint
        """

    @abstractproperty
    def cache_dir(self) -> Path:
        """
        Cache directory of the dataset.
        Used at least to get statistics.
        """
        pass

    def compute_mean_std_min_max(
        self, type_tensor: Literal["inputs", "outputs", "forcing"]
    ):
        """
        Compute mean and standard deviation for this dataset.
        """
        random_batch = next(iter(self.torch_dataloader()))
        named_tensor = getattr(random_batch, type_tensor)
        n_features = len(named_tensor.feature_names)
        sum_means = torch.zeros(n_features)
        sum_squares = torch.zeros(n_features)
        ndim_features = len(named_tensor.tensor.shape) - 1
        flat_input = named_tensor.tensor.flatten(0, ndim_features - 1)  # (X, Features)
        best_min = torch.min(flat_input, dim=0).values
        best_max = torch.max(flat_input, dim=0).values
        counter = 0
        if self.settings.standardize:
            raise ValueError("Your dataset should not be standardized.")

        for batch in tqdm(
            self.torch_dataloader(), desc=f"Computing {type_tensor} stats"
        ):
            tensor = getattr(batch, type_tensor).tensor
            tensor = tensor.flatten(1, 3)  # Flatten to be (Batch, X, Features)
            counter += tensor.shape[0]  # += batch size

            sum_means += torch.sum(tensor.mean(dim=1), dim=0)  # (d_features)
            sum_squares += torch.sum((tensor**2).mean(dim=1), dim=0)  # (d_features)

            mini = torch.min(tensor, 1).values[0]
            stack_mini = torch.stack([best_min, mini], dim=0)
            best_min = torch.min(stack_mini, dim=0).values  # (d_features)

            maxi = torch.max(tensor, 1).values[0]
            stack_maxi = torch.stack([best_max, maxi], dim=0)
            best_max = torch.max(stack_maxi, dim=0).values  # (d_features)

        mean = sum_means / counter
        second_moment = sum_squares / counter
        std = torch.sqrt(second_moment - mean**2)  # (d_features)

        stats = {}
        for i, name in enumerate(named_tensor.feature_names):
            stats[name] = {
                "mean": mean[i],
                "std": std[i],
                "min": best_min[i],
                "max": best_max[i],
            }
        return stats

    def compute_parameters_stats(self):
        """
        Compute mean and standard deviation for this dataset.
        """
        all_stats = {}
        for type_tensor in ["inputs", "outputs", "forcing"]:
            stats_dict = self.compute_mean_std_min_max(type_tensor)
            for feature, stats in stats_dict.items():
                # If feature was computed multiple times we keep only first occurence
                if feature not in all_stats.keys():
                    all_stats[feature] = stats

        dest_file = self.cache_dir / "parameters_stats.pt"
        torch_save(all_stats, dest_file)
        print(f"Parameters statistics saved in {dest_file}")

    def compute_time_step_stats(self):
        random_inputs = next(iter(self.torch_dataloader())).inputs
        n_features = len(random_inputs.feature_names)
        sum_means = torch.zeros(n_features)
        sum_squares = torch.zeros(n_features)
        counter = 0
        if not self.settings.standardize:
            raise ValueError("Your dataset should be standardized.")

        for batch in tqdm(self.torch_dataloader()):
            # Here we assume that data are in 2 or 3 D
            inputs = batch.inputs.tensor
            outputs = batch.outputs.tensor

            in_out = torch.cat([inputs, outputs], dim=1)
            diff = (
                in_out[:, 1:] - in_out[:, :-1]
            )  # Substract information on time dimension
            diff = diff.flatten(1, 3)  # Flatten everybody to be (Batch, X, Features)

            counter += in_out.shape[0]  # += batch size
            sum_means += torch.sum(diff.mean(dim=1), dim=0)  # (d_features)
            sum_squares += torch.sum((diff**2).mean(dim=1), dim=0)  # (d_features)

        diff_mean = sum_means / counter
        diff_second_moment = sum_squares / counter
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)
        store_d = {}

        # Storing variable statistics
        for i, name in enumerate(batch.inputs.feature_names):
            store_d[name] = {
                "mean": diff_mean[i],
                "std": diff_std[i],
            }
        # Diff mean and std of forcing variables are not used during training so we
        # store fixed values : mean = 0, std = 1
        for name in batch.forcing.feature_names:
            store_d[name] = {"mean": torch.tensor(0), "std": torch.tensor(1)}
        dest_file = self.cache_dir / "diff_stats.pt"
        torch_save(store_d, self.cache_dir / "diff_stats.pt")
        print(f"Parameters time diff stats saved in {dest_file}")

    @property
    def dataset_extra_statics(self) -> List[NamedTensor]:
        """
        Datasets can override this method to add
        more static data.
        """
        return []

    @cached_property
    def grid_static_features(self):
        """
        Grid static features
        """
        # -- Static grid node features --
        xy = self.meshgrid  # (2, N_x, N_y)
        grid_xy = torch.tensor(xy)
        # Need to rearange
        pos_max = torch.max(torch.max(grid_xy, dim=1).values, dim=1).values
        pos_min = torch.min(torch.min(grid_xy, dim=1).values, dim=1).values
        grid_xy = (einops.rearrange(grid_xy, ("n x y -> x y n")) - pos_min) / (
            pos_max - pos_min
        )  # Rearange and divide  by maximum coordinate

        # (Nx, Ny, 1)
        geopotential = torch.tensor(self.geopotential_info).unsqueeze(
            2
        )  # (N_x, N_y, 1)
        gp_min = torch.min(geopotential)
        gp_max = torch.max(geopotential)
        # Rescale geopotential to [0,1]
        if gp_max != gp_min:
            geopotential = (geopotential - gp_min) / (gp_max - gp_min)  # (N_x,N_y, 1)
        else:
            warnings.warn("Geopotential is constant. Set it to 1")
            geopotential = geopotential / gp_max

        grid_border_mask = torch.tensor(self.border_mask).unsqueeze(2)  # (N_x, N_y,1)

        feature_names = []
        for x in self.dataset_extra_statics:
            feature_names += x.feature_names
        state_var = NamedTensor(
            tensor=torch.cat(
                [grid_xy, geopotential, grid_border_mask]
                + [x.tensor for x in self.dataset_extra_statics],
                dim=-1,
            ),
            feature_names=["x", "y", "geopotential", "border_mask"]
            + feature_names,  # Noms des champs 2D
            names=["lat", "lon", "features"],
        )
        state_var.type_(torch.float32)
        return state_var

    @cached_property
    def statics(self) -> Statics:
        return Statics(
            **{
                "grid_static_features": self.grid_static_features,
                "grid_shape": self.grid_shape,
            }
        )

    @cached_property
    def stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "parameters_stats.pt")

    @cached_property
    def diff_stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "diff_stats.pt")

    @abstractclassmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple["DatasetABC", "DatasetABC", "DatasetABC"]:
        """
        Load a dataset from a json file + the number of expected timesteps
        taken as inputs (num_input_steps) and to predict (num_pred_steps)
        Return the train, valid and test datasets, in that order
        config_override is a dictionary that can be used to override
        some keys of the config file.
        """
        pass
