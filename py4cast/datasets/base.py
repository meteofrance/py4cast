"""
Base classes defining our software components
and their interfaces
"""

import datetime as dt
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple, Union

import cartopy
import einops
import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import collate_tensor_fn
from tqdm import tqdm

if TYPE_CHECKING:
    from py4cast.plots import DomainInfo

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    Param,
    ParamConfig,
    Settings,
)
from py4cast.utils import RegisterFieldsMixin


@dataclass(slots=True)
class TensorWrapper:
    """
    Wrapper around a torch tensor.
    We do this separated dataclass to allow lightning's introspection to see our batch size
    and move our tensors to the right device, otherwise we have this error/warning:
    "Trying to infer the `batch_size` from an ambiguous collection ..."
    """

    tensor: torch.Tensor


class NamedTensor(TensorWrapper):
    """
    NamedTensor is a wrapper around a torch tensor
    adding several attributes :

    * a 'names' attribute with the names of the
    tensor's dimensions (like https://pytorch.org/docs/stable/named_tensor.html).

    Torch's named tensors are still experimental and subject to change.

    * a 'feature_names' attribute containing the names of the features
    along the last dimension of the tensor.

    NamedTensor can be concatenated along the last dimension
    using the | operator.
    nt3 = nt1 | nt2
    """

    SPATIAL_DIM_NAMES = ("lat", "lon", "ngrid")

    def __init__(
        self,
        tensor: torch.Tensor,
        names: List[str],
        feature_names: List[str],
        feature_dim_name: str = "features",
    ):
        if len(tensor.shape) != len(names):
            raise ValueError(
                f"Number of names ({len(names)}) must match number of dimensions ({len(tensor.shape)})"
            )
        if tensor.shape[names.index(feature_dim_name)] != len(feature_names):
            raise ValueError(
                f"Number of feature names ({len(feature_names)}:{feature_names}) must match "
                f"number of features ({tensor.shape[names.index(feature_dim_name)]}) in the supplied tensor"
            )

        super().__init__(tensor)

        self.names = names
        # build lookup table for fast indexing
        self.feature_names_to_idx = {
            feature_name: idx for idx, feature_name in enumerate(feature_names)
        }
        self.feature_names = feature_names
        self.feature_dim_name = feature_dim_name

    @property
    def ndims(self):
        """
        Number of dimensions of the tensor.
        """
        return len(self.names)

    @property
    def num_spatial_dims(self):
        """
        Number of spatial dimensions of the tensor.
        """
        return len([x for x in self.names if x in self.SPATIAL_DIM_NAMES])

    def __str__(self):
        head = "--- NamedTensor ---\n"
        head += f"Names: {self.names}\nTensor Shape: {self.tensor.shape})\nFeatures:\n"
        table = [
            [feature, self[feature].min(), self[feature].max()]
            for feature in self.feature_names
        ]
        headers = ["Feature name", "Min", "Max"]
        table_string = str(tabulate(table, headers=headers, tablefmt="simple_outline"))
        return head + table_string

    def __or__(self, other: Union["NamedTensor", None]) -> "NamedTensor":
        """
        Concatenate two NamedTensors along the last dimension.
        """
        if other is None:
            return self

        if not isinstance(other, NamedTensor):
            raise ValueError("Can only concatenate NamedTensor with NamedTensor")

        # check features names are distinct between the two tensors
        if set(self.feature_names) & set(other.feature_names):
            raise ValueError(
                f"Feature names must be distinct between the two tensors for"
                f"unambiguous concat, self:{self.feature_names} other:{other.feature_names}"
            )

        if self.names != other.names:
            raise ValueError(
                f"NamedTensors must have the same dimension names to concatenate, self:{self.names} other:{other.names}"
            )
        try:
            return NamedTensor(
                torch.cat([self.tensor, other.tensor], dim=-1),
                self.names.copy(),
                self.feature_names + other.feature_names,
            )
        except Exception as e:
            raise ValueError(f"Error while concatenating {self} and {other}") from e

    def __ror__(self, other: Union["NamedTensor", None]) -> "NamedTensor":
        return self.__or__(other)

    @staticmethod
    def concat(nts: List["NamedTensor"]) -> "NamedTensor":
        """
        Safely concat a list of NamedTensors along the last dimension
        in one shot.
        """
        if len(nts) == 0:
            raise ValueError("Cannot concatenate an empty list of NamedTensors")
        if len(nts) == 1:
            return nts[0].clone()
        else:
            # Check features names are distinct between the n named tensors
            feature_names = set()
            for nt in nts:
                if feature_names & set(nt.feature_names):
                    raise ValueError(
                        f"Feature names must be distinct between the named tensors to concat\n"
                        f"Found duplicates: {feature_names & set(nt.feature_names)}"
                    )
                feature_names |= set(nt.feature_names)

            # Check that all named tensors have the same names
            if not all(nt.names == nts[0].names for nt in nts):
                raise ValueError(
                    "NamedTensors must have the same dimension names to concatenate"
                )

            # Concat in one shot
            return NamedTensor(
                torch.cat([nt.tensor for nt in nts], dim=-1),
                nts[0].names.copy(),
                list(chain.from_iterable(nt.feature_names for nt in nts)),
            )

    def dim_index(self, dim_name: str) -> int:
        """
        Return the index of a dimension given its name.
        """
        return self.names.index(dim_name)

    def clone(self):
        return NamedTensor(
            tensor=deepcopy(self.tensor).to(self.tensor.device),
            names=self.names.copy(),
            feature_names=self.feature_names.copy(),
        )

    def __getitem__(self, feature_name: str) -> torch.Tensor:
        """
        Get one feature from the features dimension of the tensor by name.
        The returned tensor has the same number of dimensions as the original tensor.
        """
        try:
            return self.select_dim(
                self.feature_dim_name, self.feature_names_to_idx[feature_name]
            ).unsqueeze(self.names.index(self.feature_dim_name))
        except KeyError:
            raise ValueError(
                f"Feature {feature_name} not found in {self.feature_names}"
            )

    def type_(self, new_type):
        """
        Modify the type of the underlying torch tensor
        by calling torch's .type method

        in_place operation for this class, the internal
        tensor is replaced by the new one.
        """
        self.tensor = self.tensor.type(new_type)

    def flatten_(self, flatten_dim_name: str, start_dim: int, end_dim: int):
        """
        Flatten the underlying tensor from start_dim to end_dim.
        Deletes flattened dimension names and insert
        the new one.
        """
        self.tensor = torch.flatten(self.tensor, start_dim, end_dim)

        # Remove the flattened dimensions from the names
        # and insert the replacing one
        self.names = (
            self.names[:start_dim] + [flatten_dim_name] + self.names[end_dim + 1 :]
        )

    def unflatten_(
        self, dim: int, unflattened_size: torch.Size, unflatten_dim_name: List
    ):
        """
        Unflatten the dimension dim of the underlying tensor.
        Insert unflattened_size dimension instead
        """
        self.tensor = self.tensor.unflatten(dim, unflattened_size)
        self.names = self.names[:dim] + [*unflatten_dim_name] + self.names[dim + 1 :]

    def squeeze_(self, dim_name: Union[List[str], str]):
        """
        Squeeze the underlying tensor along the dimension(s)
        given its/their name(s).
        """
        if isinstance(dim_name, str):
            dim_name = [dim_name]
        dim_indices = [self.names.index(name) for name in dim_name]
        self.tensor = torch.squeeze(self.tensor, dim=dim_indices)
        for name in dim_name:
            self.names.remove(name)

    def unsqueeze_(self, dim_name: str, dim_index: int):
        """ "
        Insert a new dimension dim_name of size 1 at dim_index
        """
        self.tensor = torch.unsqueeze(self.tensor, dim_index)
        self.names.insert(dim_index, dim_name)

    def select_dim(
        self, dim_name: str, index: int, bare_tensor: bool = True
    ) -> Union["NamedTensor", torch.Tensor]:
        """
        Return the tensor indexed along the dimension dim_name
        with the index index.
        The given dimension is removed from the tensor.
        See https://pytorch.org/docs/stable/generated/torch.select.html
        """
        if bare_tensor:
            return self.tensor.select(self.names.index(dim_name), index)
        else:
            # if they try to select the features it will break as
            # the feature dimension is not present anymore
            return NamedTensor(
                self.select_dim(dim_name, index, bare_tensor=True),
                self.names[: self.names.index(dim_name)]
                + self.names[self.names.index(dim_name) + 1 :],
                self.feature_names,
                feature_dim_name=self.feature_dim_name,
            )

    def index_select_dim(
        self, dim_name: str, indices: torch.Tensor, bare_tensor: bool = True
    ) -> Union["NamedTensor", torch.Tensor]:
        """
        Return the tensor indexed along the dimension dim_name
        with the indices tensor.
        The returned tensor has the same number of dimensions as the original tensor (input).
        The dimth dimension has the same size as the length of index; other dimensions have
        the same size as in the original tensor.
        See https://pytorch.org/docs/stable/generated/torch.index_select.html
        """
        if bare_tensor:
            return self.tensor.index_select(
                self.names.index(dim_name),
                torch.Tensor(indices).type(torch.int64).to(self.device),
            )
        else:
            return NamedTensor(
                self.index_select_dim(dim_name, indices, bare_tensor=True),
                self.names,
                (
                    self.feature_names
                    if dim_name != self.feature_dim_name
                    else [self.feature_names[i] for i in indices]
                ),
                feature_dim_name=self.feature_dim_name,
            )

    def dim_size(self, dim_name: str) -> int:
        """
        Return the size of a dimension given its name.
        """
        try:
            return self.tensor.size(self.names.index(dim_name))
        except ValueError as ve:
            raise ValueError(f"Dimension {dim_name} not found in {self.names}") from ve

    @property
    def spatial_dim_idx(self) -> List[int]:
        """
        Return the indices of the spatial dimensions in the tensor.
        """
        return sorted(
            self.names.index(name)
            for name in set(self.SPATIAL_DIM_NAMES).intersection(set(self.names))
        )

    def unsqueeze_and_expand_from_(self, other: "NamedTensor"):
        """
        Unsqueeze and expand the tensor to have the same number of spatial dimensions
        as another NamedTensor.
        Injects new dimensions where the missing names are.
        """
        missing_names = set(other.names) - set(self.names)
        missing_names &= set(self.SPATIAL_DIM_NAMES)

        if missing_names:
            index_to_unsqueeze = [
                (name, other.names.index(name)) for name in missing_names
            ]
            for name, idx in sorted(index_to_unsqueeze, key=lambda x: x[1]):
                self.tensor = torch.unsqueeze(self.tensor, idx)
                self.names.insert(idx, name)

            expander = []
            for _, name in enumerate(self.names):
                expander.append(other.dim_size(name) if name in missing_names else -1)

            self.tensor = self.tensor.expand(*expander)

    @staticmethod
    def new_like(tensor: torch.Tensor, other: "NamedTensor") -> "NamedTensor":
        """
        Create a new NamedTensor with the same names and feature names as another NamedTensor
        and a tensor of the same shape as the input tensor.
        """
        return NamedTensor(tensor, other.names.copy(), other.feature_names.copy())

    @staticmethod
    def expand_to_batch_like(
        tensor: torch.Tensor, other: "NamedTensor"
    ) -> "NamedTensor":
        """
        Create a new NamedTensor with the same names and feature names as another NamedTensor
        with an extra first dimension called 'batch' using the supplied tensor.
        Supplied new 'batched' tensor must have one more dimension than other.
        """
        names = ["batch"] + other.names
        if tensor.dim() != len(names):
            raise ValueError(
                f"Tensor dim {tensor.dim()} must match number of names {len(names)} with extra batch dim"
            )
        return NamedTensor(tensor, ["batch"] + other.names, other.feature_names.copy())

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def pin_memory_(self):
        """
        'In place' operation to pin the underlying tensor to memory.
        """
        self.tensor = self.tensor.pin_memory()

    def to_(self, *args, **kwargs):
        """
        'In place' operation to call torch's 'to' method on the underlying tensor.
        """
        self.tensor = self.tensor.to(*args, **kwargs)


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
class Statics(RegisterFieldsMixin):
    """
    Static fields of the dataset.
    Tensor can be registered as buffer in a lightning module
    using the register_buffers method.
    """

    # border_mask: torch.Tensor
    grid_statics: NamedTensor
    grid_shape: Tuple[int, int]
    border_mask: torch.Tensor = field(init=False)
    interior_mask: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.border_mask = self.grid_statics["border_mask"]
        self.interior_mask = 1.0 - self.border_mask

    @cached_property
    def meshgrid(self) -> torch.Tensor:
        """
        Return a tensor concatening X,Y
        """
        return einops.rearrange(
            torch.cat(
                [
                    self.grid_statics["x"],
                    self.grid_statics["y"],
                ],
                dim=-1,
            ),
            ("x y n -> n x y"),
        )


def generate_forcings(
    date: dt.datetime, output_terms: Tuple[float], grid: Grid
) -> List[NamedTensor]:
    """
    Generate all the forcing in this function.
    Return a list of NamedTensor.
    """
    # Datetime Forcing
    datetime_forcing = get_year_hour_forcing(date, output_terms).type(torch.float32)

    # Solar forcing, dim : [num_pred_steps, Lat, Lon, feature = 1]
    solar_forcing = generate_toa_radiation_forcing(
        grid.lat, grid.lon, date, output_terms
    ).type(torch.float32)

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
        NamedTensor(
            feature_names=[
                "toa_radiation",
            ],
            tensor=solar_forcing,
            names=["timestep", "lat", "lon", "features"],
        ),
    ]

    return lforcings


@dataclass(slots=True)
class DatasetInfo:
    """
    This dataclass holds all the informations
    about the dataset that other classes
    and functions need to interact with it.
    """

    name: str  # Name of the dataset
    domain_info: "DomainInfo"  # Information used for plotting
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
        print(f"Static fields {self.statics.grid_statics.feature_names}")
        print(f"Grid static features {self.statics.grid_statics}")
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
    start: dt.datetime
    end: dt.datetime
    step: int  # In hours, step btw the t0 of 2 samples
    name: str

    def __init__(self, start: int, end: int, step: int, name: str):
        self.start = dt.datetime.strptime(str(start), "%Y%m%d%H")
        self.end = dt.datetime.strptime(str(end), "%Y%m%d%H")
        self.step = step
        self.name = name

    @property
    def date_list(self):
        return np.arange(
            self.start, self.end, np.timedelta64(self.step, "h"), dtype="datetime64[s]"
        ).tolist()


def get_param_list(
    conf: dict,
    grid: Grid,
    load_param_info: Callable[[str], ParamConfig],
    get_weight_per_level: Callable[[str], float],
) -> List[Param]:
    param_list = []
    for name, values in conf["params"].items():
        for lvl in values["levels"]:
            param = Param(
                name=name,
                level=lvl,
                grid=grid,
                load_param_info=load_param_info,
                kind=values["kind"],
                get_weight_per_level=get_weight_per_level,
            )
            param_list.append(param)
    return param_list


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


@dataclass(slots=True)
class Sample:
    """Describes a sample"""

    date_t0: dt.datetime
    settings: Settings
    params: List[Param]
    stats: Stats
    grid: Grid
    pred_timesteps: Tuple[float] = field(init=False)  # gaps in hours btw step and t0
    all_dates: Tuple[dt.datetime] = field(init=False)  # date of each step
    members: List[int] = field(default_factory=[])

    def __post_init__(self):
        """Setups time variables to be able to define a sample.
        For example for n_inputs = 2, n_preds = 3, step_duration = 3h:
        all_steps = [-1, 0, 1, 2, 3]
        all_timesteps = [-3h, 0h, 3h, 6h, 9h]
        pred_timesteps = [3h, 6h, 9h]
        all_dates = [24/10/22 21:00,  24/10/23 00:00, 24/10/23 03:00, 24/10/23 06:00, 24/10/23 09:00]
        """
        n_inputs, n_preds = self.settings.num_input_steps, self.settings.num_pred_steps
        all_steps = list(range(-n_inputs + 1, n_preds + 1))
        all_timesteps = [self.settings.step_duration * step for step in all_steps]
        self.pred_timesteps = all_timesteps[-n_preds:]
        self.all_dates = [self.date_t0 + dt.timedelta(hours=ts) for ts in all_timesteps]

    def __repr__(self):
        return f"Date T0 {self.date_t0}, leadtimes {self.pred_timesteps}"

    def is_valid(self) -> bool:
        """Check that all the files necessary for this sample exist.

        Args:
            param_list (List): List of parameters
        Returns:
            Boolean: Whether the sample is available or not
        """
        for date in self.all_dates:
            for param in self.params:
                if not exists(
                    self.settings.dataset_name, param, date, self.settings.file_format
                ):
                    print("invalid sample")
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
                dates = self.all_dates[-self.settings.num_pred_steps :]
                tensor = get_param_tensor(
                    param, self.stats, dates, self.settings, no_standardize
                )
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))

            elif param.kind == "output":
                dates = self.all_dates[-self.settings.num_pred_steps :]
                tensor = get_param_tensor(
                    param, self.stats, dates, self.settings, no_standardize
                )
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))
                loutputs.append(tmp_state)

            else:  # input_output
                tensor = get_param_tensor(
                    param, self.stats, self.all_dates, self.settings, no_standardize
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
            date=self.date_t0, output_terms=self.pred_timesteps, grid=self.grid
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

        hours_delta = dt.timedelta(hours=self.settings.step_duration * step)
        plt.suptitle(f"Run: {self.date_t0} - Valid time: {self.date_t0 + hours_delta}")
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


class DatasetABC(Dataset):
    """
    Base class for gridded datasets used in weather forecasts
    """

    def __init__(
        self,
        name: str,
        grid: Grid,
        period: Period,
        params: List[Param],
        settings: Settings,
        accessor: DataAccessor,
    ):
        self.name = name
        self.grid = grid
        self.period = period
        self.params = params
        self.settings = settings
        self.shuffle = self.period.name == "train"
        self.cache_dir = accessor.get_dataset_path(name, grid)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        n_input, n_pred = self.settings.num_input_steps, self.settings.num_pred_steps
        valid_samples_filename = (
            f"valid_samples_{self.period.name}_{n_input}_{n_pred}.txt"
        )
        self.valid_samples_file = self.cache_dir / filename

    def __str__(self) -> str:
        return f"{self.name}_{self.grid.name}"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        """
        Return an item from an index of the sample_list
        """
        sample = self.sample_list[index]
        item = sample.load()
        return item

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
            weather_dim=self.input_output_dim,
            forcing_dim=self.input_dim,
            step_duration=self.settings.step_duration,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    def write_list_valid_samples(self):
        print(f"Writing list of valid samples for {self.period.name} set...")
        with open(self.valid_samples_file, "w") as f:
            for date in tqdm.tqdm(
                self.period.date_list, f"{self.period.name} samples validation"
            ):
                sample = Sample(date, self.settings, self.params, self.stats, self.grid)
                if sample.is_valid():
                    f.write(f"{date.strftime('%Y-%m-%d_%Hh%M')}\n")

    def torch_dataloader(self, tl_settings: TorchDataloaderSettings) -> DataLoader:
        """
        Builds a torch dataloader from self.
        """
        return DataLoader(
            self,
            batch_size=tl_settings.batch_size,
            num_workers=tl_settings.num_workers,
            shuffle=self.shuffle,
            prefetch_factor=tl_settings.prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=tl_settings.pin_memory,
        )

    @cached_property
    def input_dim(self) -> int:
        """
        Return the number of forcings.
        """
        res = 4  # For date
        res += 1  # For solar forcing

        for param in self.params:
            if param.kind == "input":
                res += 1
        return res

    @cached_property
    def input_output_dim(self) -> int:
        """
        Return the dimension of pronostic variable.
        """
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += 1
        return res

    @cached_property
    def output_dim(self):
        """
        Return dimensions of output variable only
        Not used yet
        """
        res = 0
        for param in self.params:
            if param.kind == "output":
                res += 1
        return res

    @abstractproperty
    def dataset_info(self) -> DatasetInfo:
        """
        Return the object DatasetInfo
        """

    @abstractproperty
    def cache_dir(self) -> Path:
        """
        Cache directory of the dataset.
        Used at least to get statistics.
        """
        pass

    @property
    def dataset_extra_statics(self) -> List[NamedTensor]:
        """
        Datasets can override this method to add
        more static data.
        Otionally, add the LandSea Mask to the statics."""

        if self.settings.add_landsea_mask:
            return [
                NamedTensor(
                    feature_names=["LandSeaMask"],
                    tensor=torch.from_numpy(self.grid.landsea_mask)
                    .type(torch.float32)
                    .unsqueeze(2),
                    names=["lat", "lon", "features"],
                )
            ]
        return []

    @cached_property
    def statics(self) -> Statics:
        return Statics(
            **{
                "grid_statics": grid_static_features(self.grid),
                "grid_shape": self.grid_shape,
            }
        )

    @cached_property
    def stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "parameters_stats.pt")

    @cached_property
    def diff_stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "diff_stats.pt")

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

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered. Usefull information for plotting."""
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )

    @classmethod
    def from_dict(
        cls,
        name: str,
        conf: dict,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
        accessor: DataAccessor,
    ) -> Tuple["DatasetABC", "DatasetABC", "DatasetABC"]:

        conf["grid"]["load_grid_info_func"] = accessor.load_grid_info
        grid = Grid(**conf["grid"])
        try:
            members = conf["members"]
        except KeyError:
            members = None

        param_list = get_param_list(
            conf, grid, accessor.load_param_info, accessor.get_weight_per_level
        )

        train_settings_settings = Settings(
            dataset_name=name,
            num_input_steps=num_input_steps,
            num_pred_steps=num_pred_steps_train,
            members=members,
            **conf["settings"],
        )
        train_period = Period(**conf["periods"]["train"], name="train")
        train_ds = DatasetABC(
            name, grid, train_period, param_list, train_settings, accessor
        )

        valid_settings = Settings(
            dataset_name=name,
            num_input_steps=num_input_steps,
            num_pred_steps=num_pred_steps_val_test,
            members=members,
            **conf["settings"],
        )
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        valid_ds = DatasetABC(
            name, grid, valid_period, param_list, valid_settings, accessor
        )

        test_period = Period(**conf["periods"]["test"], name="test")
        test_ds = DatasetABC(
            name, grid, test_period, param_list, valid_settings, accessor
        )

        return train_ds, valid_ds, test_ds

    @classmethod
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
