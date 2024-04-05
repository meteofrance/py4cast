"""
Base classes defining our software components
and their interfaces
"""

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union

import einops
import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_tensor_fn
from tqdm import tqdm

from pnia.plots import DomainInfo
from pnia.utils import RegisterFieldsMixin, torch_save


class StateVariableMetadata:
    """
    non dataclass storing StateVariable metadata.
    We do this to hide metadata from lightning and to avoid
    introspection of metadata shapes for batch size guess
    """

    def __init__(
        self, ndims: int, names: List[str], coordinates_name: List[str], *args, **kwargs
    ):
        self.ndims = ndims
        self.names = names
        self.coordinates_name = coordinates_name


@dataclass(slots=True)
class StateVariableValues:
    """
    Dataclass storing StateVariable values.
    """

    values: torch.Tensor


class StateVariable(StateVariableValues, StateVariableMetadata):
    """
    Here we use multiple inheritance to avoid a single
    dataclass which makes ightning confused about batch size.
    """

    def __init__(self, **kwargs):

        StateVariableValues.__init__(
            self,
            **{
                k: v
                for k, v in kwargs.items()
                if k in StateVariableValues.__match_args__
            },
        )
        StateVariableMetadata.__init__(self, **kwargs)

    def __str__(self):
        return (
            f" ndims : {self.ndims} \n names {self.names} \n"
            f" coordinates : {self.coordinates_name} \n values(size) :{self.values.shape} \n"
        )

    def __getattr__(self, name: str) -> torch.Tensor:
        """
        Return one element of a step variable picked by name.
        We assume that different features are in the last dimension.
        """
        if "names" in self.__dict__:
            if name in self.names:
                pos = [
                    self.names.index(name),
                ]
                pos = torch.tensor(pos, dtype=torch.int)

            else:
                raise AttributeError(f"StateVariable does not contains {name} feature.")
            return torch.index_select(self.values, -1, pos)
        else:
            raise AttributeError("names not defined.")

    def change_type(self, new_type):
        """
        Modify the type of the value
        """
        self.values = self.values.type(new_type)


@dataclass(slots=True)
class Item:
    """
    Dataclass holding one Item
    """

    inputs: List[StateVariable]
    outputs: List[StateVariable]

    forcing: Union[
        List[StateVariable], None
    ] = None  # Pour les champs de forcage (SST, Rayonnement, ... )

    # J'hesite entre inputs/outputs et pronostics/diagnostics
    # Les concepts sont totalement different (les variables ne contiendrait pas la meme chose,
    # le get_item serait different et le unroll_prediction aussi)
    # A discuter. J'aurais tendance a m'orienter vers pronostics/diagnostics qui me parait plus adapter
    # (et est en phase avec la denomination du CEP)
    # Ca ferait peut etre moins de trucs etrange dans les denominations

    def __str__(self):
        """
        Utility method to explore a batch/item shapes and names.
        """
        for attr in (f.name for f in fields(self)):
            if getattr(self, attr) is not None:
                for item in getattr(self, attr):
                    print(
                        f"{attr} {item.names} : {item.values.shape}, {item.values.size(0)}"
                    )


@dataclass
class ItemBatch(Item):
    """
    Dataclass holding a batch of items.
    """

    @cached_property
    def batch_size(self):
        return self.inputs[0].values.size(0)

    @cached_property
    def num_input_steps(self):
        return self.inputs[0].values.size(1)

    @cached_property
    def num_pred_steps(self):
        return self.outputs[0].values.size(1)


def collate_fn(items: List[Item]) -> ItemBatch:
    """
    Collate a list of item. Add one dimension to each state variable.
    Necessary to form batch.
    """
    # Here we postpone that for each batch the same dimension should be present.
    batch_of_items = {}

    for attr in (f.name for f in fields(Item)):
        batch_of_items[attr] = []
        item_zero_field_value = getattr(items[0], attr)
        if item_zero_field_value is not None:
            for i in range(len(item_zero_field_value)):
                new_values = collate_tensor_fn(
                    [getattr(item, attr)[i].values for item in items]
                ).type(torch.float32)

                new_state = StateVariable(
                    ndims=item_zero_field_value[i].ndims,
                    names=item_zero_field_value[
                        i
                    ].names,  # Contient le nom des différents niveaux
                    values=new_values,
                    coordinates_name=["batch"]
                    + item_zero_field_value[i].coordinates_name,  # Nom des coordonnées
                )
                batch_of_items[attr].append(new_state)
        else:
            batch_of_items[attr] = None
    return ItemBatch(**batch_of_items)


@dataclass
class Statics(RegisterFieldsMixin):
    """
    Static fields of the dataset.
    Tensor can be registered as buffer in a lightning module
    using the register_buffers method.
    """

    # border_mask: torch.Tensor
    grid_static_features: StateVariable
    # Remove this once we have a batch in the loss
    # Indeed, we will be able to use datasetInfo
    # information (stored with variable name)
    step_diff_mean: torch.Tensor
    step_diff_std: torch.Tensor
    data_mean: torch.Tensor
    data_std: torch.Tensor
    state_weights: torch.Tensor
    grid_shape: Tuple[int, int]
    border_mask: torch.Tensor = field(init=False)
    interior_mask: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.border_mask = self.grid_static_features.border_mask
        self.interior_mask = 1.0 - self.border_mask

    @cached_property
    def meshgrid(self) -> torch.Tensor:
        """
        Return a tensor concatening X,Y
        """
        return einops.rearrange(
            torch.cat(
                [
                    self.grid_static_features.x,
                    self.grid_static_features.y,
                ],
                dim=-1,
            ),
            ("x y n -> n x y"),
        )

    @cached_property
    def N_interior(self):
        return torch.sum(self.interior_mask).item()


@dataclass
class Stats:
    fname: Path

    def __getitem__(self, shortname: str):
        return self.stats[shortname]

    @cached_property
    def stats(self) -> Dict:
        """
        The file should contain a dictionnary like structure with the shortname as the first key and
        statistic stored as a second key.
        d[{shortname}][{stat}] = value

        Ex. :
           d["u10"]["mean"] = 5
           d["u10"]["std"] = 0.1

        This files could have been generated by
          - compute_parameters_stats
          - compute_time_step_stats
        """
        return torch.load(self.fname, "cpu")

    def to_list(
        self, aggregate: Literal["mean", "std", "min", "max"], shortnames: List[str]
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
        return [self[name][aggregate] for name in shortnames]


@dataclass(slots=True)
class DatasetInfo:
    """
    This dataclass holds all the informations
    about the dataset that other classes
    and functions need to interact with it.
    """

    name: str  # Name of the dataset
    domain_info: DomainInfo  # Information used for plotting
    shortnames: Callable
    units: Callable
    weather_dim: int
    forcing_dim: int
    step_duration: float  # Duration (in hour) of one step in the dataset. 0.25 means 15 minutes.
    statics: Statics  # A lot of static variable
    # Wait for batch in loss to use them instead of statics.
    stats: Stats  # Not used yet except in summary
    diff_stats: Stats  # Not used yet except in summary
    state_weights: Dict[str, float]  # Not used yet except in summary

    def summary(self):
        """
        Print a table summarizing variables present in the dataset (and their role)
        """
        print(f"\n Summarizing {self.name} \n")
        for p in ["forcing", "input_output", "diagnostic"]:
            names = self.shortnames(p)
            mean = self.stats.to_list("mean", names)
            std = self.stats.to_list("std", names)
            mini = self.stats.to_list("min", names)
            maxi = self.stats.to_list("max", names)
            if p != "forcing":
                diff_mean = self.diff_stats.to_list("mean", names)
                diff_std = self.diff_stats.to_list("std", names)
                weight = [self.state_weights[name] for name in names]
                data = list(
                    zip(names, mean, std, mini, maxi, diff_mean, diff_std, weight)
                )
                table = tabulate(
                    data,
                    headers=[
                        "Name",
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
                data = list(zip(names, mean, std, mini, maxi))
                table = tabulate(
                    data,
                    headers=["Name", "Mean", "Std", "Minimun", "Maximum"],
                    tablefmt="simple_outline",
                )
            if data:
                print(p.upper())  # Print the kind of variable
                print(table)  # Print the table


@dataclass(slots=True)
class TorchDataloaderSettings:
    """
    Settings for the torch dataloader
    """

    batch_size: int = 1
    num_workers: int = 10
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
    def limited_area(self) -> bool:
        """
        Returns True if the dataset is
        compatible with Limited area models
        """

    @abstractproperty
    def split(self) -> Literal["train", "valid", "test"]:
        pass

    @abstractproperty
    def weather_params(self) -> List[str]:
        """
        Return the name of all the variable present in the dataset
        """

    @abstractproperty
    def cache_dir(self) -> Path:
        """
        Cache directory of the dataset.
        Used at least to get statistics.
        """
        pass

    def compute_parameters_stats(self):
        """
        Compute mean and standard deviation for this dataset.
        """
        means = []
        squares = []
        mins = []
        maxs = []
        print(
            "We are going to compute statistics on the dataset. This can take a while."
        )
        if self.settings.standardize:
            raise ValueError("Your dataset should not be standardized.")
        # When computing stat may force every body to be input/ouput
        for batch in tqdm(self.torch_dataloader()):
            # Here we postpone that data are in 2 or 3 D
            inputs = torch.cat([x.values for x in batch.inputs], dim=-1)
            outputs = torch.cat([x.values for x in batch.outputs], dim=-1)
            # Check that no variable is a forcing variable
            f_names = [x.names for x in batch.forcing if x.ndims == 2]
            if len(f_names) > 0:
                raise ValueError(
                    "During statistics computation no forcing are accepted. All variables should be in in/out mode."
                )

            in_out = torch.cat([inputs, outputs], dim=1)
            in_out = in_out.flatten(
                1, 3
            )  # Flatten everybody to be (Batch, X, Features)

            means.append(in_out.mean(dim=1))
            squares.append((in_out**2).mean(dim=1))
            maxi, _ = torch.max(in_out, 1)
            mini, _ = torch.min(in_out, 1)
            mins.append(mini)  # (N_batch, d_features,)
            maxs.append(maxi)  # (N_batch, d_features,)

        mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
        second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
        std = torch.sqrt(second_moment - mean**2)  # (d_features)
        maxi, _ = torch.max(torch.cat(maxs, dim=0), dim=0)
        mini, _ = torch.min(torch.cat(mins, dim=0), dim=0)

        # Store in dictionnary
        store_d = {}
        # Get variables names
        names = []
        [names.extend(x.names) for x in batch.inputs]
        # Storing variable statistics
        for i, name in enumerate(names):
            store_d[name] = {
                "mean": mean[i],
                "std": std[i],
                "min": mini[i],
                "max": maxi[i],
            }
        torch_save(store_d, self.cache_dir / "parameters_stats.pt")

    def compute_time_step_stats(self):
        diff_means = []
        diff_squares = []
        if not self.settings.standardize:
            print(
                f"Your dataset {self} is not normalized. If you do not normalized your output it could be intended."
            )
            print("Otherwise consider standardizing your inputs.")
        for batch in tqdm(self.torch_dataloader()):
            # Here we postpone that data are in 2 or 3 D
            inputs = torch.cat([x.values for x in batch.inputs], dim=-1)
            outputs = torch.cat([x.values for x in batch.outputs], dim=-1)
            # Check that no variable is a forcing variable
            f_names = [x.names for x in batch.forcing if x.ndims == 2]
            if len(f_names) > 0:
                raise ValueError(
                    "During statistics computation no forcing are accepted. All variables should be in in/out mode."
                )

            in_out = torch.cat([inputs, outputs], dim=1)
            diff = (
                in_out[:, 1:] - in_out[:, :-1]
            )  # Substract information on time dimension
            diff = diff.flatten(1, 3)  # Flatten everybody to be (Batch, X, Features)

            diff_means.append(diff.mean(dim=1))
            diff_squares.append((diff**2).mean(dim=1))

        # This should not work
        # We should be aware that dataset shoul
        diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)
        diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)
        store_d = {}
        # Get variables names
        names = []
        [names.extend(x.names) for x in batch.inputs]
        # Storing variable statistics
        for i, name in enumerate(names):
            store_d[name] = {
                "mean": diff_mean[i],
                "std": diff_std[i],
            }
        torch_save(store_d, self.cache_dir / "diff_stats.pt")

    def dataset_extra_statics(self) -> List[StateVariable]:
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
        pos_max = torch.max(torch.abs(grid_xy))
        grid_xy = (
            einops.rearrange(grid_xy, ("n x y -> x y n")) / pos_max
        )  # Rearange and divide  by maximum coordinate

        # (Nx, Ny, 1)
        geopotential = torch.tensor(self.geopotential_info).unsqueeze(
            2
        )  # (N_x, N_y, 1)
        gp_min = torch.min(geopotential)
        gp_max = torch.max(geopotential)
        # Rescale geopotential to [0,1]
        geopotential = (geopotential - gp_min) / (gp_max - gp_min)  # (N_x,N_y, 1)
        grid_border_mask = torch.tensor(self.border_mask).unsqueeze(2)  # (N_x, N_y,1)

        batch_names = []
        for x in self.dataset_extra_statics:
            batch_names += x.names
        state_var = StateVariable(
            ndims=2,  # Champ 2D
            values=torch.cat(
                [grid_xy, geopotential, grid_border_mask]
                + [x.values for x in self.dataset_extra_statics],
                dim=-1,
            ),
            names=["x", "y", "geopotential", "border_mask"]
            + batch_names,  # Noms des champs 2D
            coordinates_name=["lat", "lon", "features"],
        )
        state_var.change_type(torch.float32)
        return state_var

    @cached_property
    def statics(self) -> Statics:
        # When manipulating tensor, we postpone that field will be stored in the same order
        # This may not be the case
        step_diff_mean = self.diff_stats.to_list(
            "mean", self.shortnames("input_output")
        )
        step_diff_std = self.diff_stats.to_list("std", self.shortnames("input_output"))
        data_mean = self.stats.to_list("mean", self.shortnames("input_output"))
        data_std = self.stats.to_list("std", self.shortnames("input_output"))

        # Load loss weighting vectors
        state_weights = torch.tensor(
            self.tensor_state_weights, dtype=torch.float32
        )  # (d_f,)
        return Statics(
            **{
                "grid_static_features": self.grid_static_features,
                "grid_shape": self.grid_shape,
                # Should be removed when using batch
                "step_diff_mean": torch.stack(step_diff_mean).type(torch.float32),
                "step_diff_std": torch.stack(step_diff_std).type(torch.float32),
                "data_mean": torch.stack(data_mean).type(torch.float32),
                "data_std": torch.stack(data_std).type(torch.float32),
                "state_weights": state_weights.type(torch.float32),
            }
        )

    @abstractproperty
    def tensor_state_weights(self):
        """
        Load the weights as a tensor.
        """
        pass

    @cached_property
    def stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "parameters_stats.pt")

    @cached_property
    def diff_stats(self) -> Stats:
        return Stats(
            fname=self.cache_dir / "diff_stats.pt"
        )  # "diff_parameters_stats2.pt")

    @abstractproperty
    def diagnostic_dim(self) -> int:
        """
        Return the number of diagnostic variables (output only)
        """
        pass

    @abstractclassmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
    ) -> Tuple["DatasetABC", "DatasetABC", "DatasetABC"]:
        """
        Load a dataset from a json file + the number of expected timesteps
        taken as inputs (num_input_steps) and to predict (num_pred_steps)
        Return the train, valid and test datasets, in that order
        """
        pass
