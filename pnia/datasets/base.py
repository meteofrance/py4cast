"""
Base classes defining our software components
and their interfaces
"""

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Tuple, Union

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_tensor_fn
from tqdm import tqdm

from pnia.base import RegisterFieldsMixin
from pnia.utils import torch_save


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

    def __repr__(self):
        return (
            f" ndims : {self.ndims} \n names {self.names} \n"
            f" coordinates : {self.coordinates_name} \n values(size) :{self.values.shape} \n"
        )

    def __getattr__(self, name: str) -> torch.Tensor:
        """
        Return one element of a step variable picked by name.
        We assume that different features are in the las dimension.
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

    def print_shapes_and_names(self):
        """
        Utility method to explore a batch/item shapes and names.
        """
        for attr in (f.name for f in fields(self)):
            if getattr(self, attr) is not None:
                for item in getattr(self, attr):
                    print(
                        f"{attr} {item.names} : {item.values.shape}, {item.values.size(0)}"
                    )


@dataclass(slots=True)
class ItemBatch(Item):
    """
    Dataclass holding a batch of items.
    """


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
    step_diff_mean: torch.Tensor
    step_diff_std: torch.Tensor
    data_mean: torch.Tensor
    data_std: torch.Tensor
    param_weights: torch.Tensor
    grid_shape: Tuple[int, int]
    # grid_info: np.ndarray
    border_mask: torch.Tensor = field(init=False)
    interior_mask: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.border_mask = self.grid_static_features.border_mask
        self.interior_mask = 1.0 - self.border_mask

    @cached_property
    def grid_info(self):
        # On fait l'inverse de la creation
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


class AbstractDataset(ABC):
    @abstractmethod
    def torch_dataloader(self, tl_settings: TorchDataloaderSettings) -> DataLoader:
        """
        Builds a torch dataloader from self.
        """

    @abstractproperty
    def grid_info(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (lat, lon) values
        """

    @cached_property
    def grid_shape(self) -> tuple:
        lat, _ = self.grid_info
        return lat.shape

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
    def border_mask(self) -> np.array:
        pass

    @abstractproperty
    def split(self) -> Literal["train", "valid", "test"]:
        pass

    @abstractproperty
    def weather_params(self) -> List[str]:
        pass

    @abstractproperty
    def parameter_weights(self) -> np.array:
        pass

    @abstractproperty
    def cache_dir(self) -> Path:
        pass

    def compute_parameters_stats(self):
        """
        Compute mean and standard deviation for this dataset.
        """
        # TODO : stats seems to be computed on cpu (but taking all the process) even if a gpu is available.
        means = []
        squares = []
        mins = []
        maxs = []
        flux_means = []
        flux_squares = []
        flux_mins = []
        flux_maxs = []

        weather_names = self.shortnames("input_output")
        forcing_names = self.shortnames("forcing")
        if weather_names == forcing_names:
            raise ValueError(
                "You may have a problem with your name : "
                + f"your forcing and weather variable are the same and are {forcing_names}."
            )

        forced_len = len(self.shortnames("forcing"))  # Nombre de forcage
        for init_batch, target_batch, _, forcing_batch in tqdm(self.torch_dataloader()):
            batch = torch.cat((init_batch, target_batch), dim=1)
            means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
            squares.append(torch.mean(batch**2, dim=(1, 2)))  # (N_batch, d_features,)
            maxi, _ = torch.max(batch.flatten(1, 2), 1)
            mini, _ = torch.min(batch.flatten(1, 2), 1)
            mins.append(mini)  # (N_batch, d_features,)
            maxs.append(maxi)  # (N_batch, d_features,)

            flux_batch = forcing_batch[
                :, :, :, :forced_len
            ]  # Removing date. It is cleaner than postponing forcing is the first variable
            # (as it allows multiple forcing).
            flux_means.append(
                torch.mean(flux_batch, dim=(1, 2))
            )  # (N_batch, d_forcing,)
            flux_squares.append(
                torch.mean(flux_batch**2, dim=(1, 2))
            )  # (N_batch, d_forcing,)
            maxi, _ = torch.max(flux_batch.flatten(1, 2), 1)
            mini, _ = torch.min(flux_batch.flatten(1, 2), 1)
            flux_mins.append(mini)
            flux_maxs.append(maxi)

        mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
        second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
        std = torch.sqrt(second_moment - mean**2)  # (d_features)
        maxi, _ = torch.max(torch.cat(maxs, dim=0), dim=0)
        mini, _ = torch.min(torch.cat(mins, dim=0), dim=0)

        flux_mean = torch.mean(torch.cat(flux_means, dim=0), dim=0)  # # (d_forcing)
        flux_second_moment = torch.mean(
            torch.cat(flux_squares, dim=0), dim=0
        )  # # (d_forcing)
        flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
        flux_max, _ = torch.max(torch.cat(flux_maxs, dim=0), dim=0)
        flux_min, _ = torch.min(torch.cat(flux_mins, dim=0), dim=0)
        # Store in dictionnary
        store_d = {}
        # Storing variable statistics
        for i, name in enumerate(self.shortnames("input_output")):
            store_d[name] = {
                "mean": mean[i],
                "std": std[i],
                "min": mini[i],
                "max": maxi[i],
            }

        # Storing flux statistics
        for i, name in enumerate(self.shortnames("forcing")):
            store_d[name] = {
                "mean": flux_mean[i],
                "std": flux_std[i],
                "min": flux_min[i],
                "max": flux_max[i],
            }
        torch_save(store_d, self.cache_dir / "parameters_stats.pt")

    def compute_timestep_stats(self):
        """Computes stats on the difference between 2 timesteps."""
        # Not sure that it should be exactly like this for every dataset
        print(f"Computing one-step difference mean and std.-dev for {self}...")
        diff_means = []
        diff_squares = []
        for init_batch, target_batch, _, _ in tqdm(self.torch_dataloader()):
            batch = torch.cat(
                (init_batch, target_batch), dim=1
            )  # (Nbatch, Nt, Ngrid, d_features)

            batch_diffs = (
                batch[:, 1:] - batch[:, :-1]
            )  # (Nbatch, N_t-1, Ngrid, d_features)

            diff_means.append(
                torch.mean(batch_diffs, dim=(1, 2))
            )  # (Nbatch, d_features)
            diff_squares.append(
                torch.mean(batch_diffs**2, dim=(1, 2))
            )  # (Nbatch, d_features)

        diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
        diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

        store_d = {}
        # Storing variable statistics
        for i, name in enumerate(self.shortnames("input_output")):
            store_d[name] = {"mean": diff_mean[i], "std": diff_std[i]}
        print("Saving one-step difference mean and std.-dev...")
        torch_save(store_d, self.cache_dir / "diff_stats.pt")

    @cached_property
    def timestep_stats(self):
        if self.recompute_stats or not (self.cache_dir / "diff_stats.pt").exists():
            self.compute_timestep_stats()
        return torch.load(self.cache_dir / "diff_stats.pt")

    @abstractproperty
    def recompute_stats(self) -> bool:
        """
        If True, the stats will be recomputed
        """

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
        xy = self.grid_info  # (2, N_x, N_y)
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

        # Load step diff stats
        diff_stats = self.timestep_stats  # (d_f,)
        step_diff_mean = []
        step_diff_std = []
        for name in self.shortnames("input_output"):
            if diff_stats[name]["std"] == 0:
                print(f"Error : std nulle for {name}")
                # Faire en sorte que ca passe en fixant std a 1 et mean a 0
            step_diff_mean.append(diff_stats[name]["mean"])
            step_diff_std.append(diff_stats[name]["std"])

        # Load parameter std for computing validation errors in original data scale
        stats = torch.load(self.cache_dir / "parameters_stats.pt")
        data_mean = []
        data_std = []
        for name in self.shortnames("input_output"):
            if stats[name]["std"] == 0:
                print(f"Error : std nulle for {name}")
                # Faire en sorte que ca passe en fixant std a 1 et mean a 0
            data_mean.append(stats[name]["mean"])
            data_std.append(stats[name]["std"])

        # Load loss weighting vectors
        param_weights = torch.tensor(
            self.parameter_weights, dtype=torch.float32
        )  # (d_f,)
        return Statics(
            **{
                # "border_mask": border_mask.type(torch.float32),
                "grid_static_features": self.grid_static_features,
                "step_diff_mean": torch.stack(step_diff_mean).type(torch.float32),
                "step_diff_std": torch.stack(step_diff_std).type(torch.float32),
                "data_mean": torch.stack(data_mean).type(torch.float32),
                "data_std": torch.stack(data_std).type(torch.float32),
                "grid_shape": self.grid_shape,
                # "grid_info": self.grid_info,
                "param_weights": param_weights.type(torch.float32),
            }
        )

    @cached_property
    def statistics(self, device="cpu"):
        stats = torch.load(self.cache_dir / "parameters_stats.pt", device)

        data_mean = []
        data_std = []
        data_max = []
        data_min = []
        for name in self.shortnames("input_output"):
            if stats[name]["std"] == 0:
                print(f"Error : std nulle for {name}")
                # Faire en sorte que ca passe en fixant std a 1 et mean a 0
            data_std.append(stats[name]["std"])
            data_mean.append(stats[name]["mean"])
            data_max.append(stats[name]["max"])
            data_min.append(stats[name]["min"])

        flux_mean = []
        flux_std = []
        flux_max = []
        flux_min = []
        for name in self.shortnames("forcing"):
            flux_mean.append(stats[name]["mean"])
            flux_std.append(stats[name]["std"])
            flux_min.append(stats[name]["min"])
            flux_max.append(stats[name]["max"])

        return {
            "data_mean": torch.stack(data_mean).type(torch.float32),
            "data_std": torch.stack(data_std).type(torch.float32),
            "data_min": torch.stack(data_min).type(torch.float32),
            "data_max": torch.stack(data_max).type(torch.float32),
            "flux_mean": torch.stack(flux_mean).type(torch.float32),
            "flux_std": torch.stack(flux_std).type(torch.float32),
            "flux_min": torch.stack(flux_min).type(torch.float32),
            "flux_max": torch.stack(flux_max).type(torch.float32),
        }

    @cached_property
    def data_mean(self):
        return self.statistics["data_mean"]

    @cached_property
    def data_std(self):
        return self.statistics["data_std"]

    @cached_property
    def flux_mean(self):
        return self.statistics["flux_mean"]

    @cached_property
    def flux_std(self):
        return self.statistics["flux_std"]

    @abstractproperty
    def forcing_dim(self) -> int:
        """
        Return the number of the forcing features (including date)
        """
        pass

    @abstractproperty
    def weather_dim(self) -> int:
        """
        Return the number of weather parameter features
        """
        pass

    @abstractproperty
    def diagnostic_dim(self) -> int:
        """
        Return the number of diagnostic variables (output only)
        """
        pass

    @abstractmethod
    def shortnames(self, kind) -> list:
        pass

    @abstractmethod
    def units(self, kind) -> list:
        pass

    @abstractproperty
    def grid_limits(self) -> list:
        pass

    @abstractproperty
    def projection(self):
        """
        Return the cartopy projection of the dataset
        """
        pass
