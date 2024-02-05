"""
Base classes defining our software components
and their interfaces
"""

from abc import ABC, abstractproperty
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pnia.utils import torch_save


@dataclass
class Statics:
    border_mask: torch.Tensor
    grid_static_features: torch.Tensor
    step_diff_mean: torch.Tensor
    step_diff_std: torch.Tensor
    data_mean: torch.Tensor
    data_std: torch.Tensor
    param_weights: torch.Tensor


class AbstractDataset(ABC):
    @abstractproperty
    def loader(self) -> DataLoader:
        pass

    @abstractproperty
    def grid_info(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (lat, lon) values
        """

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
    def standardize(self) -> bool:
        pass

    @abstractproperty
    def nb_pred_steps(self) -> int:
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
        for init_batch, target_batch, _, forcing_batch in tqdm(self.loader):
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
        diff_means = []
        diff_squares = []
        for init_batch, target_batch, _, _ in tqdm(self.loader):
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

    def load_static_data(self, device="cpu") -> Statics:
        # (N_grid, d_grid_static)
        grid_static_features = torch.load(self.cache_dir / "grid_features.pt", device)
        border_mask = grid_static_features[:, 3].unsqueeze(1)

        # Load step diff stats
        diff_stats = torch.load(self.cache_dir / "diff_stats.pt", device)  # (d_f,)
        step_diff_mean = []
        step_diff_std = []
        for name in self.shortnames("input_output"):
            if diff_stats[name]["std"] == 0:
                print(f"Error : std nulle for {name}")
                # Faire en sorte que ca passe en fixant std a 1 et mean a 0
            step_diff_mean.append(diff_stats[name]["mean"])
            step_diff_std.append(diff_stats[name]["std"])

        # Load parameter std for computing validation errors in original data scale
        stats = torch.load(self.cache_dir / "parameters_stats.pt", device)
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
            self.parameter_weights, dtype=torch.float32, device=device
        )  # (d_f,)
        return Statics(
            **{
                "border_mask": border_mask.type(torch.float32),
                "grid_static_features": grid_static_features.type(torch.float32),
                "step_diff_mean": torch.stack(step_diff_mean).type(torch.float32),
                "step_diff_std": torch.stack(step_diff_std).type(torch.float32),
                "data_mean": torch.stack(data_mean).type(torch.float32),
                "data_std": torch.stack(data_std).type(torch.float32),
                "param_weights": param_weights.type(torch.float32),
            }
        )

    def load_dataset_stats(self, device="cpu"):
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

    @abstractproperty
    def static_feature_dim(self) -> int:
        """
        Return the number of static feature of the dataset
        """
        pass

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
