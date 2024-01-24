"""
Base classes defining our software components
and their interfaces
"""

from abc import ABC, abstractproperty,abstractmethod
from typing import List, Literal

import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class Statics: 
    border_mask:torch.Tensor
    grid_static_features:torch.Tensor
    step_diff_mean:torch.Tensor
    step_diff_std:torch.Tensor
    data_mean:torch.Tensor
    data_std:torch.Tensor
    param_weights:torch.Tensor

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
    def parameter_weights(self)->np.array:
        pass

    def load_file(self, fname, device = "cpu"): 
        return torch.load(self.cache_dir/fname, map_location=device)

    def compute_parameters_stats(self): 
        """
        Compute mean and standard deviation for this dataset
        """
        means = []
        squares = []
        flux_mean = []
        flux_square = []
        for init_batch, target_batch, _, forcing_batch in tqdm(self.loader):
            batch = torch.cat((init_batch, target_batch), dim=1)  
            
            means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
            squares.append(torch.mean(batch**2, dim=(1, 2)))  # (N_batch, d_features,)
            flux_batch = forcing_batch[:, :, :, -4]  # Removing date. It is cleaner than postponing forcing is the first variable (as it allows multiple forcing). 
            flux_means.append(torch.mean(flux_batch))  # (,)
            flux_squares.append(torch.mean(flux_batch**2))  # (,)
        mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
        second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
        std = torch.sqrt(second_moment - mean**2)  # (d_features)

        flux_mean = torch.mean(torch.stack(flux_means))  # (,)
        flux_second_moment = torch.mean(torch.stack(flux_squares))  # (,)
        flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
        flux_stats = torch.stack((flux_mean, flux_std))

        torch.save(mean, self.cache_dir / "parameter_mean.pt")
        torch.save(std, self.cache_dir / "parameter_std.pt")
        torch.save(flux_stats, self.cache_dir / "flux_stats.pt")

    def compute_timestep_stats(self):
        """Computes stats on the difference between 2 timesteps."""
        # Not sure that it should be exactly like this for every dataset 
        diff_means = []
        diff_squares = []
        for init_batch, target_batch, _, _ in tqdm(self.loader):
            batch = torch.cat(
                (init_batch, target_batch), dim=1)  # (Nbatch, Nt, Ngrid, d_features)

            batch_diffs = batch[:, 1:] - batch[:, :-1]  # (Nbatch, N_t-1, Ngrid, d_features)

            diff_means.append(torch.mean(batch_diffs, dim=(1, 2)))  # (Nbatch, d_features)
            diff_squares.append(
                torch.mean(batch_diffs**2, dim=(1, 2))
            )  # (Nbatch, d_features)

        diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
        diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

        print("Saving one-step difference mean and std.-dev...")
        torch.save(diff_mean, self.cache_dir / "diff_mean.pt")
        torch.save(diff_std, self.cache_dir / "diff_std.pt")


    def load_static_data(self, device="cpu")->Statics:
        grid_static_features = self.load_file("grid_features.pt") # (N_grid, d_grid_static)
        border_mask = grid_static_features[:,3].unsqueeze(1)


        # Load step diff stats
        step_diff_mean = self.load_file("diff_mean.pt") # (d_f,)
        step_diff_std = self.load_file("diff_std.pt") # (d_f,)
        
        # Load parameter std for computing validation errors in original data scale
        data_mean = self.load_file("parameter_mean.pt") # (d_features,)
        data_std = self.load_file("parameter_std.pt") # (d_features,)

        # Load loss weighting vectors
        param_weights = torch.tensor(self.parameter_weights, dtype=torch.float32, device=device) # (d_f,)
        return Statics(**{
            "border_mask": border_mask.type(torch.float32),
            "grid_static_features": grid_static_features.type(torch.float32),
            "step_diff_mean": step_diff_mean.type(torch.float32),
            "step_diff_std": step_diff_std.type(torch.float32),
            "data_mean": data_mean.type(torch.float32),
            "data_std": data_std.type(torch.float32),
            "param_weights": param_weights.type(torch.float32)
        })
   
    @abstractproperty
    def static_feature_dim(self)->int:
        """
        Return the number of static feature of the dataset
        """
        pass 

    @abstractproperty
    def forcing_dim(self)->int: 
        """
        Return the number of the forcing features (including date)
        """
        pass

    @abstractproperty 
    def weather_dim(self)-> int:
        """
        Return the number of weather parameter features 
        """
        pass

    @abstractproperty
    def diagnostic_dim(self)-> int:
        """
        Return the number of diagnostic variables (output only)
        """
        pass
