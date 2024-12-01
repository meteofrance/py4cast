import einops
import torch
from tqdm import tqdm
from typing import List, Literal
import warnings

from py4cast.utils import torch_save
from base import NamedTensor, DatasetABC, Grid


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

def compute_time_step_stats(dataset : DatasetABC):
    random_inputs = next(iter(dataset.torch_dataloader())).inputs
    n_features = len(random_inputs.feature_names)
    sum_means = torch.zeros(n_features)
    sum_squares = torch.zeros(n_features)
    counter = 0
    if not dataset.settings.standardize:
        raise ValueError("Your dataset should be standardized.")

    for batch in tqdm(dataset.torch_dataloader()):
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
    dest_file = dataset.cache_dir / "diff_stats.pt"
    torch_save(store_d, dataset.cache_dir / "diff_stats.pt")
    print(f"Parameters time diff stats saved in {dest_file}")
    
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
        geopotential = torch.tensor(grid.geopotential_info).unsqueeze(
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

        grid_border_mask = torch.tensor(grid.border_mask).unsqueeze(2)  # (N_x, N_y,1)

        feature_names = []
        for x in grid.landsea_mask:
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