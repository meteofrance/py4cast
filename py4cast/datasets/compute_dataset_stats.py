import warnings
from typing import Literal

import torch
from tqdm import tqdm

from py4cast.datasets.base import DatasetABC
from py4cast.utils import torch_save


def compute_mean_std_min_max(
    dataset: DatasetABC, type_tensor: Literal["inputs", "outputs", "forcing"]
):
    """
    Compute mean and standard deviation for this dataset.
    """
    random_batch = next(iter(dataset.torch_dataloader()))
    named_tensor = getattr(random_batch, type_tensor)
    n_features = len(named_tensor.feature_names)
    sum_means = torch.zeros(n_features)
    sum_squares = torch.zeros(n_features)
    ndim_features = len(named_tensor.tensor.shape) - 1
    flat_input = named_tensor.tensor.flatten(0, ndim_features - 1)  # (X, Features)
    best_min = torch.min(torch.nan_to_num(flat_input, nan=torch.inf), dim=0).values
    best_max = torch.max(torch.nan_to_num(flat_input, nan=-torch.inf), dim=0).values

    if torch.isnan(flat_input).any():
        warnings.warn(
            "py4cast.datasets.compute_dataset_stats.compute_mean_std_min_max:\n\
            \tYour dataset contain NaN values, statistics will be calculated ignoring the NaN."
        )

    counter = 0
    if dataset.settings.standardize:
        raise ValueError("Your dataset should not be standardized.")

    for batch in tqdm(
        dataset.torch_dataloader(),
        desc=f"Computing {type_tensor} stats",
    ):
        tensor = getattr(batch, type_tensor).tensor
        tensor = tensor.flatten(1, 3)  # Flatten to be (Batch, X, Features)
        counter += tensor.shape[0]  # += batch size

        sum_means += torch.nansum(tensor.nanmean(dim=1), dim=0)  # (d_features)
        sum_squares += torch.nansum((tensor**2).nanmean(dim=1), dim=0)  # (d_features)

        mini = torch.min(torch.nan_to_num(tensor, nan=torch.inf), 1).values[0]
        stack_mini = torch.stack([best_min, mini], dim=0)
        best_min = torch.min(stack_mini, dim=0).values  # (d_features)

        maxi = torch.max(torch.nan_to_num(tensor, nan=-torch.inf), 1).values[0]
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


def compute_parameters_stats(dataset: DatasetABC):
    """
    Compute mean and standard deviation for this dataset.
    """
    all_stats = {}
    for type_tensor in ["inputs", "outputs", "forcing"]:
        stats_dict = compute_mean_std_min_max(dataset, type_tensor)
        for feature, stats in stats_dict.items():
            # If feature was computed multiple times we keep only first occurence
            if feature not in all_stats.keys():
                all_stats[feature] = stats

    dest_file = dataset.cache_dir / "parameters_stats.pt"
    torch_save(all_stats, dest_file)
    print(f"Parameters statistics saved in {dest_file}")


def compute_time_step_stats(dataset: DatasetABC):
    random_inputs = next(iter(dataset.torch_dataloader())).inputs
    n_features = len(random_inputs.feature_names)
    sum_means = torch.zeros(n_features)
    sum_squares = torch.zeros(n_features)
    counter = 0
    if not dataset.settings.standardize:
        raise ValueError("Your dataset should be standardized.")

    for batch in tqdm(dataset.torch_dataloader(), desc="Computing diff stats"):
        # Here we assume that data are in 2 or 3 D
        inputs = batch.inputs.tensor
        outputs = batch.outputs.tensor

        in_out = torch.cat([inputs, outputs], dim=1)
        diff = in_out[:, 1:] - in_out[:, :-1]  # Substract information on time dimension
        diff = diff.flatten(1, 3)  # Flatten everybody to be (Batch, X, Features)

        counter += in_out.shape[0]  # += batch size
        sum_means += torch.nansum(diff.nanmean(dim=1), dim=0)  # (d_features)
        sum_squares += torch.nansum((diff**2).nanmean(dim=1), dim=0)  # (d_features)

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
