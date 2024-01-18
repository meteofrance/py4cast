from pathlib import Path

import numpy as np
import torch
from pnia.datasets.base import AbstractDataset
from pnia.settings import CACHE_DIR

from tqdm import tqdm


def compute_parameters_stats(dataset: AbstractDataset, static_dir_path: Path = None):
    """Compute mean and std of parameters and stats of flux."""

    # Create parameter weights based on height
    # based on fig A.1 in graph cast paper
    # w_par = np.zeros((len(dataset.hparams.weather_params),))
    # TODO : WTF les poids selon les niveaux, à adapter ?
    # A migrer dans le dataset  ?

    # Les poids originaux {"2": 1.0, "0": 0.1, "65": 0.065, "1000": 0.1, "850": 0.05, "500": 0.03}
    w_dict = {"2": 1.0, "10": 0.5}
    w_list = np.array(
        [w_dict[str(param.levels)] for param in dataset.hp.weather_params]
    )
    print("Saving parameter level weights...")
    np.save(static_dir_path / "parameter_weights.npy", w_list.astype("float32"))

    print("Computing mean and std.-dev. for parameters across training set...")
    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for init_batch, target_batch, _, forcing_batch in tqdm(dataset.loader):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)

        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(torch.mean(batch**2, dim=(1, 2)))  # (N_batch, d_features,)

        flux_batch = forcing_batch[:, :, :, 0]  # Flux is first index
        flux_means.append(torch.mean(flux_batch))  # (,)
        flux_squares.append(torch.mean(flux_batch**2))  # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    flux_mean = torch.mean(torch.stack(flux_means))  # (,)
    flux_second_moment = torch.mean(torch.stack(flux_squares))  # (,)
    flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
    flux_stats = torch.stack((flux_mean, flux_std))

    print("Saving mean, std.-dev, flux_stats...")
    torch.save(mean, static_dir_path / "parameter_mean.pt")
    torch.save(std, static_dir_path / "parameter_std.pt")
    torch.save(flux_stats, static_dir_path / "flux_stats.pt")


def compute_timestep_stats(dataset: AbstractDataset, static_dir_path: Path = None):
    """Computes stats on the difference between 2 timesteps."""

    if dataset.timestep != 1:
        # Pour la version timestep > 1 : s'inspirer de ce que font les suedois
        raise NotImplementedError("Case with step_len > 1 not implemented")

    print("Computing mean and std.-dev. for one-step differences...")

    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _, _ in tqdm(dataset.loader):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (Nbatch, Nt, Ngrid, d_features)

        batch_diffs = batch[:, 1:] - batch[:, :-1]  # (Nbatch, N_t-1, Ngrid, d_features)

        diff_means.append(torch.mean(batch_diffs, dim=(1, 2)))  # (Nbatch, d_features)
        diff_squares.append(
            torch.mean(batch_diffs**2, dim=(1, 2))
        )  # (Nbatch, d_features)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, static_dir_path / "diff_mean.pt")
    torch.save(diff_std, static_dir_path / "diff_std.pt")


def create_parameter_weights(dataset: AbstractDataset):

    assert hparams.split == "train", "Dataset must be a training set"

    cache_dir_path = CACHE_DIR / "neural_lam" / str(dataset)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    dataset.hp.standardize = False
    print("dataset.hp.standardize : ", dataset.hp.standardize)
    compute_parameters_stats(dataset, static_dir_path=cache_dir_path)

    dataset.hp.standardize = True
    print("dataset.hp.standardize : ", dataset.hp.standardize)
    compute_timestep_stats(dataset, static_dir_path=cache_dir_path)


if __name__ == "__main__":
    from pnia.datasets.titan.dataset import TitanDataset, TitanHyperParams
    from argparse_dataclass import ArgumentParser

    parser = ArgumentParser(TitanHyperParams)
    hparams = parser.parse_args()
    print("hparams : ", hparams)
    dataset = TitanDataset(hparams)
    create_parameter_weights(dataset)
