import os
from tqdm import tqdm
import numpy as np
import torch

from pnia.base import AbstractDataset
from pnia.TitanDataset import TitanDataset, TitanParams
from pnia.settings import CACHE_DIR

def create_parameter_weights(batch_size: int = 4, n_workers: int = 10, step_length:int =3):
    """
    dataset: Dataset to compute weights for 
    batch_size : Batch size when iterating over the dataset
    step_length: Step length in hours to consider single time step (default: 3)
    n_workers: Number of workers in data loader (default: 4)
    """

    graph_dir_path = CACHE_DIR / 'neural_lam' / dataset.__str__
    graph_dir_path.mkdir(parents=True, exist_ok=True)
    static_dir_path = graph_dir_path /  "static"
    static_dir_path.mkdir(parents=True, exist_ok=True)

    hparams = TitanParams(
        split : "train",
        nb_pred_steps : step_length,
        standardize : False
    )
    dataset = TitanDataset(hparams)
    create_parameter_weights_step1(
        dataset,
        batch_size = batch_size, 
        static_dir_path=static_dir_path,
        n_workers=n_workers)

    hparams = TitanParams(
        split : "train",
        nb_pred_steps : step_length,
        standardize : True
    )
    dataset = TitanDataset(hparams)
    create_paramter_weights_step2(dataset,
        batch_size = batch_size, 
        static_dir_path=static_dir_path,
        n_workers=n_workers)

    return


def create_parameter_weights_step1(dataset: AbstractDataset, batch_size: int = 4, n_workers: int = 10, static_dir_path: Path):
    """
    dataset: Dataset to compute weights for 
    batch_size : Batch size when iterating over the dataset
    n_workers: Number of workers in data loader (default: 4)
    static_dir_path : path to save parameter weights
    """

    # Create parameter weights based on height
    # based on fig A.1 in graph cast paper
    w_par = np.zeros((len(dataset.hparams.weather_params),))
    w_dict = {'2': 1.0, '0': 0.1, '65': 0.065, '1000': 0.1, '850': 0.05, '500': 0.03}
    w_list = np.array([w_dict[par.split('_')[-2]] for par in dataset.hparams.weather_params])
    print("Saving parameter weights...")
    np.save(static_dir_path / 'parameter_weights.npy',
            w_list.astype('float32'))

    # Load dataset without any subsampling
    #ds = dataset(dataset, split="train", subsample_step=1, pred_length=63,
    #        standardize=False) # Without standardization
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False,
            num_workers=n_workers)
    # Compute mean and std.-dev. of each parameter (+ flux forcing) across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for init_batch, target_batch, _, forcing_batch in tqdm(loader):
        batch = torch.cat((init_batch, target_batch),
                dim=1) # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1,2))) # (N_batch, d_features,)
        squares.append(torch.mean(batch**2, dim=(1,2))) # (N_batch, d_features,)

        flux_batch = forcing_batch[:,:,:,0] # Flux is first index
        flux_means.append(torch.mean(flux_batch)) # (,)
        flux_squares.append(torch.mean(flux_batch**2)) # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0) # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2) # (d_features)

    flux_mean = torch.mean(torch.stack(flux_means)) # (,)
    flux_second_moment = torch.mean(torch.stack(flux_squares)) # (,)
    flux_std = torch.sqrt(flux_second_moment - flux_mean**2) # (,)
    flux_stats = torch.stack((flux_mean, flux_std))

    print("Saving mean, std.-dev, flux_stats...")
    torch.save(mean, static_dir_path / "parameter_mean.pt")
    torch.save(std, static_dir_path / "parameter_std.pt")
    torch.save(flux_stats, static_dir_path / "flux_stats.pt")


def create_paramter_weights_step2(dataset: AbstractDataset, batch_size: int = 4, n_workers: int = 10, static_dir_path: Path):
    """
    dataset: Dataset to compute weights for 
    batch_size : Batch size when iterating over the dataset
    n_workers: Number of workers in data loader (default: 4)
    static_dir_path : path to save parameter weights
    """
    step_length = dataset.hparams.nb_pred_steps
    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    #ds_standard = WeatherDataset(args.dataset, split="train", subsample_step=1,
    #        pred_length=63, standardize=True) # Re-load with standardization
    loader_standard = torch.utils.data.DataLoader(dataset, batch_size,
            shuffle=False, num_workers=n_workers)
    used_subsample_len = (65//step_length)*step_length

    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _, _ in tqdm(loader_standard):
        batch = torch.cat((init_batch, target_batch),
                dim=1) # (N_batch, N_t', N_grid, d_features)
        # Note: batch contains only 1h-steps
        stepped_batch = torch.cat([batch[:,ss_i:used_subsample_len:step_length]
            for ss_i in range(step_length)], dim=0)
        # (N_batch', N_t, N_grid, d_features), N_batch' = args.step_length*N_batch

        batch_diffs = stepped_batch[:,1:] - stepped_batch[:,:-1]
        # (N_batch', N_t-1, N_grid, d_features)

        diff_means.append(torch.mean(batch_diffs, dim=(1,2))) # (N_batch', d_features,)
        diff_squares.append(torch.mean(batch_diffs**2,
            dim=(1,2))) # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0) # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2) # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, static_dir_path / "diff_mean.pt")
    torch.save(diff_std, static_dir_path / "diff_std.pt")

if __name__ == "__main__":
    create_parameter_weights(batch_size= 4, n_workers= 10, step_length=3)