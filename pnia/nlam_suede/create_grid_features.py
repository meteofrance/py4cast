import numpy as np
import torch
from pnia.base import AbstractDataset
from pnia.titan_dataset import TitanParams, TitanDataset
from pnia.settings import CACHE_DIR


def prepare(dataset: AbstractDataset):
    """
    dataset: Dataset to load grid point coordinates from.
    """
    graph_dir_path = CACHE_DIR / 'neural_lam' / str(dataset)

    # -- Static grid node features --
    grid_xy = dataset.grid_info  # (2, N_x, N_y)
    grid_xy = grid_xy.flatten(1, 2).T  # (N_grid, 2)
    pos_max = torch.max(torch.abs(grid_xy))
    grid_xy = grid_xy / pos_max  # Divide by maximum coordinate

    geopotential = dataset.geopotential_info  # (N_x, N_y)
    geopotential = geopotential.flatten(0,1).unsqueeze(1) # (N_grid,1)
    gp_min = torch.min(geopotential)
    gp_max = torch.max(geopotential)
    # Rescale geopotential to [0,1]
    geopotential = (geopotential - gp_min)/(gp_max - gp_min) # (N_grid, 1)

    grid_border_mask = dataset.border_mask  # (N_x, N_y)
    grid_border_mask = grid_border_mask.flatten(0, 1).to(torch.float).unsqueeze(1) # (N_grid, 1)

    # Concatenate grid features
    grid_features = torch.cat((grid_xy, geopotential, grid_border_mask), dim=1) # (N_grid, 4)

    torch.save(grid_features, graph_dir_path / "grid_features.pt")


if __name__ == "__main__":
    # Load Dataset
    hparams = TitanParams()
    dataset = TitanDataset(hparams)
    prepare(dataset)