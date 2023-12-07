from pathlib import Path
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from pnia.base import AbstractDataset

class TitanDataset(AbstractDataset, Dataset):
    def __init__(self) -> None:
        self.ROOT_DIR = Path("/scratch/shared/Titan/")

    @property
    def grid_info(self) -> np.array:
        conf_ds = xr.load_dataset(self.ROOT_DIR / "conf.grib", engine='cfgrib')
        print(conf_ds)

    @property
    def geopotential_info(self) -> np.array:
        pass
