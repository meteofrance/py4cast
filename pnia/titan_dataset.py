from pathlib import Path
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from pnia.base import AbstractDataset
from cyeccodes.eccodes import get_multi_messages_from_file

class TitanDataset(AbstractDataset, Dataset):
    def __init__(self) -> None:
        self.ROOT_DIR = Path("/scratch/shared/Titan/")

    def __len__(self):
        pass

    @staticmethod
    def load_one_time_step():
        pass

    def __getitem__(self, index):
        pass

    @property
    def grid_info(self) -> np.array:
        conf_ds = xr.load_dataset(self.ROOT_DIR / "conf.grib", engine='cfgrib')
        print(conf_ds)

    @property
    def geopotential_info(self) -> np.array:
        pass
