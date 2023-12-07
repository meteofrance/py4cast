import numpy as np
from torch.utils.data import DataLoader, Dataset
from pnia.base import AbstractDataset
from cyeccodes.eccodes import get_multi_messages_from_file

class TitanDataset(AbstractDataset, Dataset):
    def __init__(self) -> None:
        pass

    def __len__(self):
        pass

    @staticmethod
    def load_one_time_step():
        pass

    def __getitem__(self, index):
        pass

    @property
    def grid_info(self) -> np.array:
        pass

    @property
    def geopotential_info(self) -> np.array:
        pass
