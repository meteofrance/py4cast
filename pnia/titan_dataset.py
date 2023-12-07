import numpy as np
from torch.utils.data import DataLoader, Dataset

class TitanDataset(Dataset):
    def __init__(self) -> None:
        pass

    @property
    def grid_info(self) -> np.array:
        pass

    @property
    def geopotential_info(self) -> np.array:
        pass
