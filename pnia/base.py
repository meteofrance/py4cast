"""
Base classes defining our software components
and their interfaces
"""

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty


class AbstractDataset(ABCMeta):

    @abstractproperty
    def grid_info(self) -> np.array:
        pass

    @abstractproperty
    def geopotential_info(self) -> np.array:
        pass