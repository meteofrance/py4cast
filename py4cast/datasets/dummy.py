"""
A dummy dataset for documentation and testing purposes.
Can be used as a starting point to implement your own dataset.
inputs, outputs and forcing are filled with random tensors.
"""

from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Dict, Literal, Tuple, Union

import cartopy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from py4cast.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Item,
    NamedTensor,
    Stats,
    TorchDataloaderSettings,
    collate_fn,
)
from py4cast.plots import DomainInfo
from py4cast.settings import CACHE_DIR


@dataclass
class Grid:
    border_size: int = 10
    x: int = 64  # X dimension
    y: int = 64  # Y dimension

    @cached_property
    def lat(self) -> np.array:
        x_grid, _ = (np.indices((self.x, self.y)) - 16) * 0.5
        return x_grid

    @cached_property
    def lon(self) -> np.array:
        _, y_grid = (np.indices((self.x, self.y)) + 30) * 0.5
        return y_grid

    @property
    def geopotential(self) -> np.array:
        return np.ones((self.x, self.y))

    @property
    def border_mask(self) -> np.array:
        if self.border_size > 0:
            border_mask = np.ones((self.x, self.y)).astype(bool)
            size = self.border_size
            border_mask[size:-size, size:-size] *= False
        elif self.border_size == 0:
            border_mask = np.ones((self.x, self.y)).astype(bool) * False
        else:
            raise ValueError(f"Bordersize should be positive. Get {self.border_size}")
        return border_mask

    @cached_property
    def meshgrid(self) -> np.array:
        """
        Build a meshgrid from coordinates position.
        Be careful. It's used for the graph generation and can lead to error.
        """
        return np.stack((self.lon, self.lat))

    @cached_property
    def projection(self):
        # Create projection
        return cartopy.crs.PlateCarree(central_longitude=self.lon.mean())

    @cached_property
    def grid_limits(self):
        return [  # In projection
            self.lon[0, 0],  # min x
            self.lon[-1, -1],  # max x
            self.lat[0, 0],  # min y
            self.lat[-1, -1],  # max y
        ]


@dataclass(slots=True)
class DummySettings:
    num_input_steps: int = 2  # Number of input timesteps
    num_pred_steps: int = 1  # Number of output timesteps
    standardize: bool = True

    @property
    def num_total_steps(self):
        """
        Total number of timesteps for one sample.
        """
        return self.num_input_steps + self.num_pred_steps


class DummyDataset(DatasetABC, Dataset):
    def __init__(self, grid: Grid, settings: DummySettings):
        self.grid = grid
        self.settings = settings
        self._cache_dir = CACHE_DIR / str(self)
        self.len = 10
        self.shuffle = False

    @cached_property
    def cache_dir(self):
        return self._cache_dir

    def __str__(self):
        return "DummyDataset"

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        inputs = NamedTensor(
            tensor=torch.randn(
                self.settings.num_input_steps,
                self.grid.x,
                self.grid.y,
                self.weather_dim,
            ).clamp(-3, 3),
            feature_names=self.shortnames("input_output"),
            names=["timestep", "lat", "lon", "features"],
        )
        outputs = NamedTensor(
            tensor=torch.randn(
                self.settings.num_pred_steps,
                self.grid.x,
                self.grid.y,
                self.weather_dim,
            ).clamp(-3, 3),
            feature_names=self.shortnames("input_output"),
            names=["timestep", "lat", "lon", "features"],
        )
        forcing = NamedTensor(
            tensor=torch.randn(
                self.settings.num_pred_steps,
                self.grid.x,
                self.grid.y,
                self.forcing_dim,
            ).clamp(-3, 3),
            feature_names=self.shortnames("forcing"),
            names=["timestep", "lat", "lon", "features"],
        )
        return Item(
            inputs=inputs,
            outputs=outputs,
            forcing=forcing,
        )

    def torch_dataloader(
        self, tl_settings: TorchDataloaderSettings = TorchDataloaderSettings()
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=tl_settings.batch_size,
            num_workers=tl_settings.num_workers,
            shuffle=self.shuffle,
            prefetch_factor=tl_settings.prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=tl_settings.pin_memory,
        )

    @cached_property
    def meshgrid(self) -> np.array:
        return self.grid.meshgrid

    @cached_property
    def geopotential_info(self) -> np.array:
        return self.grid.geopotential

    @cached_property
    def border_mask(self) -> np.array:
        return self.grid.border_mask

    @cached_property
    def forcing_dim(self):
        return 2

    @cached_property
    def weather_dim(self):
        return 3

    def shortnames(self, kind: Literal["forcing", "input_output", "diagnostic"]):
        """
        Define the variable names
        """
        number = 0
        if kind == "forcing":
            number = self.forcing_dim
        elif kind == "input_output":
            number = self.weather_dim
        return [f"{kind}_{str(x).zfill(2)}" for x in range(number)]

    @cached_property
    def state_weights(self):
        """
        Weights used in the loss function.
        """
        w_dict = {}
        for name in self.shortnames("input_output"):
            w_dict[name] = np.abs(np.random.randn())
        return w_dict

    @cached_property
    def units(self) -> Dict[str, int]:
        """
        Return a dictionnary with name and units
        """
        dout = {}
        for name in self.shortnames("input_output") + self.shortnames("forcing"):
            dout[name] = "FakeUnit"
        return dout

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered.
        Usefull information for plotting.
        """
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )

    @cached_property
    def dataset_info(self) -> DatasetInfo:
        """
        Return a DatasetInfo object.
        This object describes the dataset.

        Returns:
            DatasetInfo: _description_
        """
        shortnames = {
            "forcing": self.shortnames("forcing"),
            "input_output": self.shortnames("input_output"),
            "diagnostic": self.shortnames("diagnostic"),
        }
        return DatasetInfo(
            name=str(self),
            domain_info=self.domain_info,
            units=self.units,
            shortnames=shortnames,
            weather_dim=self.weather_dim,
            forcing_dim=self.forcing_dim,
            step_duration=1,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
        config_overrides: Union[Dict, None] = None,
    ) -> Tuple["DummyDataset", "DummyDataset", "DummyDataset"]:
        """
        The path is not used for DummyDataset.
        Return the train, valid and test datasets, in that order
        """
        grid = Grid()
        train_settings = DummySettings(
            num_input_steps=num_input_steps, num_pred_steps=num_pred_steps_train
        )
        val_test_settings = DummySettings(
            num_input_steps=num_input_steps, num_pred_steps=num_pred_steps_val_tests
        )
        train_dataset = DummyDataset(grid, train_settings)
        val_dataset = DummyDataset(grid, val_test_settings)
        test_dataset = DummyDataset(grid, val_test_settings)
        return train_dataset, val_dataset, test_dataset

    @cached_property
    def stats(self) -> Stats:
        """
        Read fake statistics from BytesIO files
        """
        # Generate fake stats
        d_stats = {}
        for name in self.shortnames("input_output") + self.shortnames("forcing"):
            d_stats[name] = {
                "mean": torch.tensor(0.0),
                "std": torch.tensor(1.0),
                "max": torch.tensor(3.0),
                "min": torch.tensor(-3.0),
            }
        # Save them in byteIo
        buffer = BytesIO()
        torch.save(d_stats, buffer)
        buffer.seek(0)
        # Read them
        return Stats(buffer)

    @cached_property
    def diff_stats(self) -> Stats:
        """
        Read fake statistics from BytesIO files
        """
        d_stats = {}
        for name in self.shortnames("input_output") + self.shortnames("forcing"):
            d_stats[name] = {"mean": torch.tensor(0.0), "std": torch.tensor(1.42)}
        # Save them in byteIo
        buffer = BytesIO()
        torch.save(d_stats, buffer)
        buffer.seek(0)
        # Read them
        return Stats(buffer)


if __name__ == "__main__":
    train_ds, val_ds, test_ds = DummyDataset.from_json("fakepath.json", 1, 2, 4)
    dataset_info = train_ds.dataset_info
    dataset_info.summary()
    print(dataset_info.domain_info)
    print(train_ds.__getitem__(0))
    print("Number of elment in dataset", train_ds.__len__())
