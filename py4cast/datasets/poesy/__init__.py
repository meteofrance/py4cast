import datetime as dt
import json
import time
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from py4cast.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Item,
    NamedTensor,
    Period,
    TorchDataloaderSettings,
    collate_fn,
    get_param_list,
)

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    Param,
    ParamConfig,
    Settings,
    Stats
)

from py4cast.datasets.poesy.settings import (
    LATLON_FNAME,
    METADATA,
    OROGRAPHY_FNAME,
    SCRATCH_PATH,
)
from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing
from py4cast.plots import DomainInfo
from py4cast.settings import CACHE_DIR
from py4cast.utils import merge_dicts


class PoesyAccessor(DataAccessor):

    def get_dataset_path(name:str, grid: Grid) -> Path:
        return CACHE_DIR / str(name)

    def get_weight_per_level(
    level: float,
    level_type: Literal['isobaricInhPa','heightAboveGround','surface','meanSea']) -> float:
        if level_type == "isobaricInHpa":
            return 1.0 + (level) / (90)
        elif level_type == "heightAboveGround":
            return 2.0
        elif level_type == "surface":
            return 1.0
        else:
            raise Exception(f"unknown level_type:{level_type}")

    def load_grid_info(grid: Grid) -> GridConfig:
        geopotential = np.load(SCRATCH_PATH / OROGRAPHY_FNAME)
        latlon = np.load(SCRATCH_PATH / LATLON_FNAME)
        full_size = geopotential.shape
        latitude = latlon[1, :, 0]
        longitude = latlon[0, 0]
        landsea_mask = np.where(geopotential > 0, 1.0, 0.0).astype(np.float32)
        return GridConfig(full_size, latitude, longitude, geopotential, landsea_mask)

    def load_param_info(name: str) -> ParamConfig:
        info = METADATA["WEATHER_PARAMS"][name]
        unit = info["unit"]
        long_name = info["long_name"]
        grid = info["grid"]
        level_type = info["level_type"]
        grib_name = None
        grib_param = None
        return ParamConfig(unit, level_type, long_name, grid, grib_name, grib_param)
    
    def get_grid_coords(param: Param) -> List[float]:
        raise NotImplementedError("Poesy does not require get_grid_coords")

    def get_filepath(ds_name: str, param: Param, date: dt.datetime, file_format: Optional[str]='npy') -> str:
        """
        Return the filename.
        """
        var_file_name = METADATA["WEATHER_PARAMS"][param.name]["file_name"]
        return (
            SCRATCH_PATH
            / f"{date.strftime('%Y-%m-%dT%H:%M:%SZ')}_{var_file_name}_lt1-45_crop.npy"
        )

    def load_data_from_disk(
        self,
        dataset_name: str, # name of the dataset or dataset version
        param: Param, # specific parameter (2D field associated to a grid)
        date: dt.datetime, # specific timestamp at which to load the field
        members: Tuple[int], # optional members id. when dealing with ensembles
        file_format: Literal["npy", "grib"] = "npy" # format of the base file on disk
    ) -> np.array:

        data_array = np.load(get_filepath(ds_name, param, date), mmap_mode="r")
        return data_array[
            param.grid.subdomain[0] : param.grid.subdomain[1],
            param.grid.subdomain[2] : param.grid.subdomain[3],
            term,
            members,
        ]

@dataclass(slots=True)
class Sample:
    """Describes a sample"""

    member: int
    date: dt.datetime
    settings: Settings
    input_terms: Tuple[float]
    output_terms: Tuple[float]
    stats: Stats
    grid: Grid
    params: List[Param]

    # Term wrt to the date {date}. Gives validity
    terms: Tuple[float] = field(init=False)

    def __post_init__(self):
        self.terms = self.input_terms + self.output_terms

    def is_valid(self) -> bool:
        """
        Check that all the files necessary for this samples exists.

        Args:
            param_list (List): List of parameters
        Returns:
            Boolean:  Whether the sample exists or not
        """
        for param in self.params:
            if not exists(self.settings.dataset_name, param, self.date):
                return False

        return True

    def generate_forcings(
        self, date: dt.datetime, output_terms: Tuple[float], grid: Grid
    ) -> List[NamedTensor]:
        """
        Generate all the forcing in this function.
        Return a list of NamedTensor.
        """
        # Datetime Forcing
        datetime_forcing = get_year_hour_forcing(date, output_terms).type(torch.float32)

        # Solar forcing, dim : [num_pred_steps, Lat, Lon, feature = 1]
        solar_forcing = generate_toa_radiation_forcing(
            grid.lat, grid.lon, date, output_terms
        ).type(torch.float32)

        lforcings = [
            NamedTensor(
                feature_names=[
                    "cos_hour",
                    "sin_hour",
                ],  # doy : day_of_year
                tensor=datetime_forcing[:, :2],
                names=["timestep", "features"],
            ),
            NamedTensor(
                feature_names=[
                    "cos_doy",
                    "sin_doy",
                ],  # doy : day_of_year
                tensor=datetime_forcing[:, 2:],
                names=["timestep", "features"],
            ),
            NamedTensor(
                feature_names=[
                    "toa_radiation",
                ],
                tensor=solar_forcing,
                names=["timestep", "lat", "lon", "features"],
            ),
        ]

        return lforcings

    def load(self) -> Item:
        """
        Return inputs, outputs, forcings as tensors concatenated into a Item.
        """
        linputs = []
        loutputs = []

        # Reading parameters from files
        for param in self.params:
            state_kwargs = {
                "feature_names": [param.parameter_short_name],
                "names": ["timestep", "lat", "lon", "features"],
            }
            try:
                if param.kind == "input_output":
                    # Search data for date sample.date and terms sample.terms
                    tensor = get_param_tensor(
                        param=param,
                        stats=self.stats,
                        date=self.date,
                        settings=self.settings,
                        terms=self.terms,
                        standardize=self.settings.standardize,
                        member=self.member,
                    )
                    state_kwargs["names"][0] = "timestep"
                    # Save outputs
                    tmp_state = NamedTensor(
                        tensor=tensor[self.settings.num_input_steps :],
                        **deepcopy(state_kwargs),
                    )
                    loutputs.append(tmp_state)
                    # Save inputs
                    tmp_state = NamedTensor(
                        tensor=tensor[: self.settings.num_input_steps],
                        **deepcopy(state_kwargs),
                    )
                    linputs.append(tmp_state)

            except KeyError as e:
                print(f"Error for param {param}")
                raise e

        # Get forcings
        lforcings = self.generate_forcings(
            date=self.date, output_terms=self.output_terms, grid=self.grid
        )
        for lforcing in lforcings:
            lforcing.unsqueeze_and_expand_from_(linputs[0])

        return Item(
            inputs=NamedTensor.concat(linputs),
            outputs=NamedTensor.concat(loutputs),
            forcing=NamedTensor.concat(lforcings),
        )


class InferSample(Sample):
    """
    Sample dedicated to inference. No outputs terms, only inputs.
    """

    def __post_init__(self):
        self.terms = self.input_terms


class PoesyDataset(DatasetABC, Dataset):
    def __init__(
        self, name, grid: Grid, period: Period, params: List[Param], settings: Settings
    ):
        self.name = name
        self.grid = grid
        self.period = period
        self.params = params
        self.settings = settings
        self._cache_dir = CACHE_DIR / str(self)
        self.shuffle = self.split == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    @cached_property
    def sample_list(self):
        """
        Create a list of sample from information
        """
        print("Start forming samples")
        terms = list(
            np.arange(
                METADATA["TERMS"]["start"],
                METADATA["TERMS"]["end"],
                METADATA["TERMS"]["timestep"],
            )
        )
        num_total_steps = self.settings.num_input_steps + self.settings.num_pred_steps
        sample_by_date = len(terms) // num_total_steps

        samples = []
        number = 0

        for date in self.period.date_list:
            for member in self.settings.members:
                for sample in range(0, sample_by_date):
                    input_terms = terms[
                        sample * num_total_steps : sample * num_total_steps
                        + self.settings.num_input_steps
                    ]
                    output_terms = terms[
                        sample * num_total_steps
                        + self.settings.num_input_steps : sample * num_total_steps
                        + self.settings.num_input_steps
                        + self.settings.num_pred_steps
                    ]
                    samp = Sample(
                        date=date,
                        member=member,
                        input_terms=input_terms,
                        output_terms=output_terms,
                        settings=self.settings,
                        stats=self.stats,
                        grid=self.grid,
                        params=self.params,
                    )

                    if samp.is_valid():
                        samples.append(samp)
                        number += 1

        print(f"All {len(samples)} samples are now defined")
        return samples

    @classmethod
    def prepare(cls, path_config: Path):
        print("--> Preparing Poesy Dataset...")

        print("Load train dataset configuration...")
        with open(path_config, "r") as fp:
            conf = json.load(fp)

        print("Computing stats on each parameter...")
        conf["settings"]["standardize"] = True
        train_ds, _, _ = PoesyDataset.from_json(
            fname=path_config,
            num_input_steps=2,
            num_pred_steps_train=1,
            num_pred_steps_val_test=1,
        )
        train_ds.compute_parameters_stats()

        print("Computing time stats on each parameters, between 2 timesteps...")
        conf["settings"]["standardize"] = True
        train_ds, _, _ = PoesyDataset.from_json(
            fname=path_config,
            num_input_steps=2,
            num_pred_steps_train=1,
            num_pred_steps_val_test=1,
        )
        train_ds.compute_time_step_stats()

        return train_ds


class InferPoesyDataset(PoesyDataset):
    """
    Inherite from the PoesyDataset class.
    This class is used for inference, the class overrides methods sample_list and from_json.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def sample_list(self):
        """
        Create a list of sample from information.
        Outputs terms are computed from the number of prediction steps in argument.
        """
        print("Start forming samples")
        terms = list(
            np.arange(
                METADATA["TERMS"]["start"],
                METADATA["TERMS"]["end"],
                METADATA["TERMS"]["timestep"],
            )
        )

        num_total_steps = self.settings.num_input_steps + self.settings.num_pred_steps
        sample_by_date = len(terms) // num_total_steps
        samples = []
        number = 0
        for date in self.period.date_list:
            for member in self.settings.members:
                for sample in range(0, sample_by_date):
                    input_terms = terms[
                        sample * num_total_steps : sample * num_total_steps
                        + self.settings.num_input_steps
                    ]

                    output_terms = [
                        input_terms[-1] + METADATA["TERMS"]["timestep"] * (step + 1)
                        for step in range(self.settings.num_pred_steps)
                    ]

                    samp = InferSample(
                        date=date,
                        member=member,
                        settings=self.settings,
                        input_terms=input_terms,
                        output_terms=output_terms,
                    )

                    if samp.is_valid():
                        samples.append(samp)
                        number += 1
        print(f"All {len(samples)} samples are now defined")

        return samples

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple[None, None, "InferPoesyDataset"]:
        """
        Return 1 InferPoesyDataset.
        Override configuration file if needed.
        """
        with open(fname, "r") as fp:
            conf = json.load(fp)
            if config_override is not None:
                conf = merge_dicts(conf, config_override)
                print(conf["periods"]["test"])
        conf["grid"]["load_grid_info_func"] = load_grid_info
        grid = Grid(**conf["grid"])
        param_list = get_param_list(conf, grid, load_param_info, get_weight)
        inference_period = Period(**conf["periods"]["test"], name="infer")

        ds = InferPoesyDataset(
            grid,
            inference_period,
            param_list,
            Settings(
                num_pred_steps=0,
                num_input_steps=num_input_steps,
                members=conf["members"],
                **conf["settings"],
            ),
        )

        return None, None, ds


if __name__ == "__main__":
    path_config = "config/datasets/poesy.json"

    parser = ArgumentParser(description="Prepare Poesy dataset and test loading speed.")
    parser.add_argument(
        "--path_config",
        default=path_config,
        type=Path,
        help="Configuration file for the dataset.",
    )
    parser.add_argument(
        "--n_iter",
        default=10,
        type=int,
        help="Number of samples to test loading speed.",
    )
    args = parser.parse_args()

    PoesyDataset.prepare(args.path_config)

    print("Dataset info : ")
    train_ds, _, _ = PoesyDataset.from_json(args.path_config, 2, 3, 3)
    train_ds.dataset_info.summary()

    print("Test __get_item__")
    print("Len dataset : ", len(train_ds))

    print("First Item description :")
    data_iter = iter(train_ds.torch_dataloader())
    print(next(data_iter))

    print("Speed test:")
    start_time = time.time()
    for i in tqdm.trange(args.n_iter, desc="Loading samples"):
        _ = next(data_iter)
    delta = time.time() - start_time
    speed = args.n_iter / delta
    print(f"Loading speed: {round(speed, 3)} sample(s)/sec")
