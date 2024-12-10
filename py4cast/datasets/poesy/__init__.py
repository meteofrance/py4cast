import datetime as dt
import json
import time
from argparse import ArgumentParser
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
    Grid,
    GridConfig,
    NamedTensor,
    ParamConfig,
    Period,
    Sample,
    SamplePreprocSettings,
    Stats,
    Timestamps,
    TorchDataloaderSettings,
    WeatherParam,
    collate_fn,
    get_param_list,
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


def poesy_forecast_namer(date: dt.datetime, var_file_name, **kwargs):
    """
    use to find local files
    """
    return f"{date.strftime('%Y-%m-%dT%H:%M:%SZ')}_{var_file_name}_lt1-45_crop.npy"


def get_weight(level: float, level_type: str) -> float:
    if level_type == "isobaricInHpa":
        return 1.0 + (level) / (90)
    elif level_type == "heightAboveGround":
        return 2.0
    elif level_type == "surface":
        return 1.0
    else:
        raise Exception(f"unknown level_type:{level_type}")


#############################################################
#                            GRID                           #
#############################################################


def load_grid_info(grid: Grid) -> GridConfig:
    geopotential = np.load(SCRATCH_PATH / OROGRAPHY_FNAME)
    latlon = np.load(SCRATCH_PATH / LATLON_FNAME)
    full_size = geopotential.shape
    latitude = latlon[1, :, 0]
    longitude = latlon[0, 0]
    landsea_mask = np.where(geopotential > 0, 1.0, 0.0).astype(np.float32)
    return GridConfig(full_size, latitude, longitude, geopotential, landsea_mask)


#############################################################
#                            PARAMS                         #
#############################################################


def get_filepath(ds_name: str, param: WeatherParam, date: dt.datetime) -> str:
    """
    Return the filename.
    """
    var_file_name = METADATA["WEATHER_PARAMS"][param.name]["file_name"]
    return (
        SCRATCH_PATH
        / f"{date.strftime('%Y-%m-%dT%H:%M:%SZ')}_{var_file_name}_lt1-45_crop.npy"
    )


def load_param_info(name: str) -> ParamConfig:
    info = METADATA["WEATHER_PARAMS"][name]
    unit = info["unit"]
    long_name = info["long_name"]
    grid = info["grid"]
    level_type = info["level_type"]
    grib_name = None
    grib_param = None
    return ParamConfig(unit, level_type, long_name, grid, grib_name, grib_param)


def load_data(
    ds_name: str, param: WeatherParam, date: dt.datetime, term: List, member: int
) -> np.array:
    """
    date : Date of file.
    term : Position of leadtimes in file.
    """
    data_array = np.load(get_filepath(ds_name, param, date), mmap_mode="r")
    return data_array[
        param.grid.subdomain[0] : param.grid.subdomain[1],
        param.grid.subdomain[2] : param.grid.subdomain[3],
        term,
        member,
    ]


def exists(ds_name: str, param: WeatherParam, date: dt.datetime) -> bool:
    flist = get_filepath(ds_name, param, date)
    return flist.exists()


def valid_timestamp(n_inputs: int, timestamps: Timestamps):
    limits = METADATA["TERMS"]
    for t in timestamps.terms:
        if (t > dt.timedelta(limits["end"], "h")) or (
            t < dt.timedelta(limits["start"], "h")
        ):
            return False
    return True


def get_param_tensor(
    param: WeatherParam,
    stats: Stats,
    timestamps: Timestamps,
    settings: SamplePreprocSettings,
    standardize: bool,
    member: int = 1,
) -> torch.tensor:
    """
    This function load a specific parameter into a tensor
    """
    if standardize:
        name = param.parameter_short_name
        means = np.asarray(stats[name]["mean"])
        std = np.asarray(stats[name]["std"])

    array = load_data(
        settings.dataset_name, param, timestamps.datetime, timestamps.terms, member
    )

    # Extend dimension to match 3D (level dimension)
    if len(array.shape) != 4:
        array = np.expand_dims(array, axis=-1)
    array = np.transpose(array, axes=[2, 0, 1, 3])  # shape = (steps, lvl, x, y)

    if standardize:
        array = (array - means) / std

    # Define which value is considered invalid
    tensor_data = torch.from_numpy(array)

    return tensor_data


def generate_forcings(
    date: dt.datetime, output_terms: Tuple[float], grid: Grid
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


class InferSample(Sample):
    """
    Sample dedicated to inference. No outputs terms, only inputs.
    """

    def __post_init__(self):
        self.terms = self.input_terms


class PoesyDataset(DatasetABC, Dataset):
    def __init__(
        self,
        grid: Grid,
        period: Period,
        params: List[WeatherParam],
        settings: SamplePreprocSettings,
    ):
        self.grid = grid
        self.period = period
        self.params = params
        self.settings = settings
        self._cache_dir = CACHE_DIR / str(self)
        self.shuffle = self.split == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def cache_dir(self):
        return self._cache_dir

    def __str__(self) -> str:
        return f"Poesy_{self.grid.name}"

    def __len__(self):
        return len(self.sample_list)

    @cached_property
    def dataset_info(self) -> DatasetInfo:
        """
        Return a DatasetInfo object.
        This object describes the dataset.

        Returns:
            DatasetInfo: _description_
        """

        shortnames = {
            "input": self.shortnames("input"),
            "input_output": self.shortnames("input_output"),
            "output": self.shortnames("output"),
        }

        return DatasetInfo(
            name=str(self),
            domain_info=self.domain_info,
            units=self.units,
            shortnames=shortnames,
            weather_dim=self.weather_dim,
            forcing_dim=self.forcing_dim,
            step_duration=self.period.step_duration,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    @cached_property
    def sample_list(self):
        """Creates the list of samples."""
        print("Start creating samples...")
        stats = self.stats if self.settings.standardize else None

        n_inputs, n_preds, step_duration = (
            self.settings.num_input_steps,
            self.settings.num_pred_steps,
            self.period.step_duration,
        )

        sample_timesteps = [
            step_duration * step for step in range(-n_inputs + 1, n_preds + 1)
        ]
        all_timestamps = []
        for date in tqdm.tqdm(self.period.date_list):
            for term in self.period.terms_list:
                t0 = date + dt.timedelta(hours=term)
                validity_times = [
                    t0 + dt.timedelta(hours=ts) for ts in sample_timesteps
                ]
                terms = [dt.timedelta(t + term) for t in sample_timesteps]

                timestamps = Timestamps(
                    datetime=date,
                    terms=terms,
                    validity_times=validity_times,
                )
                if valid_timestamp(n_inputs, timestamps):
                    all_timestamps.append(timestamps)

        samples = []
        for ts in all_timestamps:
            for member in self.settings.members:
                sample = Sample(
                    timestamps,
                    n_inputs,
                    self.settings,
                    self.params,
                    stats,
                    self.grid,
                    self.member,
                )
                if sample.is_valid():
                    samples.append(sample)

        print(f"--> All {len(samples)} {self.period.name} samples are now defined")
        return samples

    @cached_property
    def dataset_extra_statics(self):
        """
        We add the LandSea Mask to the statics.
        """
        return [
            NamedTensor(
                feature_names=["LandSeaMask"],
                tensor=torch.from_numpy(self.grid.landsea_mask)
                .type(torch.float32)
                .unsqueeze(2),
                names=["lat", "lon", "features"],
            )
        ]

    @cached_property
    def forcing_dim(self) -> int:
        """
        Return the number of forcings.
        """
        res = 4  # For date
        res += 1  # For solar forcing

        for param in self.params:
            if param.kind == "input":
                res += 1
        return res

    @cached_property
    def weather_dim(self) -> int:
        """
        Return the dimension of pronostic variable.
        """
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += 1
        return res

    @cached_property
    def diagnostic_dim(self):
        """
        Return dimensions of output variable only
        Not used yet
        """
        res = 0
        for param in self.params:
            if param.kind == "output":
                res += 1
        return res

    def __getitem__(self, index):
        """
        Return an item from an index of the sample_list
        """
        sample = self.sample_list[index]
        item = sample.load()
        return item

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple["PoesyDataset", "PoesyDataset", "PoesyDataset"]:
        """
        Return 3 PoesyDataset.
        Override configuration file if needed.
        """
        with open(fname, "r") as fp:
            conf = json.load(fp)
            if config_override is not None:
                conf = merge_dicts(conf, config_override)
        conf["grid"]["load_grid_info_func"] = load_grid_info
        grid = Grid(**conf["grid"])
        param_list = get_param_list(conf, grid, load_param_info, get_weight)

        train_period = Period(**conf["periods"]["train"], name="train")
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        test_period = Period(**conf["periods"]["test"], name="test")
        train_ds = PoesyDataset(
            grid,
            train_period,
            param_list,
            SamplePreprocSettings(
                dataset_name=fname.stem,
                num_pred_steps=num_pred_steps_train,
                num_input_steps=num_input_steps,
                members=conf["members"],
                **conf["settings"],
            ),
        )
        valid_ds = PoesyDataset(
            grid,
            valid_period,
            param_list,
            SamplePreprocSettings(
                dataset_name=fname.stem,
                num_pred_steps=num_pred_steps_val_test,
                num_input_steps=num_input_steps,
                members=conf["members"],
                **conf["settings"],
            ),
        )
        test_ds = PoesyDataset(
            grid,
            test_period,
            param_list,
            SamplePreprocSettings(
                dataset_name=fname.stem,
                num_pred_steps=num_pred_steps_val_test,
                num_input_steps=num_input_steps,
                members=conf["members"],
                **conf["settings"],
            ),
        )
        return train_ds, valid_ds, test_ds

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

    @property
    def meshgrid(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (X, Y) values
        """
        return self.grid.meshgrid

    @property
    def geopotential_info(self) -> np.array:
        """
        array of shape (num_lat, num_lon)
        with geopotential value for each datapoint
        """
        return self.grid.geopotential

    @property
    def limited_area(self) -> bool:
        """
        Returns True if the dataset is
        compatible with Limited area models
        """
        return True

    @property
    def border_mask(self) -> np.array:
        return self.grid.border_mask

    @property
    def split(self) -> Literal["train", "valid", "test"]:
        return self.period.name

    @cached_property
    def units(self) -> Dict[str, int]:
        """
        Return a dictionnary with name and units
        """
        return {p.parameter_short_name: p.unit for p in self.params}

    def shortnames(
        self,
        kind: List[Literal["input", "output", "input_output"]] = [
            "input",
            "output",
            "input_output",
        ],
    ) -> List[str]:
        """
        Return the name of the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return [p.parameter_short_name for p in self.params if p.kind == kind]

    @cached_property
    def state_weights(self):
        """
        Weights used in the loss function.
        """
        kinds = ["output", "input_output"]
        return {
            p.parameter_short_name: p.state_weight
            for p in self.params
            if p.kind in kinds
        }

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered.
        Usefull information for plotting.
        """
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )

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
            SamplePreprocSettings(
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
