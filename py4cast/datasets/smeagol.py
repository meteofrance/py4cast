import datetime as dt
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import cartopy
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from py4cast.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Item,
    NamedTensor,
    TorchDataloaderSettings,
    collate_fn,
)
from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing
from py4cast.plots import DomainInfo
from py4cast.utils import merge_dicts

# torch.set_num_threads(8)
SCRATCH_PATH = Path(os.environ.get("PY4CAST_SMEAGOL_PATH", "/scratch/shared/smeagol"))


# Copy from smeagol
def constant_fname(domain, model, geometry):
    return f"{domain}/{model}_{geometry}/constant/Mesh_and_SurfGeo.nc"


def smeagol_forecast_namer(
    date: dt.datetime, member: int, var, model, geometry, domain, **kwargs
):
    """
    use to find local files
    """
    return f"{domain}/{model}_{geometry}/{date.strftime('%Y%m%dH%H')}/mb_{str(member).zfill(3)}_{var}.nc"


# Fin des copie de smeagol


def get_weight(level: int, kind: str):
    if kind == "hPa":
        return 1 + (level) / (90)
    elif kind == "m":
        return 2
    else:
        raise Exception(f"unknown kind:{kind}, must be hPa or m right now")


@dataclass(slots=True)
class Period:
    start: dt.datetime
    end: dt.datetime
    step: int  # In hours
    name: str

    def __init__(self, start: int, end: int, step: int, name: str):
        self.start = dt.datetime.strptime(str(start), "%Y%m%d%H")
        self.end = dt.datetime.strptime(str(end), "%Y%m%d%H")
        self.step = step
        self.name = name

    @property
    def date_list(self):
        return pd.date_range(
            start=self.start, end=self.end, freq=f"{self.step}H"
        ).to_pydatetime()


@dataclass
class Grid:
    domain: str
    model: str
    geometry: str = "franmgsp32"
    border_size: int = 0
    # Subgrid selection. If (0,0,0,0) the whole grid is kept.
    subgrid: Tuple[int] = (0, 0, 0, 0)
    x: int = field(init=False)  # X dimension
    y: int = field(init=False)  # Y dimension

    def __post_init__(self):
        ds = xr.open_dataset(self.grid_name)
        # lat = ds.lat.values
        x, y = ds.lat.shape
        # Setting correct subgrid if no subgrid is selected.
        if self.subgrid == (0, 0, 0, 0):
            self.subgrid = (0, x, 0, y)

        self.x = self.subgrid[1] - self.subgrid[0]
        self.y = self.subgrid[3] - self.subgrid[2]

    @property
    def grid_name(self):
        return (
            SCRATCH_PATH / "nc" / constant_fname(self.domain, self.model, self.geometry)
        )

    @cached_property
    def lat(self) -> np.array:
        ds = xr.open_dataset(self.grid_name)
        return ds.lat.values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @cached_property
    def lon(self) -> np.array:
        ds = xr.open_dataset(self.grid_name)
        return ds.lon.values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @property
    def geopotential(self) -> np.array:
        ds = xr.open_dataset(self.grid_name)
        return ds["SURFGEOPOTENTIEL"].values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @property
    def landsea_mask(self) -> np.array:
        ds = xr.open_dataset(self.grid_name)
        return ds["LandSeaMask"].values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

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

    @property
    def N_grid(self) -> int:
        return self.x * self.y

    @cached_property
    def grid_limits(self):

        ds = xr.open_dataset(self.grid_name)
        grid_limits = [  # In projection
            ds.x[self.subgrid[0]].values,  # min x
            ds.x[self.subgrid[1] - 1].values,  # max x
            ds.y[self.subgrid[2]].values,  # min y
            ds.y[self.subgrid[3] - 1].values,  # max y
        ]
        return grid_limits

    @cached_property
    def meshgrid(self) -> np.array:
        """
        Build a meshgrid from coordinates position.
        """
        ds = xr.open_dataset(self.grid_name)
        x = ds.x[self.subgrid[0] : self.subgrid[1]]
        y = ds.y[self.subgrid[2] : self.subgrid[3]]
        xx, yy = np.meshgrid(x, y)
        return np.asarray([xx, yy])

    @cached_property
    def projection(self):
        # Create projection
        return cartopy.crs.LambertConformal(central_longitude=2, central_latitude=46.7)


@dataclass(slots=True)
class Param:
    name: str
    shortname: str  # To be read in nc File ?
    levels: Tuple[int]
    grid: Grid  # Parameter grid.
    # It is not necessarly the same as the model grid.
    # Function which can return the filenames.
    # It should accept member and date as argument (as well as term).
    fnamer: Callable[[], [str]]
    level_type: str = "hPa"  # To be read in nc file ?
    # Defini le statut du parametre. Les forcage sont seulement en input.
    # Les variables diagnostiques seulement en output.
    kind: Literal["input", "output", "input_output"] = "input_output"
    unit: str = "FakeUnit"  # To be read in nc FIle  ?
    ndims: int = 2

    @property
    def number(self) -> int:
        """
        Get the number of parameters.
        """
        return len(self.levels)

    @property
    def state_weights(self) -> list:
        return [get_weight(level, self.level_type) for level in self.levels]

    @property
    def parameter_name(self) -> list:
        return [f"{self.name}_{level}" for level in self.levels]

    @property
    def parameter_short_name(self) -> list:
        return [f"{self.shortname}_{level}" for level in self.levels]

    @property
    def units(self) -> list:
        """
        For a given variable, the unit is the
        same accross all levels.
        """
        return [self.unit for _ in self.levels]

    def filename(self, member: int, date: dt.datetime, term: float) -> str:
        """
        Return the filename. Even if term is not used for this example it could be (e.g. for radars).
        """
        return SCRATCH_PATH / "nc" / self.fnamer(date=date, member=member, term=term)

    def load_data(self, member: int, date: dt.datetime, terms: List):
        flist = []
        for term in terms:
            flist.append(self.filename(member=member, date=date, term=term))
        files = list(set(flist))
        # TODO : change it in order to be more generic. Here we postpone that there is only one file.
        # However it may not be the case (e.g when speaking of radars multiple files could be used).
        ds = xr.open_dataset(files[0], decode_times=False)
        return ds

    def exist(self, member: int, date: dt.datetime, terms: List):
        flist = []
        for term in terms:
            flist.append(self.filename(member=member, date=date, term=term))
        files = list(set(flist))
        # TODO: change it in order to be more generic
        return files[0].exists()


@dataclass(slots=True)
class SmeagolSettings:
    term: dict  # Terms used in this configuration. Should be present in nc files.
    num_input_steps: int  # = 2  # Number of input timesteps
    num_output_steps: int  # = 1  # Number of output timesteps (= 0 for inference)
    num_inference_pred_steps: int = 0  # 0 in training config ; else used to provide future information about forcings
    standardize: bool = True  # Do we need to standardize our data ?
    members: Tuple[int] = (
        0,
    )  # Number of member used in this configuration. Each member is independant.

    @property
    def num_total_steps(self):
        """
        Total number of timesteps
        for one sample.
        """
        # Nb of step in on sample
        return self.num_input_steps + self.num_output_steps


@dataclass(slots=True)
class Sample:
    # Describe a sample
    member: int
    date: dt.datetime
    input_terms: Tuple[float]
    output_terms: Tuple[float]
    # Term par rapport à la date {date}. Donne la validite
    terms: Tuple[float] = field(init=False)

    def __post_init__(self):
        self.terms = self.input_terms + self.output_terms

    def __repr__(self):
        return f"member : {self.member}, date {self.date}, terms {self.terms}"

    def is_valid(self, param_list: List) -> bool:
        """
        Check that all the files necessary for this samples exists.

        Args:
            param_list (List): List of parameters
        Returns:
            Boolean:  Whether the sample exist or not
        """
        for param in param_list:
            if not param.exist(self.member, self.date, self.terms):
                return False
            else:
                return True


class InferSample(Sample):
    """
    Sample dedicated to inference. No outputs terms, only inputs.
    """

    def __post_init__(self):
        self.terms = self.input_terms


class SmeagolDataset(DatasetABC, Dataset):
    def __init__(
        self, grid: Grid, period: Period, params: List[Param], settings: SmeagolSettings
    ):
        self.grid = grid
        self.period = period
        self.params = params
        self.settings = settings
        self._cache_dir = SCRATCH_PATH / str(self)
        self.shuffle = self.period.name == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.step_duration = self.settings.term["timestep"]

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
            step_duration=self.step_duration,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    @cached_property
    def sample_list(self):
        """
        Create a list of sample from information
        """
        terms = list(
            np.arange(
                self.settings.term["start"],
                self.settings.term["end"],
                self.settings.term["timestep"],
            )
        )
        sample_by_date = len(terms) // self.settings.num_total_steps
        samples = []
        number = 0
        for date in self.period.date_list:
            for member in self.settings.members:
                for sample in range(0, sample_by_date):
                    input_terms = terms[
                        sample
                        * self.settings.num_total_steps : sample
                        * self.settings.num_total_steps
                        + self.settings.num_input_steps
                    ]
                    output_terms = terms[
                        sample * self.settings.num_total_steps
                        + self.settings.num_input_steps : sample
                        * self.settings.num_total_steps
                        + self.settings.num_input_steps
                        + self.settings.num_output_steps
                    ]
                    samp = Sample(
                        member=member,
                        date=date,
                        input_terms=input_terms,
                        output_terms=output_terms,
                    )
                    if samp.is_valid(self.params):
                        samples.append(samp)
                        number += 1
        print("All samples are now defined")
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

    def __len__(self):
        length = len(self.sample_list)
        if length < 1:
            raise ValueError("No valid sample in the dataset.")
        return length

    @cached_property
    def forcing_dim(self) -> int:
        """
        Return the number of forcings.
        """
        res = 4  # For date
        res += 1  # For solar forcing
        for param in self.params:
            if param.kind == "input":
                res += param.number
        return res

    @cached_property
    def weather_dim(self) -> int:
        """
        Return the dimension of pronostic variable.
        """
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += param.number
        return res

    def test_sample(self, index):
        """
        Test if the sample is present.
        To do so, check that inputs term are present for the variable.
        Raise an error if something went wrong (for example required term does not exists).
        """
        sample = self.sample_list[index]

        # Reading parameters from files
        for param in self.params:
            ds = (
                param.load_data(sample.member, sample.date, sample.terms)
                .sel(X=range(self.grid.subgrid[0], self.grid.subgrid[1]))
                .sel(Y=range(self.grid.subgrid[2], self.grid.subgrid[3]))
            )
            if "level" in ds.coords:
                ds = ds.sel(level=param.levels)
            # Read inputs. Separate forcing field from I/O

            _ = ds[param.name].sel(step=sample.input_terms)

    def __getitem__(self, index):

        # TODO : here we should directly build a single NamedTensor per inputs, outputs and forcing attribute

        sample = self.sample_list[index]

        # Datetime Forcing
        datetime_forcing = get_year_hour_forcing(sample.date, sample.output_terms).type(
            torch.float32
        )

        # Solar forcing, dim : [num_pred_steps, Lat, Lon, feature = 1]
        solar_forcing = generate_toa_radiation_forcing(
            self.grid.lat, self.grid.lon, sample.date, sample.output_terms
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
        linputs = []
        loutputs = []

        # Reading parameters from files
        for param in self.params:
            ds = (
                param.load_data(sample.member, sample.date, sample.terms)
                .sel(X=range(self.grid.subgrid[0], self.grid.subgrid[1]))
                .sel(Y=range(self.grid.subgrid[2], self.grid.subgrid[3]))
            )
            if "level" in ds.coords:
                ds = ds.sel(level=param.levels)
            # Read inputs. Separate forcing field from I/O
            try:
                if self.settings.standardize:
                    means = np.asarray(
                        [
                            self.stats[name]["mean"]
                            for name in param.parameter_short_name
                        ]
                    )

                    std = np.asarray(
                        [self.stats[name]["std"] for name in param.parameter_short_name]
                    )

                if param.kind == "input_output":
                    tmp_in = ds[param.name].sel(step=sample.input_terms).values
                    # Extend dimension to match 3D (its a 3D with one dimension in the third one)
                    if len(tmp_in.shape) != 4:
                        tmp_in = np.expand_dims(tmp_in, axis=1)
                    tmp_in = np.transpose(tmp_in, axes=[0, 2, 3, 1])
                    if self.settings.standardize:
                        tmp_in = (tmp_in - means) / std

                    # Define the state to append.
                    tmp_state = NamedTensor(
                        tensor=torch.from_numpy(tmp_in),
                        feature_names=param.parameter_short_name,
                        names=["timestep", "lat", "lon", "features"],
                    )
                    linputs.append(tmp_state)

                    # Load outputs
                    # Inference
                    if self.settings.num_inference_pred_steps:
                        tensor_data = torch.empty(
                            (
                                self.settings.num_inference_pred_steps,
                                *tmp_state.tensor.shape[1:],
                            )
                        )

                    # Training
                    else:
                        tmp_out = ds[param.name].sel(step=sample.output_terms).values
                        if len(tmp_out.shape) != 4:
                            tmp_out = np.expand_dims(tmp_out, axis=1)
                        tmp_out = np.transpose(tmp_out, axes=[0, 2, 3, 1])
                        if self.settings.standardize:
                            tmp_out = (tmp_out - means) / std
                        tensor_data = torch.from_numpy(tmp_out)

                    tmp_state = NamedTensor(
                        tensor=tensor_data,
                        feature_names=param.parameter_short_name,
                        names=["timestep", "lat", "lon", "features"],
                    )
                    loutputs.append(tmp_state)

                # Read outputs.
                if param.kind == "ouput":

                    tmp_out = ds[param.name].sel(step=sample.output_terms).values
                    if len(tmp_out.shape) != 4:
                        tmp_out = np.expand_dims(tmp_out, axis=1)
                    tmp_out = np.transpose(tmp_out, axes=[0, 2, 3, 1])
                    if self.settings.standardize:
                        tmp_out = (tmp_out - means) / std
                    tensor_data = torch.from_numpy(tmp_out)
                    tmp_state = NamedTensor(
                        tensor=tensor_data,
                        feature_names=param.parameter_short_name,
                        names=["timestep", "lat", "lon", "features"],
                    )

                    loutputs.append(tmp_state)

            except KeyError as e:
                print("Error for param {param}")
                raise e

        # TODO : here we should directly build a single NamedTensor per inputs, outputs and forcings

        # Expand and unsqueeze our forcing fields to have the same dimension as our inputs
        for lforcing in lforcings:
            lforcing.unsqueeze_and_expand_from_(linputs[0])

        return Item(
            inputs=NamedTensor.concat(linputs),
            outputs=NamedTensor.concat(loutputs),
            forcing=NamedTensor.concat(lforcings),
        )

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple["SmeagolDataset", "SmeagolDataset", "SmeagolDataset"]:
        with open(fname, "r") as fp:
            conf = json.load(fp)

        grid = Grid(**conf["grid"])
        param_list = []
        # Reflechir a comment le faire fonctionner avec plusieurs sources.
        for data_source in conf["dataset"]:
            data = conf["dataset"][data_source]
            members = conf["dataset"][data_source].get("members", [0])
            term = conf["dataset"][data_source]["term"]
            param_grid = Grid(**data["grid"])
            for var in data["var"]:
                vard = data["var"][var]
                # Change grid definition
                if "level" in vard:
                    level_type = "hPa"
                    var_file = var
                else:
                    level_type = "m"
                    var_file = "surface"
                param = Param(
                    name=var,
                    levels=vard.pop("level", [0]),
                    grid=param_grid,
                    level_type=level_type,
                    fnamer=partial(
                        smeagol_forecast_namer,
                        model=data["grid"]["model"],
                        domain=data["grid"]["domain"],
                        geometry=data["grid"]["geometry"],
                        var=var_file,
                    ),
                    **vard,
                )
                param_list.append(param)
        train_period = Period(**conf["periods"]["train"], name="train")
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        test_period = Period(**conf["periods"]["test"], name="test")
        train_ds = SmeagolDataset(
            grid,
            train_period,
            param_list,
            SmeagolSettings(
                members=members,
                term=term,
                num_output_steps=num_pred_steps_train,
                num_input_steps=num_input_steps,
            ),
        )
        valid_ds = SmeagolDataset(
            grid,
            valid_period,
            param_list,
            SmeagolSettings(
                members=members,
                term=term,
                num_output_steps=num_pred_steps_val_test,
                num_input_steps=num_input_steps,
            ),
        )
        test_ds = SmeagolDataset(
            grid,
            test_period,
            param_list,
            SmeagolSettings(
                members=members,
                term=term,
                num_output_steps=num_pred_steps_val_test,
                num_input_steps=num_input_steps,
            ),
        )
        return train_ds, valid_ds, test_ds

    def __str__(self) -> str:
        return f"smeagol_{self.grid.geometry}"

    def torch_dataloader(
        self, tl_settings: TorchDataloaderSettings = TorchDataloaderSettings()
    ) -> DataLoader:
        return DataLoader(
            self,
            tl_settings.batch_size,
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
    def border_mask(self) -> np.array:
        return self.grid.border_mask

    def __get_params_attr(
        self,
        attribute: str,
        kind: Literal[
            "all", "input", "output", "forcing", "diagnostic", "input_output"
        ] = "all",
    ) -> List[str]:
        out_list = []
        valid_params = (
            ("output", ("output", "input_output")),
            ("forcing", ("input",)),
            ("input", ("input", "input_output")),
            ("diagnostic", ("output",)),
            ("input_output", ("input_output",)),
            ("all", ("input", "input_output", "output")),
        )
        if kind not in [k for k, pk in valid_params]:
            raise NotImplementedError(
                f"{kind} is not known. Possibilites are {[k for k,pk in valid_params]}"
            )
        for param in self.params:
            if any(kind == k and param.kind in pk for k, pk in valid_params):
                out_list += getattr(param, attribute)
        return out_list

    @cached_property
    def units(self) -> Dict[str, int]:
        """
        Return a dictionnary with name and units
        """
        dout = {}
        for param in self.params:
            names = getattr(param, "parameter_short_name")
            units = getattr(param, "units")
            for name, unit in zip(names, units):
                dout[name] = unit
        return dout

    def shortnames(self, kind: Literal["all", "input", "output"] = "all") -> List[str]:
        """
        Return the name of the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return self.__get_params_attr("parameter_short_name", kind)

    @cached_property
    def state_weights(self):
        """
        Weights used in the loss function.
        """
        w_dict = {}
        for param in self.params:
            if param.kind in ["output", "input_output"]:
                for name, weight in zip(
                    param.parameter_short_name, param.state_weights
                ):
                    w_dict[name] = weight

        return w_dict

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered.
        Usefull information for plotting.
        """
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )


class InferSmeagolDataset(SmeagolDataset):
    """
    Inherite from the SmeagolDataset class.
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
                self.settings.term["start"],
                self.settings.term["end"],
                self.settings.term["timestep"],
            )
        )

        sample_by_date = len(terms) // self.settings.num_total_steps

        samples = []
        number = 0

        for date in self.period.date_list:
            for member in self.settings.members:
                for sample in range(0, sample_by_date):

                    input_terms = terms[
                        sample
                        * self.settings.num_total_steps : sample
                        * self.settings.num_total_steps
                        + self.settings.num_input_steps
                    ]

                    output_terms = [
                        input_terms[-1] + self.settings.term["timestep"] * (step + 1)
                        for step in range(self.settings.num_inference_pred_steps)
                    ]

                    samp = InferSample(
                        date=date,
                        member=member,
                        input_terms=input_terms,
                        output_terms=output_terms,
                    )

                    if samp.is_valid(self.params):

                        samples.append(samp)
                        number += 1
        print("All samples are now defined")

        return samples

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple[None, None, "InferSmeagolDataset"]:
        """
        Return 1 InferPoesyDataset.
        Override configuration file if needed.
        """
        with open(fname, "r") as fp:
            conf = json.load(fp)
            if config_override is not None:
                conf = merge_dicts(conf, config_override)

        grid = Grid(**conf["grid"])
        param_list = []
        # Reflechir a comment le faire fonctionner avec plusieurs sources.
        for data_source in conf["dataset"]:
            data = conf["dataset"][data_source]
            members = conf["dataset"][data_source].get("members", [0])
            term = conf["dataset"][data_source]["term"]
            param_grid = Grid(**data["grid"])
            for var in data["var"]:
                vard = data["var"][var]
                # Change grid definition
                if "level" in vard:
                    level_type = "hPa"
                    var_file = var
                else:
                    level_type = "m"
                    var_file = "surface"
                param = Param(
                    name=var,
                    levels=vard.pop("level", [0]),
                    grid=param_grid,
                    level_type=level_type,
                    fnamer=partial(
                        smeagol_forecast_namer,
                        model=data["grid"]["model"],
                        domain=data["grid"]["domain"],
                        geometry=data["grid"]["geometry"],
                        var=var_file,
                    ),
                    **vard,
                )
                param_list.append(param)

        inference_period = Period(**conf["periods"]["test"], name="infer")
        ds = InferSmeagolDataset(
            grid,
            inference_period,
            param_list,
            SmeagolSettings(
                members=members,
                term=term,
                num_input_steps=num_input_steps,
                num_output_steps=0,
                num_inference_pred_steps=conf["num_inference_pred_steps"],
            ),
        )
        return None, None, ds
