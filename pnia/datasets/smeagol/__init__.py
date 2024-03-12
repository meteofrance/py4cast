import datetime as dt
import json
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import List, Literal, Tuple

import cartopy
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from pnia.datasets.base import AbstractDataset
from pnia.settings import CACHE_DIR

ssl._create_default_https_context = ssl._create_unverified_context
# torch.set_num_threads(8)
priam_path = Path("/scratch/shared/smeagol")
# Assuming no leap years in dataset (2024 is next)
SECONDS_IN_YEAR = 365 * 24 * 60 * 60


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


@dataclass
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
    # lat: np.array = field(init=False)
    # lon: np.array = field(init=False)
    x: int = field(init=False)  # X dimension
    y: int = field(init=False)  # Y dimension

    def __post_init__(self):
        grid_name = (
            priam_path / "nc" / constant_fname(self.domain, self.model, self.geometry)
        )
        ds = xr.open_dataset(grid_name)
        # lat = ds.lat.values
        x, y = ds.lat.shape
        # Setting correct subgrid if no subgrid is selected.
        if self.subgrid == (0, 0, 0, 0):
            self.subgrid = (0, x, 0, y)

        self.x = self.subgrid[1] - self.subgrid[0]
        self.y = self.subgrid[3] - self.subgrid[2]

    @cached_property
    def lat(self) -> np.array:
        grid_name = (
            priam_path / "nc" / constant_fname(self.domain, self.model, self.geometry)
        )
        ds = xr.open_dataset(grid_name)
        return ds.lat.values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @cached_property
    def lon(self) -> np.array:
        grid_name = (
            priam_path / "nc" / constant_fname(self.domain, self.model, self.geometry)
        )
        ds = xr.open_dataset(grid_name)
        return ds.lon.values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @property
    def geopotential(self) -> np.array:
        grid_name = (
            priam_path / "nc" / constant_fname(self.domain, self.model, self.geometry)
        )
        ds = xr.open_dataset(grid_name)
        return ds["SURFGEOPOTENTIEL"].values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @property
    def landsea_mask(self) -> np.array:
        grid_name = (
            priam_path / "nc" / constant_fname(self.domain, self.model, self.geometry)
        )
        ds = xr.open_dataset(grid_name)
        return ds["LandSeaMask"].values[
            self.subgrid[0] : self.subgrid[1], self.subgrid[2] : self.subgrid[3]
        ]

    @property
    def border_mask(self) -> np.array:
        border_mask = np.ones((self.x, self.y)).astype(bool)
        size = self.border_size
        border_mask[size:-size, size:-size] *= False
        return border_mask

    @property
    def N_grid(self) -> int:
        return self.x * self.y

    @cached_property
    def grid_limits(self):
        grid_name = (
            priam_path / "nc" / constant_fname(self.domain, self.model, self.geometry)
        )
        ds = xr.open_dataset(grid_name)
        grid_limits = [  # In projection
            ds.x[self.subgrid[0]].values,  # min x
            ds.x[self.subgrid[1]].values,  # max x
            ds.y[self.subgrid[2]].values,  # min y
            ds.y[self.subgrid[3]].values,  # max y
        ]
        return grid_limits

    @cached_property
    def projection(self):
        # Create projection
        return cartopy.crs.LambertConformal(central_longitude=2, central_latitude=46.7)


@dataclass
class Param:
    name: str
    shortname: str  # To be read in nc File ?
    levels: Tuple[int]
    grid: Grid
    # Function which can return the filenames.
    # It should accept member and date as argument (as well as term).
    fnamer: Callable[[], [str]]
    level_type: str = "hPa"  # To be read in nc file ?
    # Defini le statut du parametre. Les forcage sont seulement en input.
    # Les variables diagnostiques seulement en output.
    kind: Literal["input", "output", "input_output"] = "input_output"
    unit: str = "FakeUnit"  # To be read in nc FIle  ?

    @property
    def number(self) -> int:
        """
        Get the number of parameters.
        """
        return len(self.levels)

    @property
    def parameter_weights(self) -> list:
        return [get_weight(level, self.level_type) for level in self.levels]

    @property
    def parameter_name(self) -> list:
        return [f"{self.name}_{level}" for level in self.levels]

    @property
    def parameter_short_name(self) -> list:
        return [f"{self.shortname}_{level}" for level in self.levels]

    @property
    def units(self) -> list:
        return [self.unit for level in self.levels]

    def filename(self, member: int, date: dt.datetime, term: float) -> str:
        """
        Return the filename. Even if term is not used for this example it could be (e.g. for radars).
        """
        return priam_path / "nc" / self.fnamer(date=date, member=member, term=term)

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


@dataclass
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

    @property
    def hours_of_day(self) -> np.array:
        hours = []
        for term in self.output_terms:
            date_tmp = self.date + dt.timedelta(hours=term)
            hours.append(date_tmp.hour + date_tmp.minute / 60)
        return np.asarray(hours)

    @property
    def seconds_into_year(self) -> np.array:
        start_of_year = dt.datetime(self.date.year, 1, 1)
        return np.asarray(
            [
                (self.date + dt.timedelta(hours=term) - start_of_year).total_seconds()
                for term in self.output_terms
            ]
        )

    def is_valid(self, param_list):
        for param in param_list:
            if not param.exist(self.member, self.date, self.terms):
                return False
            else:
                return True


@dataclass
class HyperParam:
    term: dict  # Pas vraiment a mettre ici. Voir où le mettre
    nb_input_steps: int = 2  # Step en entrée
    nb_pred_steps: int = 1  # Step en sortie
    batch_size: int = 1  # Nombre d'élément par batch
    num_workers: int = 10  # Worker pour charger les données
    standardize: bool = False
    subset: int = 0  # Positive integer. If subset is less than 1 it means full set.
    # Otherwise describe the number of sample.
    # Pas vraiment a mettre ici. Voir où le mettre
    members: Tuple[int] = (0,)
    diagnose: bool = False  # Do we want extra diagnostic ? Do not use it for training
    prefetch: int = 2

    @property
    def nb_steps(self):
        # Nb of step in on sample
        return self.nb_input_steps + self.nb_pred_steps


class SmeagolDataset(AbstractDataset, Dataset):

    recompute_stats: bool = False

    def __init__(
        self, grid: Grid, period: Period, params: List[Param], hyper_params: HyperParam
    ):
        self.grid = grid
        self.period = period
        self.params = params
        self.hp = hyper_params

        self._cache_dir = CACHE_DIR / "neural_lam" / str(self)
        self.shuffle = self.split == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.standardize:
            ds_stats = self.statistics
            self.data_mean, self.data_std, self.flux_mean, self.flux_std = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
                ds_stats["flux_mean"],
                ds_stats["flux_std"],
            )

    @cached_property
    def sample_list(self):
        """
        Create a list of sample from information
        """
        terms = list(
            np.arange(
                self.hp.term["start"], self.hp.term["end"], self.hp.term["timestep"]
            )
        )
        sample_by_date = len(terms) // self.hp.nb_steps
        samples = []
        number = 0
        for date in self.period.date_list:
            for member in self.hp.members:
                for sample in range(0, sample_by_date):
                    input_terms = terms[
                        sample * self.hp.nb_steps : sample * self.hp.nb_steps
                        + self.hp.nb_input_steps
                    ]
                    output_terms = terms[
                        sample * self.hp.nb_steps
                        + self.hp.nb_input_steps : sample * self.hp.nb_steps
                        + self.hp.nb_input_steps
                        + self.hp.nb_pred_steps
                    ]
                    samp = Sample(
                        member=member,
                        date=date,
                        input_terms=input_terms,
                        output_terms=output_terms,
                    )
                    if samp.is_valid(self.params) and (
                        number < self.hp.subset or self.hp.subset < 1
                    ):
                        samples.append(samp)
                        number += 1
        return samples

    def __len__(self):
        return len(self.sample_list)

    def get_year_hour_forcing(self, sample: Sample):
        """
        Get the forcing term dependent of the sample time
        """
        hour_angle = (
            torch.Tensor(sample.hours_of_day) / 12
        ) * torch.pi  # (sample_len,)
        year_angle = (
            (torch.Tensor(sample.seconds_into_year) / SECONDS_IN_YEAR) * 2 * torch.pi
        )  # (sample_len,)
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (N_t, 4)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        return datetime_forcing

    @cached_property
    def subgrid(self):
        """
        Retourne les indices de la sous grille
        """
        return self.grid.subgrid

    @cached_property
    def static_feature_dim(self):
        """
        Only landSeaMask currently
        """
        return 1

    @cached_property
    def forcing_dim(self):
        res = 4  # Pour la date
        for param in self.params:
            if param.kind == "input":
                res += param.number
        return res

    @cached_property
    def weather_dim(self):
        """
        Retourne la dimensions des variables présentes en entrée et en sortie.
        """
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += param.number
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
                res += param.number
        return res

    def test_sample(self, index):
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
            try:
                _ = ds[param.name].sel(step=sample.input_terms)
            except KeyError:
                print(f"Problem in sample {sample}")

    def __getitem__(self, index):
        sample = self.sample_list[index]
        # Static features
        # Peut etre la charger prealable
        static_features = (
            torch.from_numpy(self.grid.landsea_mask).flatten().unsqueeze(1)
        )

        # Datetime Forcing
        datetime_forcing = self.get_year_hour_forcing(sample)
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, self.grid.N_grid, -1
        )

        inputs = []
        outputs = []
        forcing = []

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
                if param.kind == "input_output":
                    tmp_in = ds[param.name].sel(step=sample.input_terms).values
                    if len(tmp_in.shape) != 4:
                        tmp_in = np.expand_dims(tmp_in, axis=1)
                    inputs.append(tmp_in)
                # On lit un forcage. On le prend pour tous les pas de temps de prevision
                # Un peu etrange de prendre le forcage a l'instant de la prevision et
                # non pas l'instant initial ... mais bon.
                elif param.kind == "input":
                    tmp_in = ds[param.name].sel(step=sample.output_terms).values
                    if len(tmp_in.shape) != 4:
                        tmp_in = np.expand_dims(tmp_in, axis=1)
                    forcing.append(tmp_in)
                # Read outputs.
                if param.kind in ["ouput", "input_output"]:
                    tmp_out = ds[param.name].sel(step=sample.output_terms).values
                    if len(tmp_out.shape) != 4:
                        tmp_out = np.expand_dims(tmp_out, axis=1)
                    outputs.append(tmp_out)
            except KeyError as e:
                print("Error for param {param}")
                raise e
        ini = np.concatenate(inputs, axis=1).transpose([0, 2, 3, 1])
        force = torch.from_numpy(
            np.concatenate(forcing, axis=1).transpose([0, 2, 3, 1])
        )
        outi = np.concatenate(outputs, axis=1).transpose([0, 2, 3, 1])
        state_in = torch.from_numpy(ini)
        state_out = torch.from_numpy(outi)

        if self.standardize:
            # TO DO
            # Fait ici l'hypothese implicite que les donnes d'entree et de sortie sont les memes.
            # Rajouter un champ diagnose pour les sorties seules que l'on stackera plus tard ?
            # Trouver un truc plus logique pour faire la standardisation
            # (la faire parametre par parametre a la lecture ?) ?
            # Construire un tableau de normalisation pour la sortie different de pour
            # les entree (permettant d'avoir plus de souplesse )?
            state_in = (state_in - self.data_mean) / self.data_std
            state_out = (state_out - self.data_mean) / self.data_std
            force = (force - self.flux_mean) / self.flux_std

        # Adjust the forcing. Add datetime to forcing.
        forcing = force.flatten(1, 2)
        forcing = torch.cat((forcing, datetime_forcing), dim=-1)
        state_in = state_in.flatten(1, 2)
        state_out = state_out.flatten(1, 2)
        if self.hp.diagnose:
            print(f"In __get_item__ : {torch.mean(state_in,dim=(0,1))}")
        # To Do
        # Combine forcing over each window of 3 time steps
        # Comprendre ce que ça fait avant de voir comment l'adapter (car problematique potentiellement différente).
        return (
            state_in.type(torch.float32),
            state_out.type(torch.float32),
            static_features.type(torch.float32),
            forcing.type(torch.float32),
        )

    @classmethod
    def from_json(cls, file, args={}):
        with open(file, "r") as fp:
            conf = json.load(fp)
        grid = Grid(**conf["grid"])
        param_list = []
        # Reflechir a comment le faire fonctionner avec plusieurs sources.
        for data_source in conf["dataset"]:
            data = conf["dataset"][data_source]
            # Voir comment les rajouter
            # Pour l'instant ils sont passer en HyperParametre
            # (pas top du tout car on s'amusera certainement pas à la volée avec).
            members = conf["dataset"][data_source].get("members", [0])
            term = conf["dataset"][data_source]["term"]
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
                    grid=grid,
                    level_type=level_type,
                    fnamer=partial(
                        smeagol_forecast_namer,
                        model=data["model"],
                        domain=data["domain"],
                        geometry=data["geometry"],
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
            HyperParam(members=members, term=term, **args.get("train", {})),
        )
        valid_ds = SmeagolDataset(
            grid,
            valid_period,
            param_list,
            HyperParam(members=members, term=term, **args.get("valid", {})),
        )
        test_ds = SmeagolDataset(
            grid,
            test_period,
            param_list,
            HyperParam(members=members, term=term, **args.get("test", {})),
        )
        return train_ds, valid_ds, test_ds

    def __str__(self) -> str:
        return f"smeagol_{self.grid.geometry}"

    @property
    def loader(self) -> DataLoader:
        return DataLoader(
            self,
            self.hp.batch_size,
            num_workers=self.hp.num_workers,
            shuffle=self.shuffle,
            prefetch_factor=self.hp.prefetch,
        )

    @property
    def grid_info(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (lat, lon) values
        """
        return np.asarray([self.grid.lon, self.grid.lat])

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

    @property
    def weather_params(self) -> List[str]:
        """
        Return the name of the parameters in the dataset.
        Does not make the difference between inputs, outputs and forcing.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        """
        return self.__get_params_attr("parameter_name", "all")

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

    def shortnames(
        self,
        kind: Literal[
            "all", "input", "output", "forcing", "diagnostic", "input_output"
        ] = "all",
    ) -> List[str]:
        """
        Return the name of the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return self.__get_params_attr("parameter_short_name", kind)

    def units(self, kind: Literal["all", "input", "output"] = "all") -> List[str]:
        """
        Return the name of the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return self.__get_params_attr("units", kind)

    def create_parameter_weights(self):
        """
        Create parameter weights (for the loss function).
        Each variable has is own weight.

        May evolve to have a weight dependent of the situation ?
        """
        w_list = []
        w_dict = {}
        for param in self.params:
            if param.kind in ["output", "input_output"]:
                for x, y in zip(param.parameter_short_name, param.parameter_weights):
                    w_dict[x] = float(y)
                w_list += param.parameter_weights
        path = self.cache_dir / "parameter_weights.npz"
        if path.exists():
            path.unlink()
        np.savez(path, **w_dict)
        path.chmod(0o666)

    @property
    def parameter_weights(self) -> np.array:
        if not (self.cache_dir / "parameter_weights.npz").exists():
            self.create_parameter_weights()
        params = np.load(self.cache_dir / "parameter_weights.npz")
        w_list = []
        for param in self.params:
            if param.kind in ["output", "input_output"]:
                w_list += [params[p] for p in param.parameter_short_name]
        return np.asarray(w_list)

    @property
    def standardize(self) -> bool:
        return self.hp.standardize

    @standardize.setter
    def standardize(self, value: bool):
        self.hp.standardize = value
        if self.standardize:
            ds_stats = self.statistics
            self.data_mean, self.data_std, self.flux_mean, self.flux_std = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
                ds_stats["flux_mean"],
                ds_stats["flux_std"],
            )

    @property
    def nb_pred_steps(self) -> int:
        return self.hp.nb_pred_steps

    @property
    def members(self):
        return self.hp.members

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    # On va encoder les proprietes pour la visu
    @property
    def grid_limits(self) -> list:
        return self.grid.grid_limits

    @property
    def projection(self):
        return self.grid.projection
