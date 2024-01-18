
import datetime as dt
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union
from collections.abc import Callable
import json
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
import torch
from tqdm import tqdm

from pnia.settings import CACHE_DIR
from projects.pnia.pnia.datasets.base import AbstractDataset
from projects.pnia.pnia.datasets import create_parameter_weights

torch.set_num_threads(8)
priam_path = Path("/scratch/shared/smeagol")
# Assuming no leap years in dataset (2024 is next)
SECONDS_IN_YEAR = 365 * 24 * 60 * 60


def constant_fname(domain, model, geometry):
    return f"{domain}/{model}_{geometry}/constant/Mesh_and_SurfGeo.nc"


def smeagol_forecast_namer(date: dt.datetime, member: int, var, model, geometry, domain, **kwargs):
    """
    use to find local files
    """
    return f"{domain}/{model}_{geometry}/{date.strftime('%Y%m%dH%H')}/mb_{str(member).zfill(3)}_{var}.nc"


def get_weight(level: int, kind: str):
    if kind == "hPa":
        return (level)/(90)
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
        return pd.date_range(start=self.start, end=self.end, freq=f"{self.step}H").to_pydatetime()


@dataclass
class Grid:
    domain: str
    model: str
    geometry: str = "franmgsp32"
    border_size: int = 0
    # Subgrid selection. If (0,0,0,0) the whole grid is kept.
    subgrid: Tuple[int] = (0, 0, 0, 0)
    lat: np.array = field(init=False)
    lon: np.array = field(init=False)
    x: int = field(init=False)  # X dimension
    y: int = field(init=False)  # Y dimension

    def __post_init__(self):
        grid_name = priam_path / "nc" / \
            constant_fname(self.domain, self.model, self.geometry)
        ds = xr.open_dataset(grid_name)
        lat = ds.lat.values
        x, y = ds.lat.shape
        # Setting correct subgrid if no subgrid is selected.
        if self.subgrid == (0, 0, 0, 0):
            self.subgrid = (0, x, 0, y)

        self.x = self.subgrid[1] - self.subgrid[0]
        self.y = self.subgrid[3] - self.subgrid[2]

        self.lat = ds.lat.values[self.subgrid[0]
            :self.subgrid[1], self.subgrid[2]:self.subgrid[3]]
        self.lon = ds.lon.values[self.subgrid[0]
            :self.subgrid[1], self.subgrid[2]:self.subgrid[3]]

    @property
    def geopotential(self) -> np.array:
        grid_name = priam_path / "nc" / \
            constant_fname(self.domain, self.model, self.geometry)
        ds = xr.open_dataset(grid_name)
        return ds["SURFGEOPOTENTIEL"].values[self.subgrid[0]:self.subgrid[1], self.subgrid[2]:self.subgrid[3]]

    @property
    def landsea_mask(self) -> np.array:
        grid_name = priam_path / "nc" / \
            constant_fname(self.domain, self.model, self.geometry)
        ds = xr.open_dataset(grid_name)
        return ds["LandSeaMask"].values[self.subgrid[0]:self.subgrid[1], self.subgrid[2]:self.subgrid[3]]

    @property
    def border_mask(self) -> np.array:
        border_mask = np.ones((self.x, self.y)).astype(bool)
        size = self.border_size
        border_mask[size:-size, size:-size] *= False
        return border_mask

    @property
    def N_grid(self) -> int:
        return self.x * self.y


@dataclass
class Param:
    name: str
    shortname:str # To be read in nc File ?
    levels: Tuple[int]
    grid: Grid
    # Function which can return the filenames. It should accept member and date as argument (as well as term).
    fnamer: Callable[[], [str]]
    level_type: str = "hPa" # To be read in nc file ?
    # Defini le statut du parametre. Les forcage sont seulement en input. Les variables diagnostiques seulement en output.
    kind: Literal["input", "output", "input_output"] = "input_output"
    unit:str = "FakeUnit" # To be read in nc FIle  ?


    @property
    def parameter_weights(self) -> list:
        #wlist = []
        #for level in self.levels:
        #    wlist.append(get_weight(level, self.level_type))
        #return wlist
        return [get_weight(level, self.level_type) for level in self.levels]

    @property
    def parameter_name(self) -> list:
        return [f"{self.name}_{level}" for level in self.levels]


    @property
    def parameter_short_name(self) -> list:
        return [f"{self.shortname}_{level}" for level in self.levels]
#st

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
        # TO DO : change it in order to be more generic. Here we postpone that there is only one file.
        # However it may not be the case (e.g when speaking of radars multiple files could be used).
        ds = xr.open_dataset(files[0], decode_times=False)
        return ds

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
        for term in self.terms:
            date_tmp = self.date + dt.timedelta(hours=term)
            hours.append(date_tmp.hour + date_tmp.minute / 60)
        return np.asarray(hours)

    @property
    def seconds_into_year(self) -> np.array:
        start_of_year = dt.datetime(self.date.year, 1, 1)
        return np.asarray([(self.date + dt.timedelta(hours=term) - start_of_year).total_seconds() for term in self.terms])

@dataclass
class HyperParam:
    term: dict  # Pas vraiment a mettre ici. Voir où le mettre
    nb_input_steps: int = 2  # Step en entrée
    nb_pred_steps: int = 1  # Step en sortie
    batch_size: int = 4  # Nombre d'élément par batch
    num_workers: int = 2  # Worker pour charger les données
    standardize: bool = False
    subset:int = 0 # Positive integer. If subset is less than 1 it means full set. Otherwise describe the number of sample.
    # Pas vraiment a mettre ici. Voir où le mettre
    members: Tuple[int] = (0,)

    @property
    def nb_steps(self):
        # Nb of step in on sample
        return self.nb_input_steps + self.nb_pred_steps

class SmeagolDataset(AbstractDataset, Dataset):

    def __init__(self, grid: Grid, period: Period, params: List[Param], hyper_params: HyperParam):
        self.grid = grid
        self.period = period
        self.params = params
        self.hp = hyper_params
        self.cache_dir = CACHE_DIR / "neural_lam" / str(self)
        self.shuffle = self.split == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_list = self.init_sample_list()

        if self.hp.subset > 1:
            self.sample_list = self.sample_list[:self.subset]

        if self.standardize:
            ds_stats = self.load_dataset_stats()
            self.data_mean, self.data_std, self.flux_mean, self.flux_std =\
                ds_stats["data_mean"], ds_stats["data_std"], ds_stats["flux_mean"], \
                ds_stats["flux_std"]


    def init_sample_list(self):
        """
        Create a list of sample from information
        """
        terms = list(np.arange(
            self.hp.term["start"], self.hp.term["end"], self.hp.term["timestep"]))
        sample_by_date = len(terms) // self.hp.nb_steps
        sample_list = []
        for date in self.period.date_list:
            for member in self.hp.members:
                for sample in range(0, sample_by_date):
                    input_terms = terms[sample * self.hp.nb_steps: sample *
                                        self.hp.nb_steps + self.hp.nb_input_steps]
                    output_terms = terms[sample * self.hp.nb_steps + self.hp.nb_input_steps: sample *
                                         self.hp.nb_steps + self.hp.nb_input_steps + self.hp.nb_pred_steps]
                    samp = Sample(
                        member=member, date=date, input_terms=input_terms, output_terms=output_terms)
                    sample_list.append(samp)
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def get_year_hour_forcing(self, sample: Sample):
        """
        Get the forcing term dependent of the sample time
        """
        hour_angle = (torch.Tensor(sample.hours_of_day) / 12) * \
            torch.pi  # (sample_len,)
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

    @property
    def static_feature_dim(self):
        """
        Only landSeaMask currently
        """
        return 1

    @property
    def forcing_dim(self):
        res = 4 # Pour la date
        for param in self.params:
            if param.kind =="input":
                res +=1
        return res

    @property
    def weather_dim(self):
        """
        Retourne la dimensions des variables présentes en entrée et en sortie.
        """
        res = 0
        for param in self.params:
            if param.kind =="input_output":
                res +=1
        return res

    @property
    def diagnostic_dim(self):
        """
        Return dimensions of output variable only
        Not used yet
        """
        res = 0
        for param in self.params:
            if param.kind =="output":
                res +=1
        return res


    def __getitem__(self, index):
        sample = self.sample_list[index]

        # Static features
        static_features = torch.from_numpy(
            self.grid.landsea_mask).flatten().unsqueeze(1)

        # Datetime Forcing
        datetime_forcing = self.get_year_hour_forcing(sample)
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, self.grid.N_grid, -1
        )
        inputs = []
        outputs = []
        forcing = []

        beg = time.time()
        # Reading parameters from files
        for param in self.params:
            ds = param.load_data(sample.member, sample.date, sample.terms).sel(X=range(
                self.grid.subgrid[0], self.grid.subgrid[1])).sel(Y=range(self.grid.subgrid[2], self.grid.subgrid[3]))
            if "level" in ds.coords:
                ds = ds.sel(level=param.levels)
            # Read inputs. Separate forcing field from I/O
            if param.kind == "input_output":
                tmp_in = ds[param.name].sel(step=sample.input_terms).values
                if len(tmp_in.shape) != 4:
                    tmp_in = np.expand_dims(tmp_in, axis=1)
                inputs.append(tmp_in)
            # On lit un forcage. On le prend pour tous les pas de temps (input + prevision)
            elif param.kind == "input":
                tmp_in = ds[param.name].sel(
                    step=sample.input_terms + sample.output_terms).values
                if len(tmp_in.shape) != 4:
                    tmp_in = np.expand_dims(tmp_in, axis=1)
                forcing.append(tmp_in)
            # Read outputs.
            if param.kind in ["ouput", "input_output"]:
                tmp_out = ds[param.name].sel(step=sample.output_terms).values
                if len(tmp_out.shape) != 4:
                    tmp_out = np.expand_dims(tmp_out, axis=1)
                outputs.append(tmp_out)
        print("File reading time : ", time.time() - beg)
        # To do
        # - standardizaion
        # - accumuler (decumuler ?)
        # - Verifier flux solaire

        ini = np.concatenate(inputs, axis=1).transpose([0, 2, 3, 1])
        force = np.concatenate(forcing, axis=1).transpose([0, 2, 3, 1])
        outi = np.concatenate(outputs, axis=1).transpose([0, 2, 3, 1])

        state_in = torch.from_numpy(ini)
        state_out = torch.from_numpy(outi)

        if self.standardize:
            # May be rething the way we save mean and std in order to be consistent in standardization.
            # Saving npz with variable name may be better (the file is very small and load only one time)?
            print("Standardization is not implemented yet.")

        # Adjust the forcing. Add datetime to forcing.
        forcing = torch.from_numpy(force).flatten(1, 2)
        forcing = torch.cat((forcing, datetime_forcing), dim=-1)
        state_in = state_in.flatten(1, 2)
        state_out = state_out.flatten(1, 2)

        print(index, state_in.shape, state_out.shape,
              static_features.shape, forcing.shape)

        # To Do
        # Combine forcing over each window of 3 time steps
        # Comprendre ce que ça fait avant de voir comment l'adapter (car problematique potentiellement différente).
        return state_in, state_out, static_features, forcing

    @classmethod
    def from_json(cls, file, split="train"):
        with open(file, "r") as fp:
            conf = json.load(fp)
        grid = Grid(**conf["grid"])
        my_period = Period(**conf["periods"][split], name=split)
        param_list = []
        # Reflechir a comment le faire fonctionner avec plusieurs sources.
        for data_source in conf["dataset"]:
            data = conf["dataset"][data_source]
            # Voir comment les rajouter
            # Pour l'instant ils sont passer en HyperParametre (pas top du tout car on s'amusera certainement pas à la volée avec).
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
                        var=var_file),
                    **vard)
                param_list.append(param)
        return SmeagolDataset(grid, my_period, param_list, HyperParam(members=members, term=term))

    def __str__(self) -> str:
        return f"smeagol_{self.grid.geometry}"

    @property
    def loader(self) -> DataLoader:
        print("Shuffle ",self.shuffle)
        return DataLoader(self,
                          self.hp.batch_size,
                          num_workers=self.hp.num_workers,
                          shuffle=self.shuffle)

    @property
    def grid_info(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (lat, lon) values
        """
        return np.asarray([self.grid.lat, self.grid.lon])

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

    def __get_params_attr(self, attribute:str, kind:Literal["all","input","output"] = "all") -> List[str]:
        out_list = []
        for param in self.params:
            if kind == "all":
                out_list += getattr(param, attribute)
            elif kind == "input" and param.kind in  ["input", "input_output"]:
                out_list += getattr(param, attribute)
            elif kind == "output" and param.kind in  ["ouput", "input_output"]:
                out_list += getattr(param, attribute)
        return out_list

    def shortnames(self,kind:Literal["all","input","output"]='all') -> List[str]:
        """
        Return the name of the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return self.__get_params_attr("parameter_short_name", kind)

    def units(self,kind:Literal["all","input","output"]='all') -> List[str]:
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
        for param in self.params:
            if param.kind in ["output", "input_output"]:
                w_list += param.parameter_weights
        np.save(self.cache_dir / "parameter_weights.npy",
                np.asarray(w_list).astype("float32"))

    def load_dataset_stats(self, device="cpu"):

        data_mean = self.load_file("parameter_mean.pt") # (d_features,)
        data_std = self.load_file("parameter_std.pt") # (d_features,)
        flux_stats = self.load_file("flux_stats.pt") # (2,)
        flux_mean, flux_std = flux_stats

        return {
            "data_mean": data_mean,
            "data_std": data_std,
            "flux_mean": flux_mean,
            "flux_std": flux_std,
        }

    @property
    def parameter_weights(self)->np.array:
        if not (self.cache_dir / "parameter_weights.npy").exists():
            self.create_parameter_weights()
        return np.load(self.cache_dir / "parameter_weights.npy")

    @property
    def standardize(self) -> bool:
        return self.hp.standardize

    @standardize.setter
    def standardize(self, value:bool):
        self.hp.standardize = value

    @property
    def nb_pred_steps(self) -> int:
        return self.hp.nb_pred_steps

    @property
    def members(self):
        return self.hp.members

    @members.setter
    def members(self, value:tuple):
        self.hp.members = value
        self.sample_list = self.init_sample_list()
        if self.hp.subset > 1:
            self.sample_list = self.sample_list[:self.subset]

if __name__ == "__main__":
    from projects.pnia.pnia.models.nlam import create_mesh
    from projects.pnia.pnia.datasets import create_grid_features

    import time
    dataset = SmeagolDataset.from_json(
        "/home/mrpa/chabotv/pnia/pnia/xp_conf/smeagol.json")
    print(dataset.shortnames(kind="output"))
    print(dataset.units(kind="output"))
    dataset.create_parameter_weights()
    print(dataset.parameter_weights)
    print(dataset.weather_params)


    #print(dataset.parameter_weights)
    #dataset.create_parameter_weights()
    #beg = time.time()
    #x, y, static, forcing = dataset.__getitem__(0)
    #print("time : ", time.time() - beg)
    #print(dataset.loader)
    #
    #dataset.members=(0,)
    #dataset.shuffle = False
    #dataset.compute_parameters_stats()
    #dataset.compute_timestep_stats()
    #create_mesh.prepare(dataset)
    #create_grid_features.prepare(dataset)
    #print(dataset.load_dataset_stats())
    print(dataset.load_static_data())


