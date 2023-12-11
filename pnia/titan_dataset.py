import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import xarray as xr
import yaml
from cyeccodes import nested_dd_iterator
from cyeccodes.eccodes import get_multi_messages_from_file
from pnia.base import AbstractDataset
from torch.utils.data import DataLoader, Dataset

FORMATSTR = "%Y-%m-%d_%Hh%M"
DATA_SPLIT = {
    "train": {"start": datetime(2023, 3, 1, 6), "end": datetime(2023, 3, 9, 18)},
    "valid": {"start": datetime(2023, 3, 10, 6), "end": datetime(2023, 3, 19, 18)},
    "test": {"start": datetime(2023, 3, 20, 6), "end": datetime(2023, 3, 31, 18)},
}
TITAN_DIR = Path("/scratch/shared/Titan/")
SECONDS_IN_YEAR = 365*24*60*60 # Assuming no leap years in dataset (2024 is next)

with open(TITAN_DIR / "metadata.yaml", "r") as file:
    METADATA = yaml.safe_load(file)

@dataclass
class WeatherParam:
    name: str
    long_name: str
    param: str
    model: str
    prefix_model: str
    unit: str
    cumulative: bool
    type_level: str
    levels: Tuple[int]
    grib: str
    grid: str
    shape: Tuple[int]
    extend: Tuple[float]


@dataclass
class TitanParams:

    weather_params: Tuple[Union[str, WeatherParam]] = ("aro_t2m", "aro_r2")
    isobaric_levels: Tuple[int] = (1000, 850)  # hPa
    timestep: int = 1  # hours
    nb_input_steps: int = 2
    nb_pred_steps: int = 4
    step_btw_samples: int = 6  # hours
    sub_grid: Tuple[int] = (1000, 1256, 1200, 1456)  # grid corners (pixel), lat lon grille AROME 1S100
    border_size: int = 10  # pixels
    batch_size: int = 4
    num_workers: int = 2
    standardize: bool = False  # TODO réflechir si ce paramètre est pertinent ici ?
    # pas sur car utile que pour neural_lam
    split: Literal["train", "valid", "test"] = "train"

    def __post_init__(self):
        self.date_start: datetime = DATA_SPLIT[self.split]["start"]
        self.date_end: datetime = DATA_SPLIT[self.split]["end"]
        self.shuffle: bool = (self.split == "train")
        params = []
        for param in self.weather_params:
            if isinstance(param, str):
                wp = WeatherParam(**METADATA["WEATHER_PARAMS"][param])
                if wp.grid != "PAAROME_1S100":
                    raise NotImplementedError("Can't load Arpege or Antilope data for now")
                params.append(wp)
        self.weather_params = tuple(params)


def read_grib(path_grib: Path, names=None, levels=None):
    if names or levels:
        include_filters = {
            k: v for k, v in [("cfVarName", names), ("level", levels)] if v is not None
        }
    else:
        include_filters = None
    _, results = get_multi_messages_from_file(
        path_grib,
        storage_keys=("cfVarName", "level"),
        include_filters=include_filters,
        metadata_keys=("missingValue", "Ni", "Nj"),
        include_latlon=False,
    )

    grib_dict = {}
    for metakey, result in nested_dd_iterator(results):
        array = result["values"]
        grid = (result["metadata"]["Ni"], result["metadata"]["Nj"])
        mv = result["metadata"]["missingValue"]
        array = np.reshape(array, grid)
        array = np.where(array == mv, np.nan, array)
        name, level = metakey.split("-")
        level = int(level)
        grib_dict[name] = {}
        grib_dict[name][level] = array
    return grib_dict


class TitanDataset(AbstractDataset, Dataset):
    def __init__(self, hparams: TitanParams) -> None:
        self.root_dir = TITAN_DIR
        self.init_metadata()
        self.hparams = hparams
        self.data_dir = self.root_dir / "grib"
        self.init_list_samples_dates()

    def init_list_samples_dates(self):
        sample_step = self.hparams.step_btw_samples
        timerange = self.hparams.date_end - self.hparams.date_start
        timerange = timerange.days * 24 + timerange.seconds // 3600  # convert hours
        nb_samples = timerange // sample_step
        self.samples_dates = [
            self.hparams.date_start + i * timedelta(hours=sample_step)
            for i in range(nb_samples)
        ]

    def __str__(self) -> str:
        return "TitanDataset"

    def __len__(self):
        return len(self.samples_dates)

    def init_metadata(self):
        self.grib_params = METADATA["GRIB_PARAMS"]
        self.all_isobaric_levels = METADATA["ISOBARIC_LEVELS_HPA"]
        self.all_weather_params = METADATA["WEATHER_PARAMS"]
        self.all_grids = METADATA["GRIDS"]

    def load_one_time_step(self, date: datetime) -> dict:
        sample = {}
        date_str = date.strftime(FORMATSTR)
        for grib_name, grib_keys in self.grib_params.items():
            params = [param for param in self.hparams.weather_params if param.name in grib_keys]
            names_wp = [param.param for param in params]
            if names_wp == []:
                continue
            levels = self.hparams.isobaric_levels if "ISOBARE" in grib_name else None
            path_grib = self.root_dir / "grib" / date_str / grib_name
            dico_grib = read_grib(path_grib, names_wp, levels)
            for param in params:
                sample[param.name] = dico_grib[param.param]
        return sample

    def timestep_dict_to_array(self, dico: dict) -> torch.Tensor:
        tensors = []
        for wparam in self.hparams.weather_params:  # to keep order of params
            levels_dict = dico[wparam.name]
            if len(levels_dict.keys()) == 1:
                key0 = list(levels_dict.keys())[0]
                tensors.append(torch.from_numpy(levels_dict[key0]).float())
            else:
                for lvl in self.hparams.isobaric_levels:  # to keep order of levels
                    tensors.append(torch.from_numpy(levels_dict[lvl]).float())
        return torch.stack(tensors, dim=-1)

    def get_year_hour_forcing(self, date:datetime):
        # Extract for initial step
        init_hour_in_day = date.hour
        start_of_year = datetime(date.year,1,1)
        init_seconds_into_year = (date-start_of_year).total_seconds()

        # Add increments for all steps
        sample_len = self.hparams.nb_input_steps + self.hparams.nb_pred_steps
        hour_inc = torch.arange(sample_len) * self.hparams.timestep # (sample_len,)
        hour_of_day = init_hour_in_day + hour_inc # (sample_len,), Can be > 24 but ok
        second_into_year = init_seconds_into_year + hour_inc * 3600 # (sample_len,)
        #can roll over to next year, ok because periodicity

        # Encode as sin/cos
        hour_angle = (hour_of_day/12)*torch.pi # (sample_len,)
        year_angle = (second_into_year/SECONDS_IN_YEAR)*2*torch.pi # (sample_len,)
        datetime_forcing = torch.stack((
            torch.sin(hour_angle),
            torch.cos(hour_angle),
            torch.sin(year_angle),
            torch.cos(year_angle),
            ), dim=1) # (N_t, 4)
        datetime_forcing = (datetime_forcing + 1)/2 # Rescale to [0,1]
        return datetime_forcing

    def __getitem__(self, index):
        date_t0 = self.samples_dates[index]

        # ------- State features -------

        dates = []
        for i in range(-self.hparams.nb_input_steps + 1, self.hparams.nb_pred_steps + 1):
            dates.append(date_t0 + i * timedelta(hours=self.hparams.timestep))
        step_dicts = [self.load_one_time_step(date) for date in dates]
        step_tensors = [self.timestep_dict_to_array(dico) for dico in step_dicts]
        states = torch.stack(step_tensors, dim=0)  # (steps, Nx, Ny, features) : [6, 2801, 1791, 2]

        # Apply subgrid
        grid = self.hparams.sub_grid
        states = states[:, grid[0]:grid[1], grid[2]:grid[3], :]  # shape [6, 256, 256, 2]

        # Flatten spatial dim
        states = states.flatten(1,2)  # (steps, grid_points, features)  [6, 65536, 2]

        # Split sample in init states and target states
        init_states = states[:self.hparams.nb_input_steps]  # [2, 65536, 2]
        target_states = states[self.hparams.nb_input_steps:]  # [4, 65536, 2]

        # TODO : accumuler variables cumulatives dans initstates et targetstates
        # TODO : standardisation des variables

        # ------- Static features : only water coverage, on s'en fout ?

        # TODO : remplacer par vrai masque terre-mer
        static_features = torch.zeros((init_states.shape[1], 1)) # (grid_points, 1)

        # ------- Forcing features -------
        # TODO : flux for real : "nwp_toa_downwelling_shortwave_flux"
        flux = torch.zeros((states.shape[0], states.shape[1], 1))

        datetime_forcing = self.get_year_hour_forcing(date_t0)
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(-1,
                flux.shape[1], -1) # (sample_len, N_grid, 4)

        # Put forcing features together
        forcing_features = torch.cat((flux, datetime_forcing),
                dim=-1) # (sample_len, N_grid, d_forcing)

        # TODO : WTF ?!
        # Combine forcing over each window of 3 time steps
        forcing_windowed = torch.cat((
            forcing_features[:-2],
            forcing_features[1:-1],
            forcing_features[2:],
            ), dim=2) # (sample_len-2, N_grid, 3*d_forcing)
        # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        return init_states, target_states, static_features, forcing_windowed

    def get_dataloader(self):
        return DataLoader(
            self,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
        )

    @property
    def shape(self) -> Tuple[int]:
        corners = self.hparams.sub_grid
        shape = (corners[1] - corners[0], corners[3] - corners[2])
        return shape

    @property
    def grid_info(self) -> np.array:
        conf_ds = xr.load_dataset(self.root_dir / "conf.grib")
        corners = self.hparams.sub_grid
        latitudes = conf_ds.latitude[corners[0] : corners[1]]
        longitudes = conf_ds.longitude[corners[2] : corners[3]]
        grid = np.array(np.meshgrid(longitudes, latitudes))
        return grid

    @property
    def geopotential_info(self) -> np.array:
        conf_ds = xr.load_dataset(self.root_dir / "conf.grib")
        corners = self.hparams.sub_grid
        return conf_ds.h.values[:, corners[0] : corners[1], corners[2] : corners[3]]

    @property
    def limited_area(self) -> bool:
        return True

    @property
    def border_mask(self) -> np.array:
        border_mask = np.ones((self.shape[0], self.shape[1])).astype(bool)
        size = self.border_size
        border_mask[size:-size, size:-size] *= False
        return border_mask

    @property
    def split(self) -> Literal["train", "valid", "test"]:
        return self.hparams.split

    @property
    def nb_pred_steps(self) -> int:
        return self.hparams.nb_pred_steps

    @property
    def weather_params(self) -> List[str]:
        return self.hparams.weather_params.keys()

    @property
    def standardize(self) -> bool:
        return self.hparams.standardize


if __name__ == "__main__":
    hparams = TitanParams()
    print('hparams : ', hparams)
    dataset = TitanDataset(hparams)
    print('dataset : ', dataset)
    print('len(dataset) : ', len(dataset))
    # print("dataset.grid_info : ", dataset.grid_info)
    beg = time.time()
    x, y, static, forcing = dataset.__getitem__(3)
    print("time : ", time.time() - beg)
    print("x.shape : ", x.shape)
    print("y.shape : ", y.shape)
    loader = dataset.get_dataloader()
    print(loader)
