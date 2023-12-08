import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from pnia.base import AbstractDataset, AbstractHyperParams
from cyeccodes import nested_dd_iterator
from cyeccodes.eccodes import get_multi_messages_from_file
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import yaml
from typing import List, Tuple, Union, Literal
from pathlib import Path
from torchvision import transforms
from mfai.torch.transforms import ToTensor
import torch
import time

FORMATSTR = "%Y-%m-%d_%Hh%M"


@dataclass
class TitanParams(AbstractHyperParams):

    weather_params: Tuple[str] = ("aro_t2m", "aro_r2", "aro_u10", "aro_v10")
    isobaric_levels: Tuple[int] = (1000, 850)  # hPa
    nb_input_steps: int = 2
    input_step: int = 1  # hours
    nb_pred_steps: int = 4
    pred_step: int = 1  # hours
    step_btw_samples: int = 6  # hours
    sub_grid: Tuple[int] = (1000, 1256, 1200, 1456)  # grid corners (pixel), lat lon
    border_size: int = 10  # pixels
    date_begining: Union[str, datetime] = datetime(2023, 3, 1, 0)
    date_end: Union[str, datetime] = datetime(2023, 3, 31, 23)
    split: Literal["train", "valid", "test"] = "train"
    batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 2

    def __post_init__(self):
        if isinstance(self.date_begining, str):
            self.date_begining = datetime.strptime(self.date_begining, FORMATSTR)
        if isinstance(self.date_end, str):
            self.date_end = datetime.strptime(self.date_end, FORMATSTR)


def read_grib(path_grib: Path, names=None, levels=None):
    if names or levels:
        include_filters={k:v for k,v in [("cfVarName", names), ("level", levels)] if v is not None}
    else:
        include_filters = None
    _, results = get_multi_messages_from_file(path_grib, storage_keys=("cfVarName", "level"), include_filters=include_filters, metadata_keys=("missingValue", "Ni", "Nj"), include_latlon = False)

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
        self.root_dir = Path("/scratch/shared/Titan/")
        self.init_metadata()
        self.hparams = hparams
        self.data_dir = self.root_dir / "grib"
        self.init_list_samples_dates()

    def init_list_samples_dates(self):
        # TODO : split train et test
        sample_step = self.hparams.step_btw_samples
        timerange = self.hparams.date_end - self.hparams.date_begining
        timerange = timerange.days * 24 + timerange.seconds // 3600  # convert hours
        nb_samples = timerange // sample_step
        self.samples_dates = [self.hparams.date_begining + i * timedelta(hours=sample_step) for i in range(nb_samples)]

    def __str__(self) -> str:
        return "TitanDataset"

    def __len__(self):
        return len(self.samples_dates)

    def init_metadata(self):
        with open(self.root_dir / 'metadata.yaml', 'r') as file:
            metadata = yaml.safe_load(file)
        self.grib_params = metadata["GRIB_PARAMS"]
        self.all_isobaric_levels = metadata["ISOBARIC_LEVELS_HPA"]
        self.all_weather_params = metadata["PARAMETERS"]
        self.all_grids = metadata["GRIDS"]

    def load_one_time_step(self, date:datetime) -> dict:
        sample = {}
        date_str = date.strftime(FORMATSTR)
        for grib_name, grib_keys in self.grib_params.items():
            names_wp = [key for key in grib_keys if key in self.hparams.weather_params]
            names_wp = [name.split("_")[1] for name in names_wp]
            if names_wp == []:
                continue
            levels = self.hparams.isobaric_levels if "ISOBARE" in grib_name else None
            path_grib = self.root_dir / "grib" / date_str / grib_name
            grib_plit = grib_name.split("_")
            prefix = self.all_grids[f"{grib_plit[0]}_{grib_plit[1]}"]["prefix"]
            dico_grib = read_grib(path_grib, names_wp, levels)
            for key in dico_grib.keys():
                sample[f"{prefix}_{key}"] = dico_grib[key]
        return sample

    def timestep_dict_to_array(self, dico:dict) -> torch.Tensor:
        tensors = []
        for wparam in self.hparams.weather_params:  # to keep order of params
            levels_dict = dico[wparam]
            if len(levels_dict.keys()) == 1:
                key0 = list(levels_dict.keys())[0]
                tensors.append(torch.from_numpy(levels_dict[key0]).float())
            else:
                for lvl in self.hparams.isobaric_levels:  # to keep order of levels
                    tensors.append(torch.from_numpy(levels_dict[lvl]).float())
        return torch.stack(tensors, dim=-1)

    def __getitem__(self, index):
        date_t0 = self.samples_dates[index]

        # ------- State features -------

        input_step, pred_step = self.hparams.input_step, self.hparams.pred_step

        input_dates = []
        for i in range(self.hparams.nb_input_steps):
            input_dates = [date_t0 - i * timedelta(hours=input_step)] + input_dates
        input_dicts = [self.load_one_time_step(date) for date in input_dates]
        input_tensors = [self.timestep_dict_to_array(dico) for dico in input_dicts]
        init_states = torch.stack(input_tensors, dim=0) # (steps, Nx, Ny, features)

        pred_dates = []
        for i in range(1, self.hparams.nb_pred_steps + 1):
            pred_dates.append(date_t0 + i * timedelta(hours=pred_step))
        pred_dicts = [self.load_one_time_step(date) for date in pred_dates]
        pred_tensors = [self.timestep_dict_to_array(dico) for dico in pred_dicts]
        target_states = torch.stack(pred_tensors, dim=0) # (steps, Nx, Ny, features)

        # TODO : stacker toutes les steps puis spliter en input et target
        # TODO : apply subgrid
        # TODO : accumuler variables cumulatives dans initstates et targetstates
        # TODO : flatten spatial dim dans initstates et targetstates
        # TODO : standardisation des variables

        # ------- Static features : only water coverage, on s'en fout ?

        # ------- Forcing features -------
        # TODO : flux, hour and year angles

        return init_states, target_states #, static_features, forcing_windowed


    def get_dataloader(self):
        return DataLoader(
            self, self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=self.hparams.shuffle
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
        latitudes = conf_ds.latitude[corners[0]: corners[1]]
        longitudes = conf_ds.longitude[corners[2]: corners[3]]
        grid = np.meshgrid(longitudes, latitudes)
        return grid

    @property
    def geopotential_info(self) -> np.array:
        conf_ds = xr.load_dataset(self.root_dir / "conf.grib")
        corners = self.hparams.sub_grid
        return conf_ds.h.values[:, corners[0]: corners[1], corners[2]: corners[3]]

    @property
    def limited_area(self) -> bool:
        return True

    @property
    def border_mask(self) -> np.array:
        border_mask = np.ones((self.shape[0], self.shape[1])).astype(bool)
        size = self.border_size
        border_mask[size: -size, size: -size]*=False
        return border_mask



if __name__=="__main__":
    hparams = TitanParams()
    dataset = TitanDataset(TitanParams)
    date  = datetime(2023, 3, 19, 12, 0)
    print('dataset.grid_info : ', dataset.grid_info)
    beg = time.time()
    x, y = dataset.__getitem__(3)
    print("time : ", time.time() - beg)
    print('x.shape : ', x.shape)
    print('y.shape : ', y.shape)
    loader = dataset.get_dataloader()
    print(loader)