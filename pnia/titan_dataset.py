import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from pnia.base import AbstractDataset
from cyeccodes import nested_dd_iterator
from cyeccodes.eccodes import get_multi_messages_from_file
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import yaml
from typing import List, Tuple
from pathlib import Path


@dataclass
class TitanParams:

    weather_params: Tuple[str] = ("aro_t2m", "aro_r2", "aro_z")
    isobaric_levels: Tuple[int] = (1000, 850)  # hPa
    nb_input_steps: int = 2
    input_step: int = 1  # hours
    nb_pred_steps: int = 4
    pred_step: int = 1  # hours
    sub_grid: Tuple[int] = (1000, 1256, 1200, 1456)  # grid corners (pixel), lat lon
    border_size: int = 10  # pixels
    # TODO : date_begining


def read_grib(path_grib: Path, names=None, levels=None):
    try:
        if names or levels:
            include_filters={k:v for k,v in [("cfVarName", names), ("level", levels)] if v is not None}
        else:
            include_filters = None
        _, results = get_multi_messages_from_file(path_grib, storage_keys=("cfVarName", "level"), include_filters=include_filters, metadata_keys=("missingValue", "Ni", "Nj"), include_latlon = False)
    except Exception as e:
        print(e)
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
        self.data_dir = self.root_dir / "grib"
        self.init_metadata()
        self.hparams = hparams
        corners = self.hparams.sub_grid
        self.shape = (corners[1] - corners[0], corners[3] - corners[2])


    def __len__(self):
        pass

    def init_metadata(self):
        with open(self.root_dir / 'metadata.yaml', 'r') as file:
            metadata = yaml.safe_load(file)
        self.grib_params = metadata["GRIB_PARAMS"]
        self.all_isobaric_levels = metadata["ISOBARIC_LEVELS_HPA"]
        self.all_weather_params = metadata["PARAMETERS"]
        self.all_grids = metadata["GRIDS"]

    def load_one_time_step(self, date:datetime):
        sample = {}
        date_str = date.strftime("%Y-%m-%d_%Hh%M")
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

    def __getitem__(self, index):
        pass

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
    sample = dataset.load_one_time_step(date)
    print("sample")
    print(sample)
    print('dataset.grid_info : ', dataset.grid_info)
