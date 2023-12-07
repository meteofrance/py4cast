import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from pnia.base import AbstractDataset
from cyeccodes import nested_dd_iterator
from cyeccodes.eccodes import get_multi_messages_from_file
from datetime import datetime, timedelta
from pnia.settings import TITAN_DIR
import yaml
from typing import List
from pathlib import Path

def read_grib(path_grib: Path, grid, names=None, levels=None):
    try:
        if names or levels:
            include_filters={k:v for k,v in [("shortName", names), ("level", levels)] if v is not None}
        else:
            include_filters = None
        _, results = get_multi_messages_from_file(path_grib, storage_keys=("shortName", "level"), include_filters=include_filters, metadata_keys=("missingValue",), include_latlon = False)
    except Exception as e:
        print(e)
    for metakey, result in nested_dd_iterator(results):
        array = result["values"]
        mv = result["metadata"]["missingValue"]
        array = np.reshape(array, grid)
        array = np.where(array == mv, np.nan, array)
        name, level = metakey.split("-")
        level = int(level)
        results[name][level] = array
    return results


class TitanDataset(AbstractDataset, Dataset):
    def __init__(self, weather_params:List[str], isobaric_levels:List[int]) -> None:
        self.ROOT_DIR = Path("/scratch/shared/Titan/")
        self.init_metadata()
        self.weather_params = weather_params
        self.isobaric_levels = isobaric_levels
        self.shape = 256
        self.min_x = 500
        self.min_y = 500
        self.border_size = 10

    def __len__(self):
        pass

    def init_metadata(self):
        with open('config.yml', 'r') as file:
            metadata = yaml.safe_load(TITAN_DIR / "metadata.yaml")
        self.grib_params = metadata["GRIB_PARAMS"]
        self.all_isobaric_levels = metadata["ISOBARIC_LEVELS_HPA"]
        self.all_weather_params = metadata["PARAMETERS"]
        self.all_grids = metadata["GRIDS"]

    def get_grid_and_prefix(self, grib):
        if "ANTJP7" in grib:
            return self.all_grids["Antilope"]["size"], "ant"
        if "PAAROME_1S100" in grib:
            return self.all_grids["Arome 1S100"]["size"], "aro"
        if "PAAROME_1S40" in grib:
            return self.all_grids["Arome 1S40"]["size"], "aro"
        if "PA_" in grib:
            return self.all_grids["Arpege"]["size"], "arp"

    def load_one_time_step(self, date:datetime):
        sample = {}
        date_str = date.strptime("%Y-%m-%d_%Hh%M")
        for grib_name, grib_keys in self.grib_params:
            names_wp = [key for key in grib_keys if key in self.weather_params]
            names_wp = [name.split("_")[1] for name in names_wp]
            if names_wp == []:
                continue
            levels = self.isobaric_levels if "ISOBARE" in grib_name else None
            path_grib = self.ROOT_DIR / "grib" / date_str / grib_name
            grid, prefix = self.get_grid_and_prefix(grib_name)
            dico_grib = read_grib(path_grib, grid, names_wp, levels)
            sample[f"{prefix}_{key}"] = dico_grib[key]
            for key in dico_grib.keys():
                sample[f"{prefix}_{key}"] = dico_grib[key]
        return sample

    def __getitem__(self, index):
        pass

    @property
    def grid_info(self) -> np.array:
        conf_ds = xr.load_dataset(self.ROOT_DIR / "conf.grib")
        latitudes = conf_ds.latitude
        longitudes = conf_ds.longitude
        return np.meshgrid(longitudes, latitudes)[self.min_x: self.min_x + self.shape, self.min_y: self.min_y + self.shape]

    @property
    def geopotential_info(self) -> np.array:
        conf_ds = xr.load_dataset(self.ROOT_DIR / "conf.grib")
        return conf_ds.h.values[self.min_x: self.min_x + self.shape, self.min_y: self.min_y + self.shape]

    @property
    def limited_area(self) -> bool:
        return True

    @property
    def border_mask(self) -> np.array:
        border_mask = np.ones((self.shape, self.shape)).astype(bool)
        border_mask[self.border_size: -self.border_size, self.border_size: -self.border_size]*=False
        return border_mask

if __name__=="__main__":
    dataset = TitanDataset(["aro_t2m", "aro_r2"], [1000, 850])

    
