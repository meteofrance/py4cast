import numpy as np
from torch.utils.data import DataLoader, Dataset
from pnia.base import AbstractDataset
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
        self.init_metadata()
        self.weather_params = weather_params
        self.isobaric_levels = isobaric_levels

    def __len__(self):
        pass

    def init_metadata(self):
        with open('config.yml', 'r') as file:
            metadata = yaml.safe_load(TITAN_DIR / "metadata.yaml")
        self.grib_params = metadata["GRIB_PARAMS"]
        self.all_isobaric_levels = metadata["ISOBARIC_LEVELS_HPA"]
        self.all_weather_params = metadata["PARAMETERS"]
        self.grid_shape = metadata["AROME_1S100_GRID"]

    @staticmethod
    def load_one_time_step(self, date:datetime):
        sample = {}
        for grib_name, grib_keys in self.grib_params:
            names_wp = [key for key in grib_keys if key in self.weather_params]
            names_wp = [name.split("_")[1] for name in names_wp]
            if names_wp == []:
                continue
            levels = self.isobaric_levels if "ISOBARE" in grib_name else None
            dico_grib = read_grib(path_sample / grib_name, get_grid(grib_name), names_wp, levels)
            for key in dico_grib.keys():
                if "ANT" in grib_name:
                    sample["ant_" + key] = dico_grib[key]
                if "PAAROME_" in grib_name:
                    sample["aro_" + key] = dico_grib[key]
                if "PA_" in grib_name:
                    sample["arp_" + key] = dico_grib[key]
        return sample

    def __getitem__(self, index):
        pass

    @property
    def grid_info(self) -> np.array:
        pass

    @property
    def geopotential_info(self) -> np.array:
        pass
