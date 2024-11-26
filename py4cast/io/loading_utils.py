
import datetime as dt
import numpy as np
from typing import Callable, List, Literal
from skimage.transform import resize

from py4cast.datasets.base import (
    Param
)

def fit_to_grid(
    param: Param, arr: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    get_grid_coords:  Callable[[], List[str]]
) -> np.ndarray:
    # already on good grid, nothing to do:
    if param.grid.name == param.native_grid:
        return arr

    # crop arpege data to arome domain:
    if param.native_grid == "PA_01D" and param.grid.name in [
        "PAAROME_1S100",
        "PAAROME_1S40",
    ]:
        grid_coords = get_grid_coords()
        # Mask ARPEGE data to AROME bounding box
        mask_lon = (lons >= grid_coords[2]) & (lons <= grid_coords[3])
        mask_lat = (lats >= grid_coords[1]) & (lats <= grid_coords[0])
        arr = arr[mask_lat, :][:, mask_lon]

    anti_aliasing = param.grid.name == "PAAROME_1S40"  # True if downsampling
    # upscale or downscale to grid resolution:
    return resize(arr, param.grid.full_size, anti_aliasing=anti_aliasing)

def load_data_grib(self, param: Param, date: dt.datetime) -> np.ndarray:
    path_grib = param.get_filepath(date, "grib")
    ds = read_grib(path_grib)
    assert param.grib_param is not None
    level_type = ds[param.grib_param].attrs["GRIB_typeOfLevel"]
    lats = ds.latitude.values
    lons = ds.longitude.values
    if level_type != "isobaricInhPa":  # Only one level
        arr = ds[param.grib_param].values
    else:
        arr = ds[param.grib_param].sel(isobaricInhPa=self.level).values
    return arr, lons, lats

def load_data(
    param: Param, date: dt.datetime, file_format: Literal["npy", "grib"] = "grib"
):
    if file_format == "grib":
        arr, lons, lats = load_data_grib(date)
        arr = fit_to_grid(arr, lons, lats)
        subdomain = param.grid.subdomain
        arr = arr[subdomain[0] : subdomain[1], subdomain[2] : subdomain[3]]
        return arr[::-1]  # invert latitude
    else:
        return np.load(get_filepath(param, date, file_format))

def exist(param: Param, date: dt.datetime,
          get_filepath: Callable[[Param,dt.datetime,str],Path],
          file_format: Literal["npy", "grib"] = "grib") -> bool:
    filepath = get_filepath(param, date, file_format)
    return filepath.exists()