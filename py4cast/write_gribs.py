import cfgrib as cf
from cfgrib import xarray_to_grib as xtg
import xarray as xr
from copy import deepcopy
import numpy as np

file = "/scratch/shared/Titan/AROME/2023061812/AROME_1S100_ECH0_10M.grib"

print(cf.__version__)

isobaric_levels = xr.open_dataset(
    file,
    backend_kwargs={
        "indexpath": "",
        },
)
print(isobaric_levels)
#print(isobaric_levels.surface.values)
"""raw_data = isobaric_levels["tirf"].to_numpy()
dims = isobaric_levels["tirf"].dims
print(dims)
new_data = np.ones_like(raw_data)
new_grib = deepcopy(isobaric_levels)
new_grib = new_grib.assign(
    {"tp" : (
        (dims,new_data)
    )},
    )
print(new_grib.tp)
#new_grib["tp"] = new_grib.tp.assign_attrs(isobaric_levels.tirf.attrs)

xtg.to_grib(new_grib, "../new_grid.arome-forecast.eurw1s100+0000:00.grib")

test_grib = cf.open_dataset(
    "../new_grid.arome-forecast.eurw1s100+0000:00.grib",
    backend_kwargs={
        "indexpath": "",
        "read_keys": ["level"],
        "filter_by_keys": {"level": [0], "cfVarName": ["tp"]},
    },
)
#print(test_grib)"""
