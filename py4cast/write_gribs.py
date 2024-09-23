import cfgrib as cf
from cfgrib import xarray_to_grib as xtg
import xarray as xr
from copy import deepcopy
import numpy as np

file = "../grid.arome-forecast.eurw1s40+0000:00.grib"

print(cf.__version__)

isobaric_levels = xr.open_dataset(
    file,
    backend_kwargs={
        "indexpath": "",
        "read_keys": [
            "level",
            "shortname",
            "centre",
            "typeOfGeneratingProcess",
            "generatingProcessIdentifier",
        ],
        "filter_by_keys": {"level": [850.0, 900.0], "cfVarName": ["u", "v"]},
    },
)

raw_data = isobaric_levels["u"].to_numpy()
dims = isobaric_levels["u"].dims

new_data = np.ones_like(raw_data)
new_grib = deepcopy(isobaric_levels)
new_grib["u"] = (dims, new_data)
new_grib["u"] = new_grib.u.assign_attrs(isobaric_levels.u.attrs)

xtg.to_grib(new_grib, "../new_grid.arome-forecast.eurw1s40+0000:00.grib")

test_grib = cf.open_dataset(
    "../new_grid.arome-forecast.eurw1s40+0000:00.grib",
    backend_kwargs={
        "indexpath": "",
        "read_keys": ["level"],
        "filter_by_keys": {"level": [850.0, 900.0], "cfVarName": ["u", "v"]},
    },
)
u850 = test_grib.u.loc[{"isobaricInhPa": 850.0}].values
print(u850.shape, type(u850), u850.max())
