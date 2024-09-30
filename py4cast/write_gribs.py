from copy import deepcopy
from time import perf_counter

import cfgrib as cf
import numpy as np
import xarray as xr
from cfgrib import xarray_to_grib as xtg

# grid.arome-forecast.eurw1s100+0001:00.grib
# grid.emul_aro_ai_dataset_poesy_eurw1s40_date2021-06-14 21:00:00_ech23.0.grib"
file = "/scratch/shared/py4cast/gribs_writing/grid.arome-forecast.eurw1s40+0001:00.grib"

print(cf.__version__)


for i in range(1):
    t0 = perf_counter()

    isobaric_levels = xr.open_dataset(
        file,
        backend_kwargs={
            "indexpath": "",
            "read_keys": [
                "level",
                "cfVarname",
                "shortname",
                "centre",
                "typeOfGeneratingProcess",
                "generatingProcessIdentifier",
                "typeOfLevel",
                "discipline",
                "parameterCategory",
                "unit",
            ],
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": [2]},
        },
    )

    print(isobaric_levels)

    t1 = perf_counter() - t0
    print(t1)


# print(isobaric_levels.surface.values)
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
