"""Scan GRIB files to get weather parameters dictionary"""

from pathlib import Path

import xarray as xr
import yaml
from tqdm import tqdm

from py4cast.datasets.titan.settings import GRIDS, ISOBARIC_LEVELS_HPA


def get_prefixes(grib_name):
    if "ANTJP7" in grib_name:
        return "ant", "Antilope"
    elif "PA_" in grib_name:
        return "arp", "Arpege"
    elif "PAAROME_" in grib_name:
        return "aro", "Arome"
    else:
        return


path = Path("/scratch/shared/Titan/grib/2023-03-19_12h00/")
grib_files = sorted(list(path.glob("*.grib")))


param_dict = {}
grib_dict = {}

for grib in tqdm(grib_files):
    print("--------------")
    print(grib.name)
    ds = xr.open_dataset(grib, engine="cfgrib")

    grib_dict[grib.name] = []

    key_pref, name_pref = get_prefixes(grib.stem)
    for var in ds.data_vars:
        print("----->", var)
        print(ds[var].attrs)
        levels = ds[var][ds[var].GRIB_typeOfLevel].values
        print("levels : ", levels)
        print(type(levels))
        key = f"{key_pref}_{var}"
        name = f"{name_pref} {ds[var].long_name}"
        grib_plit = grib.name.split("_")
        grid_name = f"{grib_plit[0]}_{grib_plit[1]}"
        param_dict[key] = {
            "name": key,
            "long_name": name,
            "param": var,
            "model": name_pref,
            "prefix_model": key_pref,
            "unit": ds[var].units,
            "cumulative": ds[var].GRIB_stepType == "accum",
            "type_level": ds[var].GRIB_typeOfLevel,
            "levels": levels.astype(int).tolist(),
            "grib": grib.name,
            "grid": grid_name,
            "shape": GRIDS[grid_name]["size"],
            "extent": GRIDS[grid_name]["extent"],
        }
        print(param_dict[key])
        grib_dict[grib.name].append(key)

print(param_dict)
print(grib_dict)

yaml_dict = {
    "GRIDS": GRIDS,
    "ISOBARIC_LEVELS_HPA": ISOBARIC_LEVELS_HPA,
    "GRIB_PARAMS": grib_dict,
    "WEATHER_PARAMS": param_dict,
}
with open("/scratch/shared/Titan/metadata.yaml", "w") as file:
    documents = yaml.dump(yaml_dict, file)
