from glob import glob
import yaml

"""
write metadata from gribs in a given directory machting a given pattern
checking files with grib extension in the directory
loading each grib and sorting all data in a common yaml file
"""

file_pattern_registry = {
    "grid.{model}-forecast.{domain}+{leadtime}.grib": ["model", "domain", "leadtime"],
    "PA{model}_{domain}_ECH{leadtime}_{levelkw}.grib": [
        "model",
        "domain",
        "leadtime",
        "levelkw",
    ],
    "PA_01D_{variable}.grib": ["variable"],
    "ANTJP7CLIM_{domain}_60_SOL.grib": ["domain"],
}
