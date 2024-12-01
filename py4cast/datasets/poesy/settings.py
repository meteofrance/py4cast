from pathlib import Path

import yaml

SCRATCH_PATH = Path("/scratch/shared/poesy/poesy_crop")
OROGRAPHY_FNAME = "PEARO_EURW1S40_Orography_crop.npy"
LATLON_FNAME = "latlon_crop.npy"

# Shape of cropped poesy data (lon x lat x leadtimes x members)
DATA_SHAPE = (600, 600, 45, 16)

# Assuming no leap years in dataset (2024 is next)
SECONDS_IN_YEAR = 365 * 24 * 60 * 60

with open(Path(__file__).parents[0] / "metadata.yaml", "r") as file:
    METADATA = yaml.safe_load(file)

DEFAULT_CONFIG = Path(__file__).parents[3] / "config/datasets/poesy_refacto.json"
