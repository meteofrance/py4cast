from pathlib import Path

import yaml

SCRATCH_PATH = Path("/scratch/shared/Titan")
FORMATSTR = "%Y-%m-%d_%Hh%M"

with open(SCRATCH_PATH / "metadata.yaml", "r") as file:
    METADATA = yaml.safe_load(file)

DEFAULT_CONFIG = Path(__file__).parents[3] / "config" / "titan.json"
