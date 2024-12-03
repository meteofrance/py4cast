import os
from pathlib import Path

import yaml

SCRATCH_PATH = Path(os.environ.get("PY4CAST_TITAN_PATH", "/scratch/shared/Titan"))
AROME_PATH = SCRATCH_PATH / "AROME"
FORMATSTR = "%Y-%m-%d_%Hh%M"

with open(Path(__file__).parents[0] / "metadata.yaml", "r") as file:
    METADATA = yaml.safe_load(file)

DEFAULT_CONFIG = Path(__file__).parents[3] / "config/datasets/titan_refacto.json"
