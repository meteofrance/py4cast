import os
from pathlib import Path

ROOTDIR = Path(os.environ.get("PY4CAST_ROOTDIR", "/scratch/shared/py4cast"))
CACHE_DIR = ROOTDIR / "cache"
DEFAULT_CONFIG_DIR = DEFAULT_CONFIG = Path(__file__).parents[1] / "config"
