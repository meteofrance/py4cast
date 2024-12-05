import os
from pathlib import Path

ROOTDIR = Path(os.environ.get("PY4CAST_ROOTDIR", "/scratch/shared/py4cast"))
CACHE_DIR = ROOTDIR / "cache"