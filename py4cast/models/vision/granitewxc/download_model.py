import wget
from py4cast.models.vision.granitewxc.utils.config import get_config
from py4cast.settings import ROOTDIR
import os
from pathlib import Path

MODEL_DIR = ROOTDIR / "models" / "granite_wxc"
config_path = os.path.join(MODEL_DIR, 'config.yaml')

wget.download('https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling/resolve/main/config.yaml', out = config_path)
config = get_config(config_path)

config.download_path = os.path.join(MODEL_DIR, 'pytorch_model.bin')
wget.download('https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling/resolve/main/pytorch_model.bin', out = config.download_path)