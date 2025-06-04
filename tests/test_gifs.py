from datetime import datetime

import cartopy
import numpy as np

from py4cast.datasets.titan.settings import METADATA
from py4cast.utils import make_gif


def test_make_gif():
    NUM_OUTPUS = 3
    SIZE_IMG = 64
    for feature_name in ["aro_r2_2m", "aro_tp_0m"]:
        target_feat = np.random.rand(NUM_OUTPUS, SIZE_IMG, SIZE_IMG)
        list_preds_feat = [
            np.random.rand(NUM_OUTPUS, SIZE_IMG, SIZE_IMG) for _ in range(2)
        ]
        assert make_gif(
            feature_name,
            datetime(2024, 1, 1, 12).strftime("%Y%m%d%H"),
            target_feat,
            list_preds_feat,
            ["target", "model 1", "model 2"],
            cartopy.crs.PlateCarree(),
            (0, 10, 0, 10),
            METADATA,
        )
