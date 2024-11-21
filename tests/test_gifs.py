from datetime import datetime

import cartopy
import numpy as np

from bin.gif_comparison import make_gif
from py4cast.plots import DomainInfo


def test_make_gif():
    NUM_OUTPUS = 3
    SIZE_IMG = 64
    for feature_name in ["aro_r2_2m", "aro_tp_0m"]:
        target_feat = np.random.rand(NUM_OUTPUS, SIZE_IMG, SIZE_IMG)
        list_preds_feat = [
            np.random.rand(NUM_OUTPUS, SIZE_IMG, SIZE_IMG) for _ in range(2)
        ]
        info = DomainInfo((0, 10, 0, 10), cartopy.crs.PlateCarree())
        make_gif(
            feature_name,
            datetime(2024, 1, 1, 12).strftime("%Y%m%d%H"),
            target_feat,
            list_preds_feat,
            ["target", "model 1", "model 2"],
            info,
        )


if __name__ == "__main__":
    test_make_gif()
