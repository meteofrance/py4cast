import json
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from py4cast.datasets.titan.settings import METADATA


def plot_scores(features: List[str], data: dict, max_timestep: int = 12) -> None:
    """Plots one frame of the animation."""

    lines = int(math.sqrt(len(features)))
    cols = len(features) // lines
    if len(features) % lines != 0:
        cols += 1

    if (lines, cols) == (1, 3):
        figsize = (12, 5)
    elif (lines, cols) == (2, 2):
        figsize = (4 * cols, 4 * lines)
    else:
        figsize = (4 * cols, 5 * lines)

    fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=200)
    axes = fig.subplots(nrows=lines, ncols=cols)
    axs = axes.flat

    for i, feature in enumerate(features):
        max_rmse = 0
        for model in data.keys():
            values = data[model][feature][:max_timestep]
            if max(values) > max_rmse:
                max_rmse = max(values)
            axs[i].plot(range(1, len(values) + 1), values, label=model)
            feature_name = "_".join(feature.split("_")[:2])
            name = METADATA["WEATHER_PARAMS"][feature_name]["long_name"]
            axs[i].set_title(name)
            axs[i].set_ylim(bottom=0, top=max_rmse)
            axs[i].set_xlabel("Leadtime (h)")
            if i == 0:
                axs[i].legend()

    fig.suptitle("RMSE on Test set per leadtime")
    copyright = "Météo-France, Py4cast project."
    fig.text(0.4, 0.01, copyright, fontsize=8, ha="left")
    plt.savefig("rmse_scores.png")


if __name__ == "__main__":
    parser = ArgumentParser("Plot animations")
    parser.add_argument(
        "--ckpt",
        type=str,
        action="append",
        help="Paths to the AI model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--num_pred_steps",
        type=int,
        default=12,
        help="Number of auto-regressive steps/prediction steps.",
    )
    args = parser.parse_args()

    scores = {}
    models_names = []
    for ckpt in args.ckpt:
        path = Path(ckpt)
        with open(path.parent / "Test_rmse_scores.json", "r") as json_file:
            loaded_data = json.load(json_file)
        model_name = f"{path.parent.name}"
        scores[model_name] = loaded_data

    plot_scores(
        ["aro_t2m_2m", "aro_tp_0m", "aro_r2_2m", "aro_u10_10m"],
        scores,
        args.num_pred_steps,
    )
