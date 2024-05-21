import ssl
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import py4cast.datasets.titan.dataset as dataset

ssl._create_default_https_context = ssl._create_unverified_context


def plot_grid(sample, dataset):
    coord = dataset.all_grids["PAAROME_1S100"]["extend"]
    coord = (55.4, 37.5, -12.0, 16.0)
    ulat, llat, llon, ulon = coord
    fig = plt.figure(figsize=(10 * len(sample), 10), dpi=300)
    extent = (llon, ulon, llat, ulat)

    i = 1
    for wp, sample_dict in sample.items():
        ax = fig.add_subplot(1, len(sample), i, projection=ccrs.PlateCarree())
        key = [key for key in sample_dict.keys()]
        l, c = sample_dict[key[0]].shape
        sample_array = np.reshape(sample_dict[key[0]], (c, l))
        img = ax.imshow(sample_array, extent=extent, interpolation="none")
        ax.set_extent(extent)
        name = dataset.all_weather_params[wp]["name"]
        ax.set_title(f"{name}")
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.08)
        units = dataset.all_weather_params[wp]["units"]
        cbar.set_label(f"{units}")
        ax.coastlines(resolution="50m", color="black", linewidth=1)
        ax.gridlines(
            draw_labels=True, linewidth=0.5, alpha=0.4, color="k", linestyle="--"
        )
        i += 1
    plt.savefig("/scratch/labia/ferreiram/plot_wp.png")
    fig = plt.figure(figsize=(35, 10), dpi=300)
    te = dataset.geopotential_info
    ax = fig.add_subplot(1, 1, 1)
    img = ax.imshow(te, interpolation="none")
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.08)
    plt.savefig("/scratch/labia/ferreiram/test2.png")


hparams = dataset.TitanHyperParams()
dataset = dataset.TitanDataset(hparams)
date = datetime(2023, 3, 19, 12, 0)
sample = dataset.load_one_time_step(date)
print(sample)
plot_grid(sample, dataset)
