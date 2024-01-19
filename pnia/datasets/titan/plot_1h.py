from argparse import ArgumentParser
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
from settings import SCRATCH_PATH

ulat, llat, llon, ulon = 55.4, 37.5, -12, 16
EXTENT = (llon, ulon, llat, ulat)

ulat, llat, llon, ulon = 72, 20, -32, 42
EXTENT_PA = (llon, ulon, llat, ulat)


def gribs_to_netcdf():
    pass


def plot_var(fig, ax, var, name="", arpege=False):
    extent = EXTENT_PA if arpege else EXTENT
    ax.imshow(var, extent=extent, interpolation="none")
    if name == "":
        name = f"{var.attrs['GRIB_name']} ({var.attrs['units']})"
    ax.set_title(name)
    # fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def plot_arome_1s100(folder: Path):
    fig, axs = plt.subplots(
        2, 5, figsize=(22, 10), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    grib = folder / "PAAROME_1S100_ECH0_2M.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 0], ds.t2m)  # "Température à 2m (K)"
    plot_var(fig, axs[0, 1], ds.r2)  # Humidité Relative à 2m (%

    grib = folder / "PAAROME_1S100_ECH0_10M.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 2], ds.u10)  # Composante U du vent à 10m (m/s)
    plot_var(fig, axs[0, 3], ds.v10)  # Composante V du vent à 10m (m/s)

    grib = folder / "PAAROME_1S100_ECH0_SOL.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 4], ds.sd)  # Snow depth water equivalent (kg/m²)

    grib = folder / "PAAROME_1S100_ECH1_10M.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[1, 0], ds.ugust)  # Composante U des raffales (m/s)
    plot_var(fig, axs[1, 1], ds.vgust)  # Composante V des raffales (m/s)

    grib = folder / "PAAROME_1S100_ECH1_SOL.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[1, 2], ds.tp)  # Précipitations cumulées (mm; HS)
    plot_var(fig, axs[1, 3], ds.tirf)  # Time integral of rain flux (mm; HS)
    plot_var(fig, axs[1, 4], ds.sprate)  # Snow precipitation rate (mm/s; HS)

    for i in range(2):
        for j in range(5):
            axs[i, j].coastlines(resolution="50m", color="black", linewidth=1)
            axs[i, j].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")

    plt.suptitle(ds.time.values)
    plt.tight_layout()
    plt.show()


def plot_arome_1s40(folder: Path):
    fig, axs = plt.subplots(
        3, 5, figsize=(22, 8), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    grib = folder / "PAAROME_1S40_ECH0_MER.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 0], ds.prmsl)

    grib = folder / "PAAROME_1S40_ECH0_SOL.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 1], ds.tciwv)

    grib = folder / "PAAROME_1S40_ECH0_ISOBARE.grib"
    ds = xr.open_dataset(grib, engine="cfgrib").sel(isobaricInhPa=500)
    for i, var in enumerate(ds.keys()):
        attrs = ds[var].attrs
        x, y = (i + 2) // 5, (i + 2) % 5
        name = f"{attrs['GRIB_name']} at 500hPa ({attrs['units']})"
        plot_var(fig, axs[x, y], ds[var], name)

    grib = folder / "PAAROME_1S40_ECH1_SOL.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[2, 3], ds.str)
    plot_var(fig, axs[2, 4], ds.ssr)

    for i in range(3):
        for j in range(5):
            axs[i, j].coastlines(resolution="50m", color="black", linewidth=1)
            axs[i, j].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")

    plt.suptitle(ds.time.values)
    plt.tight_layout()
    plt.show()


def plot_arpege_eurat01(folder: Path):
    fig, axs = plt.subplots(
        2, 5, figsize=(22, 8), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    grib = folder / "PA_01D_2M.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 0], ds.t2m, arpege=True)
    plot_var(fig, axs[0, 1], ds.r2, arpege=True)

    grib = folder / "PA_01D_10M.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 2], ds.u10, arpege=True)
    plot_var(fig, axs[0, 3], ds.v10, arpege=True)

    grib = folder / "PA_01D_MER.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    plot_var(fig, axs[0, 4], ds.prmsl, arpege=True)

    grib = folder / "PA_01D_ISOBARE.grib"
    ds = xr.open_dataset(grib, engine="cfgrib").sel(isobaricInhPa=500)
    plot_var(fig, axs[1, 0], ds.z, arpege=True)
    plot_var(fig, axs[1, 1], ds.t, arpege=True)
    plot_var(fig, axs[1, 2], ds.u, arpege=True)
    plot_var(fig, axs[1, 3], ds.v, arpege=True)
    plot_var(fig, axs[1, 4], ds.r, arpege=True)

    for i in range(2):
        for j in range(5):
            axs[i, j].coastlines(resolution="50m", color="black", linewidth=1)
            axs[i, j].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")

    plt.suptitle(ds.time.values)
    plt.tight_layout()
    plt.show()


def plot_antilope(folder: Path):
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ulat, llat, llon, ulon = 51.5, 41, -6, 10.5
    extent = (llon, ulon, llat, ulat)

    grib = folder / "ANTJP7CLIM_1S100_60_SOL.grib"
    ds = xr.open_dataset(grib, engine="cfgrib")
    img = ax.imshow(ds.prec, extent=extent, interpolation="none")
    name = f"{ds.prec.attrs['GRIB_name']} ({ds.prec.attrs['units']})"
    ax.set_title(name)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")

    plt.suptitle(ds.time.values)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--folder",
        type=Path,
        default=SCRATCH_PATH / "grib",
        help="Folder where 1h gribs are stored.",
    )
    args = parser.parse_args()

    print("Plot AROME 1S100")
    plot_arome_1s100(args.folder)
    print("Plot AROME 1S40")
    plot_arome_1s40(args.folder)
    print("Plot ARPEGE EURAT01")
    plot_arpege_eurat01(args.folder)
    print("Plot ANTILOPE")
    plot_antilope(args.folder)
