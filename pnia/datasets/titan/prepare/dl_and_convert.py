import gzip
import tarfile
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from pnia.datasets.titan.utils import get_list_file_hendrix, retrieve_from_hendrix
from mfai.log_utils import get_logger
from pnia.datasets.titan.settings import GRIB_PARAMS, PARAMETERS, SCRATCH_PATH

this_module_name = Path(__file__).name
LOGGER = get_logger(this_module_name)

tar_dir = SCRATCH_PATH / "tar"
targz_dir = SCRATCH_PATH / "targz"
grib_dir = SCRATCH_PATH / "grib"
npy_dir = SCRATCH_PATH / "npy"
list_files_hendrix = get_list_file_hendrix()


def check_hour_done(date: datetime):
    hour_str = date.strftime("%Y-%m-%d_%Hh%M")
    hourly_npy_dir = npy_dir / hour_str
    if not hourly_npy_dir.exists():
        return False
    npy_files = hourly_npy_dir.glob("*.npy")
    nb_files = len(list(npy_files))
    nb_params = len(list(PARAMETERS.keys()))
    return nb_files == nb_params


def check_day_done(date: datetime):
    for hour in range(24):
        hour_date = date + hour * timedelta(hours=1)
        if not check_hour_done(hour_date):
            return False
    return True


def download_daily_archive(date):
    day_filename = date.strftime("%Y-%m-%d.tar")
    if not (tar_dir / day_filename).exists():
        LOGGER.info(f"Downloading file for date {date}")
        if day_filename not in list_files_hendrix:
            LOGGER.warning(f"File {day_filename} does not exist on Hendrix")
            return
        retrieve_from_hendrix(day_filename, tar_dir)
    return tar_dir / day_filename


def untar_daily_archive(tar: Path):
    list_targz = list(targz_dir.glob(f"{tar.stem}*.tar.gz"))
    if len(list_targz) != 24:
        LOGGER.info(f"Untar file {tar.name}")
        with tarfile.open(tar, "r") as tar_file:
            tar_file.extractall(targz_dir)


def unzip_hourly_archive(targz: Path, dest: Path):
    LOGGER.info(f"Unzipping archive {targz}...")
    dest.mkdir()
    with gzip.open(targz, "rb") as gz_file:
        with tarfile.open(fileobj=gz_file, mode="r") as tar:
            tar.extractall(dest)


def grib2npy(grib: Path, dest: Path):
    LOGGER.debug(f"Converting grib {grib}...")
    ds = xr.open_dataset(grib, engine="cfgrib")
    keys = GRIB_PARAMS[grib.name]
    ds_keys = [key.split("_")[1] for key in keys]
    for i in range(len(keys)):
        if (dest / f"{keys[i]}.npy").exists():
            continue
        try:
            array = ds[ds_keys[i]].values
        except KeyError:
            LOGGER.error(f"Key {ds_keys[i]} not available in grib !")
            continue
        np.save(dest / f"{keys[i]}.npy", array)


def process_hour(hour: datetime, convert=False):
    LOGGER.info(f"Processing hour {hour}...")
    hour_str = hour.strftime("%Y-%m-%d_%Hh%M")
    hourly_grib_dir = grib_dir / hour_str
    if not hourly_grib_dir.exists():
        unzip_hourly_archive(targz_dir / f"{hour_str}.tar.gz", hourly_grib_dir)

    if convert:
        hourly_npy_dir = npy_dir / hour_str
        hourly_npy_dir.mkdir(exist_ok=True)
        LOGGER.debug("Converting gribs...")
        for grib in hourly_grib_dir.glob("*.grib"):
            grib2npy(grib, hourly_npy_dir)


def process_day(date: datetime, convert=False):
    """Downloads data from Hendrix for a day and convert it to npy"""
    LOGGER.info(f"Processing day {date}...")

    if check_day_done(date):
        LOGGER.info(f"Day {date} already done !")
        return
    tar = download_daily_archive(date)
    untar_daily_archive(tar)

    for hour in range(24):
        hour_date = date + hour * timedelta(hours=1)
        if not check_hour_done(hour_date):
            process_hour(hour_date, convert)
        else:
            LOGGER.info(f"Hour {hour_date} already done !")

    tar.unlink()
    list_targz = targz_dir.glob(f"{tar.stem}*.tar.gz")
    for targz in list_targz:
        targz.unlink()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "start",
        type=str,
        help="Start date, format YYYYMMDD",
    )
    parser.add_argument(
        "end",
        type=str,
        help="End date, format YYYYMMDD",
    )
    parser.add_argument(
        "--convert_npy",
        action="store_true",
        help="Converts gribs to npy array.",
    )

    args = parser.parse_args()

    # Get list of dates
    d_start = datetime.strptime(args.start, "%Y%m%d")
    d_end = datetime.strptime(args.end, "%Y%m%d")
    nb_days = int((d_end - d_start).days + 1)
    list_days = [d_start + i * timedelta(days=1) for i in range(nb_days)]

    LOGGER.info(f"Downloading data for {nb_days} days : {d_start} to {d_end}")
    for day in list_days:
        process_day(day, args.convert_npy)
