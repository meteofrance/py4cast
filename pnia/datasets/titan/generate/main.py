import os
import shutil
import subprocess
import tarfile
from argparse import ArgumentParser
from datetime import datetime, timedelta

from pnia.datasets.titan.utils import get_list_file_hendrix, upload_to_hendrix
from joblib import Parallel, delayed
from settings import BASE_PATH, GRIB_PATH, REQUESTS_PATH, TAR_PATH
from tqdm import trange


class EmptyGribFile(Exception):
    "Raised when the desired GRIB file is empty"
    pass


def download_1_bdap_request(request, date):
    subprocess_kwargs = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }

    # AROME ou ANTILOPE : dispo à chaque heure
    if "PA_" not in request.name:
        os.environ["DMT_DATE_PIVOT"] = date.strftime("%Y%m%d%H%M%S")
        process_dap3 = subprocess.Popen(  # nosec B602
            f"dap3_dev_echeance {request}", **subprocess_kwargs  # nosec B602
        )  # nosec B602
        process_dap3.wait(timeout=600)

    # ARPEGE : échéance et heure de réseau changent selon l'heure demandée
    else:
        leadtime = timedelta(hours=date.hour % 6)
        runtime = date - leadtime
        leadtime = int(leadtime.seconds / 3600)
        os.environ["DMT_DATE_PIVOT"] = runtime.strftime("%Y%m%d%H%M%S")
        process_dap3 = subprocess.Popen(  # nosec B602
            f"dap3_dev {leadtime} {request}", **subprocess_kwargs  # nosec B602
        )  # nosec B602
        process_dap3.wait(timeout=600)

    # Check file exists and not empty
    grib_file = BASE_PATH / request.with_suffix(".grib").name
    if not grib_file.exists():
        raise FileNotFoundError()
    if grib_file.stat().st_size < 1000:
        raise EmptyGribFile()

    # Move to grib folder
    shutil.move(grib_file, GRIB_PATH / grib_file.name)


def download_1h_gribs(date):
    requests_arp = list(REQUESTS_PATH.glob("PA_*.txt"))
    requests_other = list(REQUESTS_PATH.glob("*.txt"))
    requests_other = [req for req in requests_other if req not in requests_arp]
    Parallel(n_jobs=2)(
        delayed(download_1_bdap_request)(request, date) for request in requests_other
    )
    # As env_var DMT_DATE_PIVOT changes for ARPEGE, we do not parallelize
    for request in requests_arp:
        download_1_bdap_request(request, date)
    (BASE_PATH / "DIAG_BDAP").unlink()


def compress_1h(output_file: str):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(GRIB_PATH, arcname=".")
    for f in GRIB_PATH.glob("*.grib"):
        f.unlink()  # Remove grib files


def tar_1day(day_filename):
    with tarfile.open(day_filename, "w") as tar:
        tar.add(TAR_PATH, arcname=".")
    for f in TAR_PATH.glob("*.tar.gz"):
        f.unlink()


def main(list_days):
    """Downloads data for a list of days and send archive to Hendirx"""
    for date in list_days:
        print("date : ", date)

        # If archive already on Hendrix, skip
        day_filename = date.strftime("%Y-%m-%d.tar")
        if day_filename in get_list_file_hendrix():
            print("already done")
            continue

        for hour in trange(24):
            dateh = date + hour * timedelta(hours=1)
            filename_targz_1h = dateh.strftime("%Y-%m-%d_%Hh%M.tar.gz")
            if (TAR_PATH / filename_targz_1h).exists():
                continue
            download_1h_gribs(dateh)
            compress_1h(filename_targz_1h)
            shutil.move(filename_targz_1h, TAR_PATH / filename_targz_1h)

        # Tar all 24 files for one day
        tar_1day(day_filename)

        print(f"Sending file {day_filename} to Hendrix...")
        upload_to_hendrix(BASE_PATH / day_filename)
        (BASE_PATH / day_filename).unlink()


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

    args = parser.parse_args()

    GRIB_PATH.mkdir(exist_ok=True)
    TAR_PATH.mkdir(exist_ok=True)

    # Get list of dates
    d_start = datetime.strptime(args.start, "%Y%m%d")
    d_end = datetime.strptime(args.end, "%Y%m%d")
    nb_days = int((d_end - d_start).days + 1)
    list_days = [d_start + i * timedelta(days=1) for i in range(nb_days)]
    print(f"Downloading data for {nb_days} days : {d_start} to {d_end}")

    main(list_days)
