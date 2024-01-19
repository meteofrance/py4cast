import os
import shutil
import subprocess
import tarfile
from argparse import ArgumentParser
from datetime import datetime, timedelta

from utils import get_list_file_hendrix, upload_to_hendrix
from joblib import Parallel, delayed
from tqdm import trange
from pathlib import Path


class EmptyGribFile(Exception):
    "Raised when the desired GRIB file is empty"
    pass


def download_1_bdap_request(request, date, working_dir, grib_dir):
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
    grib_file = working_dir / request.with_suffix(".grib").name
    if not grib_file.exists():
        raise FileNotFoundError(grib_file.name)
    if grib_file.stat().st_size < 1000:
        raise EmptyGribFile()

    # Move to grib folder
    shutil.move(grib_file, grib_dir / grib_file.name)


def download_1h_gribs(date, working_dir, grib_dir):
    request_path = working_dir / "bdap_requests"
    requests_arp = list(request_path.glob("PA_*.txt"))
    requests_other = list(request_path.glob("*.txt"))
    requests_other = [req for req in requests_other if req not in requests_arp]
    Parallel(n_jobs=2)(
        delayed(download_1_bdap_request)(request, date, working_dir, grib_dir) for request in requests_other
    )
    # As env_var DMT_DATE_PIVOT changes for ARPEGE, we do not parallelize
    for request in requests_arp:
        download_1_bdap_request(request, date, working_dir, grib_dir)
    (working_dir / "DIAG_BDAP").unlink()


def compress_1h(output_file: str, grib_dir):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(grib_dir, arcname=".")
    for f in grib_dir.glob("*.grib"):
        f.unlink()  # Remove grib files


def tar_1day(day_filename, tar_dir):
    with tarfile.open(day_filename, "w") as tar:
        tar.add(tar_dir, arcname=".")
    for f in tar_dir.glob("*.tar.gz"):
        f.unlink()


def main(list_days, working_dir):
    """Downloads data for a list of days and send archive to Hendirx"""
    grib_dir = pwd / "grib"
    tar_dir = pwd / "tar"
    grib_dir.mkdir(exist_ok=True)
    tar_dir.mkdir(exist_ok=True)
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
            if (tar_dir / filename_targz_1h).exists():
                continue
            download_1h_gribs(dateh, working_dir, grib_dir)
            compress_1h(filename_targz_1h, working_dir / "grib")
            shutil.move(filename_targz_1h, tar_dir / filename_targz_1h)

        # Tar all 24 files for one day
        tar_1day(day_filename, tar_dir)

        print(f"Sending file {day_filename} to Hendrix...")
        upload_to_hendrix(working_dir / day_filename)
        (working_dir / day_filename).unlink()


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

    pwd = Path(__file__).parents[0]

    grib_dir = pwd / "grib"
    tar_dir = pwd / "tar"
    grib_dir.mkdir(exist_ok=True)
    tar_dir.mkdir(exist_ok=True)

    # Get list of dates
    d_start = datetime.strptime(args.start, "%Y%m%d")
    d_end = datetime.strptime(args.end, "%Y%m%d")
    nb_days = int((d_end - d_start).days + 1)
    list_days = [d_start + i * timedelta(days=1) for i in range(nb_days)]
    print(f"Downloading data for {nb_days} days : {d_start} to {d_end}")

    main(list_days, pwd)
