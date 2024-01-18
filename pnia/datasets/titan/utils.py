import netrc
from argparse import ArgumentParser
from ftplib import FTP
from pathlib import Path

from mfai.log_utils import get_logger
from settings import BASE_PATH, HENDRIX, HENDRIX_PATH, NETRC_PATH

this_module_name = Path(__file__).name
LOGGER = get_logger(this_module_name)


def login_hendrix(ftp: FTP):
    netrc_inst = netrc.netrc(NETRC_PATH)
    creds = netrc_inst.authenticators(HENDRIX)
    ftp.login(user=creds[0], passwd=creds[2])
    ftp.cwd(str(HENDRIX_PATH))


def upload_to_hendrix(src_file: Path):
    with FTP(HENDRIX) as ftp:
        login_hendrix(ftp)
        with open(src_file, "rb") as file:
            ftp.storbinary(f"STOR {src_file.name}", file)


def get_list_file_hendrix():
    with FTP(HENDRIX) as ftp:
        login_hendrix(ftp)
        file_list = ftp.nlst()
    return file_list


def retrieve_from_hendrix(filename, dst_dir):
    with FTP(HENDRIX) as ftp:
        login_hendrix(ftp)
        with open(dst_dir / filename, "wb") as file:
            ftp.retrbinary("RETR " + filename, file.write)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("filename", type=str, help="Filename on Hendrix")
    args = parser.parse_args()

    print(f"Retrieving file from hendrix : {BASE_PATH / args.filename}")
    retrieve_from_hendrix(args.filename)
