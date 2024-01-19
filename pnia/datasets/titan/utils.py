import netrc
from argparse import ArgumentParser
from ftplib import FTP
from pathlib import Path


NETRC_PATH = Path.home() / ".netrc"
HENDRIX = "hendrix.meteo.fr"
HENDRIX_PATH = Path("/home/berthomierl/Titan")


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
    pwd = Path(__file__).parents[0]

    print(f"Retrieving file from hendrix : {pwd / args.filename}")
    retrieve_from_hendrix(args.filename)
