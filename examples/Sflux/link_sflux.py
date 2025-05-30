from datetime import datetime, timedelta
import os
import argparse
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                    level=logging.INFO,
                    force=True)
# logging.getLogger("pyschism").setLevel(logging.DEBUG)


def parse_args():
    """
    Parse command line arguments for the sflux linking script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=
        "Link sflux files from source directory to destination directory")

    parser.add_argument(
        "--srcdir",
        type=str,
        required=True,
        help="Source data directory containing subfolders named as YYYYMMDD")

    parser.add_argument(
        "--dstdir",
        type=str,
        required=True,
        help="Destination directory path (run/sflux directory)")

    parser.add_argument("--start-date",
                        type=str,
                        required=True,
                        help="Start date in YYYY-MM-DD format")

    parser.add_argument("--rnday",
                        type=int,
                        required=True,
                        help="Number of days to process")

    return parser.parse_args()


def link_sflux_files(srcdir: str, dstdir: str, start_date: datetime,
                     rnday: int):
    """
    Link sflux files from source to destination directory.
    
    Args:
        srcdir (str): Source data directory
                Should have subfolder named as "20220201", "20220202"        
        dstdir (str): Destination directory path
        start_date (datetime): Start date for processing
        rnday (int): Number of days to process
    """
    vars = ["air", "prc", "rad"]

    timevector = np.arange(start_date, start_date + timedelta(days=rnday),
                           timedelta(days=1)).astype(datetime)
    print(f"Processing dates: {timevector}")

    for i, date in enumerate(timevector):
        srcgfs = f"{srcdir}/{date.strftime('%Y%m%d')}/gfs_{date.strftime('%Y%m%d%H')}.nc"
        srchrrr = f"{srcdir}/{date.strftime('%Y%m%d')}/hrrr_{date.strftime('%Y%m%d%H')}.nc"

        for var in vars:
            try:
                dstgfs = f"{dstdir}/sflux_{var}_1.{str(i+1).zfill(4)}.nc"
                os.symlink(srcgfs, dstgfs)
            except FileExistsError:
                logger.info(f"File {dstgfs} already exists, skipping")
            try:
                dsthrrr = f"{dstdir}/sflux_{var}_2.{str(i+1).zfill(4)}.nc"
                os.symlink(srchrrr, dsthrrr)
            except FileExistsError:
                logger.info(f"File {dsthrrr} already exists, skipping")


def main():
    """
    Main function to handle command line arguments and execute sflux linking.
    """
    args = parse_args()
    logger.info(f"args: {args}")
    # Parse start_date string to datetime object
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date must be in YYYY-MM-DD format")

    # Create destination directory if it doesn't exist
    os.makedirs(args.dstdir, exist_ok=True)

    link_sflux_files(args.srcdir, args.dstdir, start_date, args.rnday)


if __name__ == "__main__":
    main()
