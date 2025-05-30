#python routine to pull gfs data and create netcdf files based on startdate
#and rnday (# of days). Data written to gfs_{date}.nc files
from datetime import datetime
import argparse
import logging

from pyschism.mesh.hgrid import Hgrid
from pyschism.forcing.nws.nws2.gfs2 import GFS

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                    level=logging.INFO,
                    force=True)
cli_bool = lambda x: x.lower() in ['true', '1', 't', 'y', 'yes']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Generate surface forcing files from GFS data for SCHISM model")

    parser.add_argument('--hgrid',
                        type=str,
                        default='./hgrid.gr3',
                        help='Path to hgrid.gr3 file (default: ./hgrid.gr3)')
    parser.add_argument('--start-date',
                        type=datetime.fromisoformat,
                        help='Model start date in ISO format (YYYY-MM-DD)')
    parser.add_argument('--rnday',
                        type=float,
                        help='Model run duration in days (can be fractional)')
    parser.add_argument('--record',
                        type=int,
                        default=1,
                        help='Record interval in hours (default: 1)')
    parser.add_argument(
        '--pscr',
        type=str,
        default='.',
        help='Output directory for generated files (default: .)')

    parser.add_argument(
        '--use-tempdir',
        type=cli_bool,
        default=False,
        help=
        'Use temporary directory for generated files (default: True); if the files already exist, pass their path as --pscr and set --use-tempdir to False to avoid re-downloading the files.'
    )

    args = parser.parse_args()

    hgrid = Hgrid.open(args.hgrid, crs='epsg:4326')
    gfs = GFS(pscr=args.pscr, bbox=hgrid.bbox)
    logger.info(f'writing gfs data to {args.pscr}')
    logger.info(f'writing with args {args}')
    gfs.write(args.start_date,
              args.rnday,
              air=True,
              prc=True,
              rad=True,
              use_tempdir=bool(args.use_tempdir))
