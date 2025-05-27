#python routine to pull gfs data and create netcdf files based on startdate
#and rnday (# of days). Data written to gfs_{date}.nc files
from datetime import datetime
import argparse

from pyschism.mesh.hgrid import Hgrid
from pyschism.forcing.nws.nws2.gfs2 import GFS

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

    args = parser.parse_args()

    hgrid = Hgrid.open(args.hgrid, crs='epsg:4326')
    gfs = GFS(start_date=args.start_date,
              rnday=args.rnday,
              pscr=args.pscr,
              record=args.record,
              bbox=hgrid.bbox)
