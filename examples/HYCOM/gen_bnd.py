from datetime import datetime
import logging
import argparse

from pyschism.mesh.hgrid import Hgrid
from pyschism.forcing.hycom.hycom2schism import OpenBoundaryInventory, get_raw_hycom
'''
outputs:
    elev2D.th.nc (elev=True)
    SAL_3D.th.nc (TS=True)
    TEM_3D.th.nc (TS=True)
    uv3D.th.nc   (UV=True)
'''

logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger('pyschism')
logger.setLevel(logging.INFO)


def parse_boundary_ids(arg):
    """Parse comma-separated boundary IDs into a list of integers."""
    return [int(x) for x in arg.split(',')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Generate boundary conditions from HYCOM data for SCHISM model")

    parser.add_argument('--hgrid',
                        type=str,
                        default='./hgrid.gr3',
                        help='Path to hgrid.gr3 file (default: ./hgrid.gr3)')
    parser.add_argument('--vgrid',
                        type=str,
                        default='./vgrid.in',
                        help='Path to vgrid.in file (default: ./vgrid.in)')
    parser.add_argument(
        '--outdir',
        type=str,
        default='./',
        help='Output directory for generated files (default: ./)')
    parser.add_argument('--start-date',
                        type=datetime.fromisoformat,
                        help='Model start date in ISO format (YYYY-MM-DD)')
    parser.add_argument('--rnday',
                        type=float,
                        help='Model run duration in days (can be fractional)')
    parser.add_argument(
        '--ocean-bnd-ids',
        type=parse_boundary_ids,
        # default=[0, 1, 2],
        help='Comma-separated list of ocean boundary IDs (e.g., "0,1,2")')
    parser.add_argument(
        '--elev2d',
        action='store_true',
        default=True,
        help='Generate elevation boundary conditions (default: True)')
    parser.add_argument(
        '--ts',
        action='store_true',
        default=True,
        help=
        'Generate temperature and salinity boundary conditions (default: True)'
    )
    parser.add_argument(
        '--uv',
        action='store_true',
        default=True,
        help='Generate velocity boundary conditions (default: True)')
    parser.add_argument(
        '--forecast-mode',
        action='store_true',
        default=False,
        help='Generate boundary conditions from forecast mode (default: False)'
    )
    parser.add_argument('--archive-data',
                        action='store_true',
                        default=False,
                        help='Archive data (default: False)')
    parser.add_argument('--timeout-seconds',
                        type=int,
                        default=15,
                        help='Timeout in seconds (default: 15)')
    parser.add_argument('--forecast_length_hours',
                        type=int,
                        default=15,
                        help='forecast_length_hours')
    parser.add_argument('--forecast_freq_hours',
                        type=int,
                        default=15,
                        help='forecast_freq_hours')
    args = parser.parse_args()

    hgrid = Hgrid.open(args.hgrid, crs='epsg:4326')
    bnd = OpenBoundaryInventory(hgrid, args.vgrid)
    try:
        bnd.fetch_data(args.outdir,
                       args.start_date,
                       args.rnday,
                       elev2D=args.elev2d,
                       TS=args.ts,
                       UV=args.uv,
                       ocean_bnd_ids=args.ocean_bnd_ids,
                       forecast_mode=bool(args.forecast_mode),
                       archive_data=bool(args.archive_data),
                       timeout_seconds=args.timeout_seconds,
                       forecast_length_hours=args.forecast_length_hours,
                       forecast_freq_hours=args.forecast_freq_hours)
    except Exception as e:
        logger.error(f'Error fetching data: {e}')
        raise e
