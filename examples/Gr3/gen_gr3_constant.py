import argparse
from pyschism.mesh.hgrid import Gr3


def parse_float_list(arg):
    """Parse comma-separated float values into a list."""
    return [float(x) for x in arg.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate constant GR3 files with specified values")

    parser.add_argument(
        '--hgrid',
        type=str,
        default='./hgrid.gr3',
        help='Path to input hgrid.gr3 file (default: ./hgrid.gr3)')
    parser.add_argument(
        '--names',
        type=str,
        default='albedo,diffmax,diffmin,watertype,windrot_geo2proj',
        help=
        'Comma-separated list of GR3 file names (default: albedo,diffmax,diffmin,watertype,windrot_geo2proj)'
    )
    parser.add_argument(
        '--values',
        type=parse_float_list,
        default=[0.1, 1.0, 1e-6, 1.0, 0.0],
        help=
        'Comma-separated list of constant values (default: 0.1,1.0,1e-6,1.0,0.0)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='./',
        help='Output directory for generated files (default: ./)')

    args = parser.parse_args()

    # Validate that names and values lists have the same length
    if len(args.names.split(',')) != len(args.values):
        raise ValueError("Number of names must match number of values")

    hgrid = Gr3.open(args.hgrid, crs='epsg:4326')
    grd = hgrid.copy()

    for name, value in zip(args.names.split(','), args.values):
        grd.description = name
        grd.nodes.values[:] = value
        grd.write(f'{args.outdir}/{name}.gr3', overwrite=True)
