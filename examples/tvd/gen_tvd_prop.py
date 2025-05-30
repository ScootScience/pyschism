#!/usr/bin/env python3

import argparse
import os
from pathlib import Path


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description=
        "Generate TVD property file with configurable grid size and output directory"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help=
        "Output directory for the TVD property file (default: current directory)"
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=10000000,
        help="Number of grid points to generate (default: 10000000)")

    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the TVD property file
    output_file = output_dir / "tvd.prop"
    with open(output_file, "w") as file:
        for i in range(1, args.n_grid):
            file.write(f"{i} 0\n")


if __name__ == "__main__":
    main()
