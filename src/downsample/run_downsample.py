#!/usr/bin/env python3
import re
import os
import sys
import argparse

import matplotlib.pyplot as plt
from mintpy.cli.save_kite import main as skite
from src.shared.csv_functions import displacement_csv
from src.downsample.objects.downsample import Downsample

EXAMPLE = """
        run_downsample.py --folder CampiFlegrei --satellite Sen --method uniform --show
"""
SCRATCHDIR = os.getenv('SCRATCHDIR')


def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder.")
    parser.add_argument('--satellite', type=str, nargs='+', default=['Sen'], help="Satellite names.")
    parser.add_argument('--method', choices=['uniform', 'quadtree'], default='uniform', help="Downsampling method.")
    parser.add_argument('--downsample-factor', type=int, default=3, help="Reduce the number of pixels for uniform method(default:  %(default)s).")
    parser.add_argument("--epsilon", type=float, default=0.0029, help="Epsilon value for quadtree method (default:  %(default)s)")
    parser.add_argument("--tile-size-max", type=float, default=0.02, help="Maximum tile size for quadtree method (default:  %(default)s)")
    parser.add_argument("--tile-size-min", type=float, default=0.002, help="Minimum tile size for quadtree method (default: %(default)s)")
    parser.add_argument('--show', action='store_true', help="Show the plot.")
    parser.add_argument('--period', nargs='*', metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD', type=str, help='Period of the search')

    # Parse arguments
    inps = parser.parse_args()

    inps.folder_path = inps.folder if SCRATCHDIR in inps.folder else os.path.join(SCRATCHDIR, inps.folder)

    if inps.period:
        inps.period_folder = []
        for p in inps.period:
            delimiters = '[,:\-\s]'
            dates = re.split(delimiters, p)

            if len(dates[0]) and len(dates[1]) != 8:
                msg = 'Date format not valid, it must be in the format YYYYMMDD'
                raise ValueError(msg)

            inps.period_folder.append(f"{dates[0]}_{dates[1]}")

    else:
        inps.period_folder = []

    return inps


def process_folder(input_folder, period_folder, node, out_file, inps):
    # Velocity file is in the period folder
    velocity_file = [os.path.join(period_folder, f) for f in os.listdir(period_folder) if 'velocity_msk.h5' in f]

    # Other files are in the parent folder
    mask_file = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if 'maskTempCoh.h5' in f]
    geom_file = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if 'geometryRadar.h5' in f]

    kite_args = [velocity_file[0], "-d", "velocity", "-g", geom_file[0], "-o", out_file]

    if inps.method == 'uniform':
        down = Downsample(velocity_file=velocity_file[0], geometry_file=geom_file[0])
        down.uniform(reduction=inps.reduce)

    elif inps.method == 'quadtree':
        skite(kite_args)
        down = Downsample(velocity_file=velocity_file[0], kite_file=out_file + '.yml', geometry_file=geom_file[0])
        down.quadtree(epsilon=inps.epsilon, tile_size_max=inps.tile_size_max, tile_size_min=inps.tile_size_min)

    # Save the downsampled data
    displacement_csv(file=out_file, x=down.x, y=down.y, z=down.z, err=down.err, lose=down.lose, losn=down.losn, losz=down.losz)

    if inps.show:
        fig, ax = plt.subplots()
        ax.scatter(down.x, down.y, c=down.z, s=1)
        plt.show()


def main(iargs=None):
    print("#" * 50)
    print("Starting Decomposition Module...")
    print("#" * 50)
    print()

    inps = create_parser() if not isinstance(iargs, argparse.Namespace) else iargs

    if inps.satellite:
        p = '|'.join([f"{s}[AD]T?" for s in inps.satellite])

    pattern = f"({p})\d+"
    regex = re.compile(pattern)
    folder_list = [f for f in os.listdir(inps.folder_path) if os.path.isdir(os.path.join(inps.folder_path, f))]

    for folder in folder_list:
        # Search for the keyword in the path
        match = regex.match(folder)

        if match:
            node = match.group(0)
            input_folder = os.path.join(inps.folder_path, node)

            # If periods are specified, process each period folder
            if inps.period_folder:
                for period in inps.period_folder:
                    period_folder = os.path.join(input_folder, period)
                    if not os.path.exists(period_folder):
                        print(f"Period folder {period_folder} does not exist.")
                        continue

                    out_file = os.path.join(period_folder, inps.folder + node)
                    process_folder(input_folder, period_folder, node, out_file, inps)
            else:
                # Process the main folder as usual
                out_file = os.path.join(input_folder, inps.folder + node)
                process_folder(input_folder, input_folder, node, out_file, inps)


if __name__ == '__main__':
    main(iargs=sys.argv)
