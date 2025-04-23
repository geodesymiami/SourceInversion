#!/usr/bin/env python3
import re
import os
import sys
import argparse

import matplotlib.pyplot as plt
from mintpy.cli.save_kite import main as skite
from src.shared.csv_functions import displacement_csv
from src.decomposition.objects.downsample import Downsample

EXAMPLE = """
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
    parser.add_argument('--reduce', type=int, default=3, help="Use masked velocity file.")
    parser.add_argument("--epsilon", type=float, default=0.0029, help="Epsilon value (default:  %(default)s)")
    parser.add_argument("--tile-size-max", type=float, default=0.02, help="Maximum tile size (default:  %(default)s)")
    parser.add_argument("--tile-size-min", type=float, default=0.002, help="Minimum tile size (default: %(default)s)")
    parser.add_argument('--show', action='store_true', help="Show the plot.")

    # Parse arguments
    inps = parser.parse_args()

    inps.folder_path = inps.folder if SCRATCHDIR in inps.folder else os.path.join(SCRATCHDIR, inps.folder)

    return inps


def main(iargs=None):
    inps = create_parser()

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
            out_file = os.path.join(input_folder, inps.folder + node)
            velocity_file = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if 'velocity_msk.h5' in f]
            mask_file = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if 'maskTempCoh.h5' in f]
            geom_file = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if 'geometryRadar.h5' in f]
            kite_args = [velocity_file[0], "-d", "velocity", "-g", geom_file[0], "-o", out_file]

            if inps.method == 'uniform':
                down = Downsample(velocity_file=velocity_file[0])
                down.uniform(reduction=inps.reduce)

            elif inps.method == 'quadtree':
                skite(kite_args)
                down = Downsample(velocity_file=velocity_file[0],kite_file=out_file + '.yml')
                down.quadtree(epsilon=inps.epsilon, tile_size_max=inps.tile_size_max, tile_size_min=inps.tile_size_min,)

            # Save the downsampled data
            displacement_csv(file=out_file, x=down.x, y=down.y, z=down.z, err=down.err, lose=down.lose, losn=down.losn, losz=down.losz)

            if inps.show:
                fig, ax = plt.subplots()
                ax.scatter(down.x, down.y, c=down.z, s=1)
                plt.show()


if __name__ == '__main__':
    main(iargs=sys.argv)