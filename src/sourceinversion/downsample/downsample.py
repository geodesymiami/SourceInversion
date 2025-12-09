#!/usr/bin/env python3
import re
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.interpolate import griddata
from mintpy.cli.save_kite import main as skite
from sourceinversion.shared.csv_functions import displacement_csv
from sourceinversion.downsample.objects.downsample_methods import Downsample
from sourceinversion.shared.helper_functions import convert_to_utm

EXAMPLE = """
        run_downsample.py --path CampiFlegrei --satellite Sen --method uniform --show
"""
SCRATCHDIR = os.getenv('SCRATCHDIR')


def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument('--path', type=str, nargs='+', required=True, help="Path to the folder or file.")
    parser.add_argument('--geometry', type=str, nargs='+', help="Path to the geometry file.")
    parser.add_argument('--satellite', type=str, nargs='+', default=['Sen'], help="Satellite names.")

    parser.add_argument('--method', choices=['uniform', 'quadtree'], default='uniform', help="Downsampling method.")
    parser.add_argument('--reduce', type=int, nargs='+', default=[3], help="Use masked velocity file.")
    parser.add_argument("--epsilon", type=float, default=0.0029, help="Epsilon value (default:  %(default)s)")
    parser.add_argument("--tile-size-max", type=float, default=0.02, help="Maximum tile size (default:  %(default)s)")
    parser.add_argument("--tile-size-min", type=float, default=0.002, help="Minimum tile size (default: %(default)s)")

    parser.add_argument('--period', nargs='*', metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD', type=str, help='Period of the search')

    parser.add_argument('--color-map', type=str, default='viridis', help="Colormap for plotting (default: %(default)s).")
    parser.add_argument('--style', choices=['scatter', 'image'], default='image', help="Plotting style (default: %(default)s).")
    parser.add_argument('--no-show', dest='show', action='store_false', help="Show the plot.")

    # Parse arguments
    inps = parser.parse_args()

    inps.folder_path = [folder if SCRATCHDIR in folder else os.path.join(SCRATCHDIR, folder) for folder in inps.path]

    inps.reduce = [inps.reduce[0], inps.reduce[0]] if len(inps.reduce)==1 else inps.reduce

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


def find_velocity_and_geometry(input_folder, period_folder):
    # Velocity file is in the period folder
    velocity_files = [os.path.join(period_folder, f) for f in os.listdir(period_folder) if 'velocity_msk.h5' in f]

    # If 'velocity_msk.h5' is not found, search for 'velocity.h5'
    if not velocity_files:
        velocity_files = [os.path.join(period_folder, f) for f in os.listdir(period_folder) if 'velocity.h5' in f]

    # Other files are in the parent folder
    geom_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if 'geometryRadar.h5' in f]

    if not velocity_files:
        raise FileNotFoundError(f"No velocity file found in period folder: {period_folder}")
    if not geom_files:
        raise FileNotFoundError(f"No geometryRadar.h5 found in input folder: {input_folder}")

    return velocity_files[0], geom_files[0]


def process_folder(velocity_file, geom_file, out_file, inps):
    kite_args = [velocity_file, "-d", "velocity", "-g", geom_file, "-o", out_file]

    if inps.method == 'uniform':
        down = Downsample(velocity_file=velocity_file, geometry_file=geom_file)
        down.uniform(reduction=inps.reduce[0])
        inps.reduce.pop(0)
        if inps.show:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            if inps.style == 'scatter':
                ax[0].scatter(down.x, down.y, c=down.z, s=1, cmap=inps.color_map)
            elif inps.style == 'image':
                ax[0].imshow(down.height, cmap='gray', extent=(np.nanmin(down.x), np.nanmax(down.x), np.nanmin(down.y), np.nanmax(down.y)), origin='upper')
                ax[0].imshow(down.imshow, cmap=inps.color_map, extent=(np.nanmin(down.x), np.nanmax(down.x), np.nanmin(down.y), np.nanmax(down.y)), origin='upper')

            #Interpolated result
            xx, yy = convert_to_utm(longitude=down.longitude, latitude=down.latitude)
            x = np.linspace(np.nanmin(xx), np.nanmax(xx), down.velocity.shape[1])
            y = np.linspace(np.nanmax(yy), np.nanmin(yy), down.velocity.shape[0])
            grid_x, grid_y = np.meshgrid(x, y)

            valid_mask = ~np.isnan(down.velocity)

            # Filter out NaN values from the input data
            valid_points = ~np.isnan(down.x) & ~np.isnan(down.y) & ~np.isnan(down.z)
            filtered_x = down.x[valid_points]
            filtered_y = down.y[valid_points]
            filtered_z = down.z[valid_points]

            # Perform interpolation with the filtered data
            interpolated_data = griddata((filtered_x, filtered_y), filtered_z, (grid_x, grid_y), method="linear")
            masked_data = np.where(valid_mask, interpolated_data, np.nan)

            ax[1].imshow(masked_data, cmap=inps.color_map, extent=(np.nanmin(down.x), np.nanmax(down.x), np.nanmin(down.y), np.nanmax(down.y)), origin='upper')
            ax[2].imshow(down.velocity, cmap=inps.color_map, extent=(np.nanmin(down.x), np.nanmax(down.x), np.nanmin(down.y), np.nanmax(down.y)), origin='upper')

            ax[0].set_title('Reduced', fontsize=16, pad=10)
            ax[1].set_title('Interpolated', fontsize=16, pad=10)
            ax[2].set_title('Original', fontsize=16, pad=10)

    elif inps.method == 'quadtree':
        skite(kite_args)
        down = Downsample(velocity_file=velocity_file[0], kite_file=out_file + '.yml', geometry_file=geom_file[0])
        down.quadtree(epsilon=inps.epsilon, tile_size_max=inps.tile_size_max, tile_size_min=inps.tile_size_min)
        if inps.show:
            fig = plt.figure()
            if inps.style == 'scatter':
                ax = fig.add_subplot(111)
                ax.scatter(down.x, down.y, c=down.z, s=1, cmap=inps.color_map)
            elif inps.style == 'image':
                ax = fig.gca()

                limit = np.abs(down.qt.leaf_medians).max()
                color_map = cm.ScalarMappable(
                    norm=colors.Normalize(vmin=-limit, vmax=limit),
                    cmap=cm.get_cmap(inps.color_map))

                for rect, leaf in zip(down.qt.getMPLRectangles(), down.qt.leaves):
                    color = color_map.to_rgba(leaf.median)
                    rect.set_facecolor(color)
                    ax.add_artist(rect)

                ax.set_xlim(down.qt.leaf_eastings.min(), down.qt.leaf_eastings.max())
                ax.set_ylim(down.qt.leaf_northings.min(), down.qt.leaf_northings.max())


    # Save the downsampled data
    displacement_csv(file=out_file, x=down.x, y=down.y, z=down.z, err=down.err, lose=down.lose, losn=down.losn, losz=down.losz)


def main(iargs=None):
    print("-" * 50)
    print("Starting Decomposition Module...")
    print("-" * 50)
    print()

    inps = create_parser() if not isinstance(iargs, argparse.Namespace) else iargs

    if inps.satellite:
        p = '|'.join([f"{s}[AD]T?" for s in inps.satellite])

    if len(inps.path)==1:
        pattern = f"({p})\d+"
        regex = re.compile(pattern)
        folder_list = [f for f in os.listdir(inps.folder_path[0]) if os.path.isdir(os.path.join(inps.folder_path[0], f))]
        for folder in folder_list:
            # Search for the keyword in the path
            match = regex.match(folder)

            if match:
                node = match.group(0)
                input_folder = os.path.join(inps.folder_path[0], node)

                # If periods are specified, process each period folder
                if inps.period_folder:
                    for period in inps.period_folder:
                        period_folder = os.path.join(input_folder, period)
                        if not os.path.exists(period_folder):
                            print(f"Period folder {period_folder} does not exist.")
                            continue

                        out_file = os.path.join(period_folder, inps.path[0] + node)
                        velocity_file, geom_file = find_velocity_and_geometry(input_folder, period_folder)
                        process_folder(velocity_file, geom_file, out_file, inps)
                else:
                    # Process the main folder as usual
                    out_file = os.path.join(input_folder, inps.path[0] + node)
                    velocity_file, geom_file = find_velocity_and_geometry(input_folder, input_folder)
                    process_folder(velocity_file, geom_file, out_file, inps)

    else:
        pattern = f"({p})\d+"
        regex = re.compile(pattern)
        for f in inps.path:
            velocity_file = []
            geom_file = None
            if 'velocity.h5' in f or 'velocity_msk.h5' in f:
                velocity_file.append(f)
                for g in os.listdir(os.path.dirname(f)):
                    if 'geometryRadar.h5' in g:
                        geom_file = os.path.join(os.path.dirname(f), g)
                        break
                if not geom_file:
                    for g in os.listdir(os.path.dirname(os.path.dirname(f))):
                        if 'geometryRadar.h5' in g:
                            geom_file = os.path.join(os.path.dirname(os.path.dirname(f)), g)
                            break
            else:
                for v in os.listdir(f):
                    if 'velocity.h5' in v or 'velocity_msk.h5' in v:
                        velocity_file.append(os.path.join(f, v))
                    if 'geometryRadar.h5' in f:
                        geom_file = f
                if not geom_file:
                    for g in os.listdir(os.path.dirname(f)):
                        if 'geometryRadar.h5' in g:
                            geom_file = os.path.join(os.path.dirname(f), g)
                            break

            velocity_file = next((v for v in velocity_file if '_msk' in os.path.basename(v)), velocity_file[0])

            match = regex.search(velocity_file)
            base = velocity_file.split(match.group(0))[0]
            name = os.path.basename(os.path.normpath(base)) if base else os.path.basename(os.path.normpath(velocity_file))
            out_file = os.path.join(os.path.dirname(velocity_file), name + match.group(0))
            process_folder(velocity_file, geom_file, out_file, inps)

    if inps.show:
        plt.show()


if __name__ == '__main__':
    main(iargs=sys.argv)
