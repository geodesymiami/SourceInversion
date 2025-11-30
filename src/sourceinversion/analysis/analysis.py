#!/usr/bin/env python3
import re
import os
import sys
import argparse
from mintpy.utils import readfile
from src.sourceinversion.analysis import analysis_methods as am


EXAMPLE = ''
SCRATCHDIR = os.getenv('SCRATCHDIR')


def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder.")
    parser.add_argument('--satellite', type=str, nargs='+', default=['Sen'], help="Satellite names.")
    parser.add_argument('--method', choices=['spatial_correlation', 'centroid_shift'], default='uniform', help="Downsampling method.")
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


def get_file(path):
    # Velocity file is in the period folder
    velocity_file = [os.path.join(path, f) for f in os.listdir(path) if 'velocity_msk.h5' in f]

    # If 'velocity_msk.h5' is not found, search for 'velocity.h5'
    if not velocity_file:
        velocity_file = [os.path.join(path, f) for f in os.listdir(path) if 'velocity.h5' in f]

    return velocity_file[0]


def main(iargs=None):
    print("-" * 50)
    print("Starting Analysis Module...")
    print("-" * 50)
    print()

    inps = create_parser() if not isinstance(iargs, argparse.Namespace) else iargs

    if inps.satellite:
        p = '|'.join([f"{s}[AD]T?" for s in inps.satellite])

    pattern = f"({p})\d+"
    regex = re.compile(pattern)
    folder_list = [f for f in os.listdir(inps.folder_path) if os.path.isdir(os.path.join(inps.folder_path, f))]
    files = {}

    for folder in folder_list:
        # Search for the keyword in the path
        match = regex.match(folder)

        if match:
            node = match.group(0)
            input_folder = os.path.join(inps.folder_path, node)
            files[node] = {}

            # If periods are specified, process each period folder
            if inps.period_folder:
                for period in inps.period_folder:
                    files[node][period] = {}
                    period_folder = os.path.join(input_folder, period)
                    if not os.path.exists(period_folder):
                        print(f"Period folder {period_folder} does not exist.")
                        continue
                    out_file = os.path.join(period_folder, inps.folder + node)
                    vel_file = get_file(period_folder)

                    files[node][period]['velocity'] = vel_file
            else:
                # Process the main folder as usual
                out_file = os.path.join(input_folder, inps.folder + node)
                vel_file = get_file(input_folder)
                files[node] = vel_file

    if inps.method == 'spatial_correlation':
        print('-' * 50)
        print("Performing spatial correlation analysis...\n")
        for key in files.keys():
            data = []
            print(f"Processing node: {key}")
            for period in files[key]:
                velocity_file = files[key][period]['velocity']
                print(f"Reading velocity map for period: {period} from file: {velocity_file}")
                data.append(readfile.read(velocity_file)[0])
            print('-' * 50)
            print(f"Comparing maps for node: {key}\n")
            am.spatial_correlation_maps(*data)

    if inps.method == "centroid_shift":
        pass


if __name__ == '__main__':
    main(iargs=sys.argv)