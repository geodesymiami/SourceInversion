#!/usr/bin/env python3

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from src.shared.csv_functions import read_csv, displacement_csv 
from src.simulation.simulate import main as simulate
from src.inversion.run_inversion import main as inversion


EXAMPLE = """
        run_simulation.py --folder CampiFlegrei --satellite Sen --show
"""
SCRATCHDIR = os.getenv('SCRATCHDIR')


def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder.")
    parser.add_argument('--satellite', type=str, default=None, choices=['Sen', 'Csk'], help="Satellite name.")
    parser.add_argument('--txt-file', type=str, default=None , help="Path of the template file.")
    parser.add_argument('--shear', type=float, default=0.5, help="Shear value (default: %(default)s).")
    parser.add_argument('--poisson', type=float, dest='nu', default=0.25, help="Poisson ratio (default: %(default)s).")
    parser.add_argument('--x-range', type=float, nargs=2, default=[float('inf'), float('-inf')], help="X range.")
    parser.add_argument('--y-range', type=float, nargs=2, default=[float('inf'), float('-inf')], help="Y range.")
    parser.add_argument('--z-range', type=float, nargs=2, default=(0, 5000), help="Z range (default: %(default)s).")
    parser.add_argument('--volume', type=str, default='1.e6 2.e7', help="Volume value (default: %(default)s).")
    parser.add_argument('--sampling_id', type=str, choices=['0', '1'], default='0', help="Sampling ID (default: %(default)s).")
    parser.add_argument('--weight-sar', type=float, default=1.0, help="Weight for SAR data (default: %(default)s).")
    parser.add_argument('--weight-gps', type=float, default=0.0, help="Weight for GPS data (default: %(default)s).")
    parser.add_argument('--model', type=str, nargs='+', choices=['mogi', 'point', 'penny', 'spheroid', 'moment', 'okada'], default=['mogi'], help="One or more models: Mogi (1958), McTigue point source (1987), Fialko et al.(2001), Penny-shaped crack, Yang et al. (1988). Spheroid, Davis (1986) Moment tensor, Okada 1985.")
    parser.add_argument('--show', action='store_true', help="Show the plot.")
    parser.add_argument('--noise', type=float, default=0.0, help="Noise value (default: %(default)s).")
    parser.add_argument('--period', nargs='*', metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD', type=str, help='Period of the search')

    # Parse arguments
    inps = parser.parse_args()

    inps.folder_path = inps.folder if SCRATCHDIR in inps.folder else os.path.join(SCRATCHDIR, inps.folder)

    inps.params = os.path.join(inps.folder_path, 'VSM_best.csv')

    if inps.txt_file:
        if os.path.dirname(inps.txt_file) == '':
            inps.txt_file = os.path.join(inps.folder_path, inps.txt_file)
    else:
        inps.txt_file = os.path.join(inps.folder_path, 'simulation', 'VSM_input.txt')

    if inps.satellite and inps.weight_sar == 0.0:
        inps.weight_sar = 1.0

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


def main(iargs=None):

    print("#" * 50)
    print("Starting Simulation Module...")
    print("#" * 50)
    print()

    inps = create_parser()

    if inps.satellite:
        p = '|'.join([f"{inps.satellite}[AD]T?"])
        input_sar = ''

        pattern = f"({p})\d+"
        regex = re.compile(pattern)
        folder_list = [f for f in os.listdir(inps.folder_path) if os.path.isdir(os.path.join(inps.folder_path, f))]
        simulation_folder = os.path.join(inps.folder_path, 'simulation')

        for folder in folder_list:
            # Search for the keyword in the path
            match = regex.match(folder)

            if match:
                for f in os.listdir(os.path.join(inps.folder_path, folder)):
                    if f.endswith('.csv') and match.group(0) in f:
                        input_sar += os.path.join(inps.folder_path, folder, f) + ' '
                        out_folder = os.path.join(simulation_folder, folder)
                        os.makedirs(out_folder, exist_ok=True)

                        df = pd.read_csv(os.path.join(inps.folder_path, folder, f))
                        parameters = read_csv(inps.params)

                        ux, uy, uz = simulate(x=df['xx'], y=df['yy'], paramters=parameters, **inps.__dict__)

                        utot = np.array([ux, uy, uz])
                        los_sar = np.array([df['lx'], df['ly'], df['lz']]).T
                        displacement = np.sum(utot.T * los_sar, axis=1)

                        if inps.noise > 0:
                            displacement = displacement + np.random.normal(0, inps.noise, size=displacement.shape)

                        simulation_csv = displacement_csv(file=os.path.join(out_folder,f), x=df['xx'], y=df['yy'], z=displacement, err=df['ee'], lose=df['lx'], losn=df['ly'], losz=df['lz'])

                        if inps.show:
                            fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 5))
                            ax.scatter(df['xx'], df['yy'], c=displacement, s=3)
                            ax.set_title('Simulation')
                            ax1.scatter(df['xx'], df['yy'], c=df['dd'], s=3)
                            ax1.set_title('Observed')
                            plt.show()

    inps.folder_path=simulation_folder
    inversion(iargs=inps)

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    sim = pd.read_csv(os.path.join(inps.folder_path, 'VSM_best.csv'))
    inf = pd.read_csv(os.path.join((inps.folder_path).replace('simulation', ''), 'VSM_best.csv'))

    print("#" * 50)
    print("Inverted simulation results:")
    print(f"xcen: {sim['xcen'].values}\nycen: {sim['ycen'].values}\ndepth: {sim['depth'].values}\ndVol: {sim['dVol'].values}\n")
    print("#" * 50)
    print("Inverted Observed results:")
    print(f"xcen: {inf['xcen'].values}\nycen: {inf['ycen'].values}\ndepth: {inf['depth'].values}\ndVol: {inf['dVol'].values}\n")
    print("#" * 50)
    print("Difference:")
    print(f"xcen: {sim['xcen'].values - inf['xcen'].values}\nycen: {sim['ycen'].values - inf['ycen'].values}\ndepth: {sim['depth'].values - inf['depth'].values}\ndVol: {sim['dVol'].values - inf['dVol'].values}\n")

if __name__ == '__main__':
    main(iargs=sys.argv)