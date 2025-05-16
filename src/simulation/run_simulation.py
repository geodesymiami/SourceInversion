#!/usr/bin/env python3

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    parser.add_argument('--satellite', type=str, default='Sen', choices=['Sen', 'Csk'], help="Satellite name.")
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

    parser.add_argument('--mogi-volume', type=float, nargs=2, default=[1e6, 2e7], help="Mogi volume range (default: %(default)s).")

    # Penny parameters
    parser.add_argument('--penny-radius', type=float, nargs=2, default=[800, 800], help="Penny radius range (default: %(default)s).")
    parser.add_argument('--penny-dp_mu', type=float, nargs=2, default=[0.0001, 0.01], help="Penny dp/mu range (default: %(default)s).")

    # Spheroid example
    parser.add_argument('--spheroid-strike', type=float, nargs=2, default=[0, 360], help="Spheroid strike range (default: %(default)s).")
    parser.add_argument('--spheroid-dip', type=float, nargs=2, default=[0, 90], help="Spheroid dip range (default: %(default)s).")
    parser.add_argument('--spheroid-axis-ratio', type=float, nargs=2, default=[0.5, 1], help="Spheroid axis ratio range (default: %(default)s).")
    parser.add_argument('--spheroid-semi-axis', type=float, nargs=2, default=[500, 3000], help="Spheroid semi-axis range (default: %(default)s).")
    parser.add_argument('--spheroid-dp_mu', type=float, nargs=2, default=[0.0001, 0.01], help="Spheroid dp/mu range (default: %(default)s).")

    # Okada / Dislocation (model id = 5 R)
    parser.add_argument('--okada-length', type=float, nargs=2, default=[1000, 5000], help="Fault length range (meters) (default: %(default)s).")
    parser.add_argument('--okada-width', type=float, nargs=2, default=[1000, 5000], help="Fault width range (meters) (default: %(default)s).")
    parser.add_argument('--okada-strike', type=float, nargs=2, default=[0, 360], help="Strike angle range (degrees) (default: %(default)s).")
    parser.add_argument('--okada-dip', type=float, nargs=2, default=[0, 90], help="Dip angle range (degrees) (default: %(default)s).")
    parser.add_argument('--okada-slip', type=float, nargs=2, default=[0, 10], help="Slip amount range (meters) (default: %(default)s).")
    parser.add_argument('--okada-rake', type=float, nargs=2, default=[0, 0], help="Rake angle range (degrees) (default: %(default)s).")
    parser.add_argument('--okada-opening', type=float, nargs=2, default=[0.0, 1.0], help="Opening displacement range (meters) (default: %(default)s).")


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


def generate_displacement(inps, fpath, out_folder, params):
    df = pd.read_csv(fpath)
    parameters = read_csv(params)

    ux, uy, uz = simulate(x=df['xx'], y=df['yy'], paramters=parameters, **inps.__dict__)
    utot = np.array([ux, uy, uz])
    los_sar = np.array([df['lx'], df['ly'], df['lz']]).T
    displacement = np.sum(utot.T * los_sar, axis=1)

    if inps.noise > 0:
        displacement += np.random.normal(0, inps.noise, size=displacement.shape)

    displacement_csv(
        file=os.path.join(out_folder, os.path.basename(fpath)),
        x=df['xx'], y=df['yy'], z=displacement,
        err=df['ee'], lose=df['lx'], losn=df['ly'], losz=df['lz']
    )

    if inps.show:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 5))
        ax.scatter(df['xx'], df['yy'], c=displacement, s=3)
        ax.set_title('Simulation')
        ax1.scatter(df['xx'], df['yy'], c=df['dd'], s=3)
        ax1.set_title('Observed')
        plt.show()

def compare(sim_out_folder):
    sim = pd.read_csv(os.path.join(sim_out_folder, 'VSM_best.csv'))
    inf = pd.read_csv(os.path.join(sim_out_folder.replace('simulation', ''), 'VSM_best.csv'))

    print("#" * 50)
    print("Inverted simulation results:")
    for key in sim.keys():
        print(f"{key}: {sim[key].values}")
    print("#" * 50)
    print("Inverted Observed results:")
    for key in inf.keys():
        print(f"{key}: {inf[key].values}")
    print("#" * 50)
    print("Difference:")
    for s,i in zip(sim.keys(), inf.keys()):
        print(f"{s}: {sim[s].values - inf[i].values}")

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

        periods = inps.period_folder if inps.period_folder else [None]

        for period in periods:
            for folder in folder_list:
                match = regex.match(folder)
                if not match:
                    continue

                input_folder = os.path.join(inps.folder_path, folder)
                period_folder = os.path.join(input_folder, period) if period else input_folder
                output_folder = os.path.join(inps.folder_path, period) if period else inps.folder_path
                params = os.path.join(output_folder, 'VSM_best.csv')

                if not inps.txt_file:
                    if period:
                        inps.txt_file = os.path.join(inps.folder_path, 'simulation', period, 'VSM_input.txt')
                    else:
                        inps.txt_file = os.path.join(inps.folder_path, 'simulation', 'VSM_input.txt')

                sim_out_folder = os.path.join(simulation_folder, period) if period else simulation_folder
                simulation_input = os.path.join(simulation_folder, folder,  period) if period else os.path.join(simulation_folder, folder)
                # os.makedirs(sim_out_folder, exist_ok=True)  ALREADY CREATED IN inversion
                os.makedirs(simulation_input, exist_ok=True)

                for f in os.listdir(period_folder):
                    if f.endswith('.csv') and match.group(0) in f:
                        fpath = os.path.join(period_folder, f)
                        generate_displacement(inps, fpath, simulation_input, params)

            inps.folder_path = simulation_folder
            inversion(iargs=inps)
            # compare(simulation_folder)
            compare(sim_out_folder)


if __name__ == '__main__':
    main(iargs=sys.argv)