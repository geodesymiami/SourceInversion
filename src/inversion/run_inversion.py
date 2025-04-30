#!/usr/bin/env python3

import os
import re
import sys
import VSM
import glob
import argparse
import pandas as pd
from src.shared.plot import plot_results
from src.shared.csv_functions import results_csv
from src.shared.helper_functions import inversion_template

EXAMPLE = """
        run_inversion.py --folder CampiFlegrei --satellite Csk  -model mogi spheroid --show
        run_inversion.py --folder /path/to/folder --satellite Sen --txt-file template.txt --shear 0.5 --poisson 0.25 --x-range 0 100 --y-range 0 200 --z-range 0 5000 --volume 1.e6 2.e7 --sampling_id 0 --weight-sar 1.0 --weight-gps 0.0 --model mogi --show
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
    parser.add_argument('--shear', type=float, default=0.5, help="Shear value (default: 0.5).")
    parser.add_argument('--poisson', type=float, dest='nu', default=0.25, help="Poisson ratio (default: %(default)s).")
    parser.add_argument('--x-range', type=float, nargs=2, default=[float('inf'), float('-inf')], help="X range.")
    parser.add_argument('--y-range', type=float, nargs=2, default=[float('inf'), float('-inf')], help="Y range.")
    parser.add_argument('--z-range', type=float, nargs=2, default=(0, 5000), help="Z range (default: %(default)s).")
    parser.add_argument('--volume', type=str, default='1.e6 2.e7', help="Volume value (default: %(default)s).")
    parser.add_argument('--sampling_id', type=str, choices=['0', '1'], default='0', help="Sampling ID (default: %(default)s).")
    parser.add_argument('--weight-sar', type=float, default=1.0, help="Weight for SAR data (default: 1.0).")
    parser.add_argument('--weight-gps', type=float, default=0.0, help="Weight for GPS data (default: 1.0).")
    parser.add_argument('--model', type=str, nargs='+', choices=['mogi', 'point', 'penny', 'spheroid', 'moment', 'okada'], default=['mogi'], help="One or more models: Mogi (1958), McTigue point source (1987), Fialko et al.(2001), Penny-shaped crack, Yang et al. (1988). Spheroid, Davis (1986) Moment tensor, Okada 1985.")
    parser.add_argument('--show', action='store_true', help="Show the plot.")
    parser.add_argument('--period', nargs='*', metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD', type=str, help='Period of the search')

    # Parse arguments
    inps = parser.parse_args()

    inps.folder_path = inps.folder if SCRATCHDIR in inps.folder else os.path.join(SCRATCHDIR, inps.folder)

    if inps.txt_file:
        if os.path.dirname(inps.txt_file) == '':
            inps.txt_file = os.path.join(inps.folder_path, inps.txt_file)
    else:
        inps.txt_file = os.path.join(inps.folder_path, 'VSM_input.txt')

    if inps.satellite and inps.weight_sar == 0.0:
        inps.weight_sar = 1.0

    model = {
        'mogi': '0',
        'point': '1',
        'penny': '2',
        'spheroid': '3',
        'moment': '4',
        'okada': '5'
    }

    inps.model = " ".join([model[m] for m in inps.model])

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


def define_range(tupla, df):
    tupla[0] = round(min(tupla[0], df.min()))
    tupla[1] = round(max(tupla[1], df.max()))

    return tupla


def run_vsm(inps, output_folder, input_sar):
    inversion_template(
        inps.txt_file,
        output_folder,
        input_sar=input_sar,
        shear=inps.shear,
        poisson=inps.nu,
        x_range=f"{inps.x_range[0]} {inps.x_range[1]}",
        y_range=f"{inps.y_range[0]} {inps.y_range[1]}",
        z_range=f"{inps.z_range[0]} {inps.z_range[1]}",
        Volume=inps.volume,
        sampling_id=inps.sampling_id,
        weight_sar=inps.weight_sar,
        weight_gps=inps.weight_gps
    )

    if not glob.glob(os.path.join(output_folder, 'VSM_synth_*.csv')):
        VSM.read_VSM_settings(inps.txt_file)
        VSM.iVSM()


def plot_results(inps, output_folder):
    for file in os.listdir(output_folder):
        if 'VSM_synth' in file and file.endswith('.csv'):
            east, north, data, synth = results_csv(os.path.join(output_folder, file))
            plot_results(east, north, data, synth)


def main(iargs=None):
    print("#" * 50)
    print("Starting Inversion Module...")
    print("#" * 50)
    print()

    inps = create_parser() if not isinstance(iargs, argparse.Namespace) else iargs

    if inps.satellite:
        pattern = f"({'|'.join([f'{inps.satellite}[AD]T?'])})\\d+"
        regex = re.compile(pattern)
        folder_list = [f for f in os.listdir(inps.folder_path) if os.path.isdir(os.path.join(inps.folder_path, f))]

        def gather_input_sar(base_folder, match_str):
            input_sar = ''
            for f in os.listdir(base_folder):
                if f.endswith('.csv') and match_str in f:
                    input_sar += os.path.join(base_folder, f) + ' '
                    df = pd.read_csv(os.path.join(base_folder, f))
                    inps.x_range = define_range(inps.x_range, df['xx'])
                    inps.y_range = define_range(inps.y_range, df['yy'])
            return input_sar

        if inps.period_folder:
            for period in inps.period_folder:
                input_sar = ''
                for folder in folder_list:
                    match = regex.match(folder)
                    if match:
                        input_folder = os.path.join(inps.folder_path, folder)
                        period_folder = os.path.join(input_folder, period)
                        output_folder = os.path.join(inps.folder_path, period)

                        if not os.path.exists(period_folder):
                            print(f"Period folder {period_folder} does not exist.")
                            continue

                        os.makedirs(output_folder, exist_ok=True)
                        input_sar += gather_input_sar(period_folder, match.group(0))

                run_vsm(inps, output_folder, input_sar)
                if inps.show:
                    plot_results(inps, output_folder)

        else:
            input_sar = ''
            for folder in folder_list:
                match = regex.match(folder)
                if match:
                    input_folder = os.path.join(inps.folder_path, folder)
                    input_sar += gather_input_sar(input_folder, match.group(0))

            run_vsm(inps, inps.folder_path, input_sar)
            if inps.show:
                plot_results(inps, inps.folder_path)


if __name__ == '__main__':
    main(iargs=sys.argv)