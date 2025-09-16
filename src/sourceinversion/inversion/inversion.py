#!/usr/bin/env python3

import os
import re
import sys
import glob
import logging
import argparse
import pandas as pd
from VSM import VSM
import matplotlib.pyplot as plt
from mintpy.utils import readfile
from sourceinversion.shared.plot import InversionPlotter
# from sourceinversion.shared.plot import plot_results as plot
from sourceinversion.shared.csv_functions import results_csv, read_best_values
from sourceinversion.shared.helper_functions import inversion_template, get_bounding_box, SCRATCHDIR, MODEL_DEFS
from sourceinversion.shared.argument_parser import add_mogi_parameters, add_penny_parameters, add_spheroid_parameters, add_okada_parameters, add_sampling_parameters, add_coordinates_parameters


EXAMPLE = """
        run_inversion.py --folder CampiFlegrei --satellite Csk  -model mogi spheroid --show
        run_inversion.py --folder /path/to/folder --satellite Sen --txt-file template.txt --shear 0.5 --poisson 0.25 --x-range 0 100 --y-range 0 200 --z-range 0 5000 --model mogi --mogi-volume 1.e6 2.e7 --sampling_id 0 --weight-sar 1.0 --weight-gps 0.0 --show
"""


def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder.")
    parser.add_argument('--satellite', type=str, default='Sen', choices=['Sen', 'Csk'], help="Satellite name.")
    parser.add_argument('--txt-file', type=str, default=None , help="Path of the template file.")
    parser.add_argument('--shear', type=float, default=5e9, help="Shear value (default: 0.5).")
    parser.add_argument('--poisson', type=float, dest='nu', default=0.25, help="Poisson ratio (default: %(default)s).")
    parser.add_argument('--model', type=str, choices=['mogi', 'penny', 'spheroid', 'moment', 'okada'], nargs='+', help='Source model(s) to include.')
    parser.add_argument('--weight-sar', type=float, default=1.0, help="Weight for SAR data (default: 1.0).")
    parser.add_argument('--weight-gps', type=float, default=0.0, help="Weight for GPS data (default: 1.0).")
    parser.add_argument('--period', nargs='*', metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD', type=str, help='Period of the search')
    parser.add_argument('--bbox', action='store_true', help="Show bounding box of x and y range on plot.")
    parser.add_argument('--no-show', dest='show', action='store_false', help="Show the plot.")
    parser.add_argument('--fullres', dest='fullres', action='store_true', help="Show full resolution data.")
    parser.add_argument('--save', action='store_true', help="Save the plot as PNG.")

    parser = add_mogi_parameters(parser)
    parser = add_penny_parameters(parser)
    parser = add_spheroid_parameters(parser)
    parser = add_okada_parameters(parser)
    parser = add_sampling_parameters(parser)
    parser = add_coordinates_parameters(parser)

    # Parse arguments
    inps = parser.parse_args()

    inps.folder_path = inps.folder if SCRATCHDIR in inps.folder else os.path.join(SCRATCHDIR, inps.folder)

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

    for attr in ['x_range', 'y_range']:
        while len(inps.model) > len(getattr(inps, attr)):
            getattr(inps, attr).append(None)

    return inps


def extract_model_parameters(inps):
    model_dict = {}

    for model in inps.model:
        model = model.lower()
        if model not in MODEL_DEFS:
            continue

        model_id = MODEL_DEFS[model]['id']
        param_keys = MODEL_DEFS[model]['params']

        param_values = []
        for param in param_keys:
            val = getattr(inps, f'{model}_{param}', None)
            if val is None:
                raise ValueError(f'Missing parameter --{model}-{param}')
            param_values.append(val)

        model_dict[model_id] = {
            'name': model,
            'params': param_values
        }

    return model_dict


def define_range(tupla, df):
    tupla[0] = round(min(tupla[0], df.min()))
    tupla[1] = round(max(tupla[1], df.max()))

    return tupla


def run_vsm(inps, output_folder, input_sar, model_inputs):
    if not inps.txt_file:
        inps.txt_file = os.path.join(output_folder, 'VSM_input.txt')

    inversion_template(
        txt_file=inps.txt_file,
        output_folder=output_folder,
        input_sar=input_sar,
        input_gps=getattr(inps, "input_gps", None),
        shear=inps.shear,
        poisson=inps.nu,
        x_range=inps.x,
        y_range=inps.y,
        z_range=inps.z_range,
        models=model_inputs,
        sampling_id=inps.sampling_id,
        weight_sar=inps.weight_sar,
        weight_gps=inps.weight_gps,
        p1=inps.p1,
        p2=inps.p2,
        p3=inps.p3
    )

    if not glob.glob(os.path.join(output_folder, 'VSM_synth_*.csv')):
        print("Inverting with\n")
        print(
            """
                 /$$    /$$  /$$$$$$  /$$      /$$
                | $$   | $$ /$$__  $$| $$$    /$$$
                | $$   | $$| $$  \__/| $$$$  /$$$$
                |  $$ / $$/|  $$$$$$ | $$ $$/$$ $$
                 \  $$ $$/  \____  $$| $$  $$$| $$
                  \  $$$/   /$$  \ $$| $$\  $ | $$
                   \  $/   |  $$$$$$/| $$ \/  | $$
                    \_/     \______/ |__/     |__/
            """)
        VSM.read_VSM_settings(inps.txt_file)
        VSM.iVSM()

        print("-" * 50)
        print("Inversion completed with VSM.\n")
    else:
        print("-" * 50)
        print("VSM_synth already exists, skipping inversion.\n")

    sar_dict = {}
    for i, path in enumerate(input_sar.split()):
        matching_files = [os.path.join(os.path.dirname(path), f) for f in os.listdir(os.path.dirname(path)) if 'velocity_msk.h5' in f]
        sar_dict[f"VSM_synth_sar{i+1}.csv"] = matching_files[0] if matching_files else None

    return sar_dict


def plot_results(inps, output_folder, period=None, file_dictionary=None):
    if os.path.exists(os.path.join(output_folder, 'VSM_best.csv')):
        sources_center = read_best_values(os.path.join(output_folder, 'VSM_best.csv'))
    else:
        # raise FileNotFoundError(f"VSM_best.csv not found in {output_folder}")
        print(f"VSM_best.csv not found in {output_folder}")
        sources_center = None
    figures = []
    for file in os.listdir(output_folder):
        if 'VSM_synth' in file and file.endswith('.csv'):
            east, north, data, synth = results_csv(os.path.join(output_folder, file))
            deformation, metadata = readfile.read(file_dictionary[file])
            lat, lon = get_bounding_box(metadata)

            plotter = InversionPlotter(inps, east, north, data, synth, deformation, inps.model, sources_center=sources_center, period=period, latitude=lat, longitude=lon, bbox=inps.bbox)

            fig = plotter.plot()
            figures.append(fig)

    return figures


def gather_input_sar(inps, base_folder, match_str):
    input_sar = ''
    for f in os.listdir(base_folder):
        if f.endswith('.csv') and match_str in f:
            input_sar += os.path.join(base_folder, f) + ' '
            df = pd.read_csv(os.path.join(base_folder, f))

            x_range = define_range([float('inf'), float('-inf')], df['xx'])
            range_width = (x_range[1] - x_range[0])/2
            inps.x = []
            for x in inps.x_range:
                inps.x.append(x_range if not x else ([x - (range_width * inps.scaling_box), x + (range_width * inps.scaling_box)]))

            y_range = define_range([float('inf'), float('-inf')], df['yy'])
            range_length = (y_range[1] - y_range[0])/2
            inps.y = []
            for y in inps.y_range:
                inps.y.append(y_range if not y else ([y - (range_length * inps.scaling_box), y + (range_length * inps.scaling_box)]))

    if input_sar == '':
        raise FileNotFoundError(f"No matching CSV files found in {base_folder} for pattern {match_str}")

    return input_sar


def process_folder(inps, input_sar, period=None):
    """Run inversion and plotting for a given folder."""
    models = '_'.join(inps.model)
    output_folder = os.path.join(inps.folder_path, period, models) if period else os.path.join(inps.folder_path, models)
    os.makedirs(output_folder, exist_ok=True)

    model_inputs = extract_model_parameters(inps)
    sar_dict = run_vsm(inps, output_folder, input_sar, model_inputs)

    if inps.show or inps.save:
        figures = plot_results(inps, output_folder, period, file_dictionary=sar_dict)
        if inps.save:
            prefix = f"Inversion_result" if period else "VSM_results"
            for i, fig in enumerate(figures, start=1):
                save_path = os.path.join(output_folder, f"{prefix}_{i}.png")
                fig.savefig(save_path, dpi=300)
                print(f"Figure saved as {save_path}\n")


def gather_all_inputs(inps, folder_list, regex, period=None):
    """Collect input_sar across all matching folders."""
    input_sar = ''
    for folder in folder_list:
        match = regex.match(folder)
        if not match:
            continue

        input_folder = os.path.join(inps.folder_path, folder)
        period_folder = os.path.join(input_folder, period) if period else input_folder

        if period and not os.path.exists(period_folder):
            print(f"Period folder {period_folder} does not exist.")
            continue

        input_sar += gather_input_sar(inps, period_folder, match.group(0))

    return input_sar


def main(iargs=None):
    inps = create_parser() if not isinstance(iargs, argparse.Namespace) else iargs

    # Configure logging to write to a log file
    logging.basicConfig(filename=os.path.join(inps.folder_path, 'log'), level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d')

    # Log the command-line command
    cmd_command = ' '.join(sys.argv)
    logging.info(cmd_command)

    print("-" * 50)
    print("Starting Inversion Module...")
    print("-" * 50)
    print()

    if inps.satellite:
        pattern = f"({'|'.join([f'{inps.satellite}[AD]T?'])})\\d+"
        regex = re.compile(pattern)
        folder_list = [f for f in os.listdir(inps.folder_path) if os.path.isdir(os.path.join(inps.folder_path, f))]

    if inps.period_folder:
        for period in inps.period_folder:
            input_sar = gather_all_inputs(inps, folder_list, regex=regex, period=period)
            process_folder(inps, input_sar, period)
    else:
        input_sar = gather_all_inputs(inps, folder_list, regex=regex)
        process_folder(inps, input_sar)

    if inps.show:
        print("-" * 50)
        print("Plotting results...\n")
        plt.show()


if __name__ == '__main__':
    main(iargs=sys.argv)