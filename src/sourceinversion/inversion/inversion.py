#!/usr/bin/env python3

import os
import re
import sys
from VSM.VSM import VSM
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sourceinversion.shared.plot import plot_results as plot
from sourceinversion.shared.csv_functions import results_csv, read_best_values
from sourceinversion.shared.helper_functions import inversion_template, SCRATCHDIR, MODEL_DEFS
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
    parser.add_argument('--no-show', dest='show', action='store_false', help="Show the plot.")
    parser.add_argument('--period', nargs='*', metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD', type=str, help='Period of the search')
    parser.add_argument('--bbox', action='store_true', help="Show bounding box of x and y range on plot.")

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
        x_range=inps.x_range,
        y_range=inps.y_range,
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
        VSM.read_VSM_settings(inps.txt_file)
        VSM.iVSM()

        print("#" * 50)
        print("Inversion completed with VSM.\n")
    else:
        print("#" * 50)
        print("VSM_synth already exists, skipping inversion.\n")



def plot_results(inps, output_folder, period=None):
    if os.path.exists(os.path.join(output_folder, 'VSM_best.csv')):
        sources_center = read_best_values(os.path.join(output_folder, 'VSM_best.csv'))
    else:
        # raise FileNotFoundError(f"VSM_best.csv not found in {output_folder}")
        print(f"VSM_best.csv not found in {output_folder}")
        sources_center = None

    for file in os.listdir(output_folder):
        if 'VSM_synth' in file and file.endswith('.csv'):
            east, north, data, synth = results_csv(os.path.join(output_folder, file))
            plot(inps, east, north, data, synth, sources_center, inps.model, period)


def gather_input_sar(inps, base_folder, match_str):
    input_sar = ''
    for f in os.listdir(base_folder):
        if f.endswith('.csv') and match_str in f:
            input_sar += os.path.join(base_folder, f) + ' '
            df = pd.read_csv(os.path.join(base_folder, f))
            if float('inf') in inps.x_range:
                inps.x_range = define_range(inps.x_range, df['xx'])
            else:
                if len(inps.x_range) == 1:
                    # Get the min and max values of df['xx']
                    data_range = define_range([float('inf'), float('-inf')], df['xx'])
                    range_width = data_range[1] - data_range[0]
                    inps.x_range = [
                        inps.x_range[0] - range_width * inps.scaling_box,
                        inps.x_range[0] + range_width * inps.scaling_box
                    ]

            if float('inf') in inps.y_range:
                inps.y_range = define_range(inps.y_range, df['yy'])
            else:
                if len(inps.y_range) == 1:
                    data_range = define_range([float('inf'), float('-inf')], df['yy'])
                    range_width = data_range[1] - data_range[0]
                    inps.y_range = [
                        inps.y_range[0] - range_width * inps.scaling_box,
                        inps.y_range[0] + range_width * inps.scaling_box
                    ]
    return input_sar


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
                        input_sar += gather_input_sar(inps, period_folder, match.group(0))

                model_inputs = extract_model_parameters(inps)
                run_vsm(inps, output_folder, input_sar, model_inputs)
                if inps.show:
                    plot_results(inps, output_folder, period)

        else:
            input_sar = ''
            for folder in folder_list:
                match = regex.match(folder)
                if match:
                    input_folder = os.path.join(inps.folder_path, folder)
                    input_sar += gather_input_sar(inps, input_folder, match.group(0))

            model_inputs = extract_model_parameters(inps)
            run_vsm(inps, inps.folder_path, input_sar, model_inputs)
            if inps.show:
                plot_results(inps, inps.folder_path)

    if inps.show:
        print("#" * 50)
        print("Plotting results...\n")
        plt.show()


if __name__ == '__main__':
    main(iargs=sys.argv)