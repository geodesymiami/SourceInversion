#!/usr/bin/env python3

import os
import re
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from VSM.VSM import VSM
import matplotlib.pyplot as plt
from mintpy.utils import readfile
from sourceinversion.shared.plot import InversionPlotter
# from sourceinversion.shared.plot import plot_results as plot
from sourceinversion.shared.csv_functions import results_csv, read_best_values
from sourceinversion.shared.helper_functions import inversion_template, get_bounding_box, SCRATCHDIR, MODEL_DEFS, convert_to_utm
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
    parser.add_argument('--path', type=str, required=True, help="Path to the folder.")
    parser.add_argument('--output-folder', type=str, default=None, help="Output folder for inversion results.")
    parser.add_argument('--satellite', type=str, default='Sen', choices=['Sen', 'Csk'], help="Satellite name.")
    parser.add_argument('--txt-file', type=str, default=None , help="Path of the template file.")
    parser.add_argument('--shear', type=float, default=5e9, help="Shear value (default: %(default)s).")
    parser.add_argument('--poisson', type=float, dest='nu', default=0.25, help="Poisson ratio (default: %(default)s).")
    parser.add_argument('--model', type=str, choices=['mogi', 'penny', 'spheroid', 'moment', 'okada'], nargs='+', help='Source model(s) to include.')
    parser.add_argument('--weight-sar', type=float, default=1.0, help="Weight for SAR data (default: 1.0).")
    parser.add_argument('--weight-gps', type=float, default=0.0, help="Weight for GPS data (default: 1.0).")
    parser.add_argument('--period', nargs='*', metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD', type=str, help='Period of the search')
    parser.add_argument('--bbox', action='store_true', help="Show bounding box of x and y range on plot.")
    parser.add_argument('--no-show', dest='show', action='store_false', help="Show the plot.")
    parser.add_argument('--fullres', dest='fullres', action='store_true', help="Show full resolution data.")
    parser.add_argument('--zoom', type=float, default=1, help="Zoom factor for the plot (default: %(default)s).")
    parser.add_argument('--subset', type=str, default=None, help="Subsection coordinates for zoom in, LAT,LON:LAT,LON")
    parser.add_argument('--save', action='store_true', help="Save the plot as PNG.")
    parser.add_argument('--size', type=float, default=None, help="Marker size for scatter plot (default: 20).")
    parser.add_argument('--style', type=str, default='image', choices=['scatter', 'image'], help="Plot style: scatter or image (default: scatter).")
    parser.add_argument('--vlim', type=float, nargs=2, default=None, help="Velocity limits for color scale (default: min and max of data).")

    parser = add_mogi_parameters(parser)
    parser = add_penny_parameters(parser)
    parser = add_spheroid_parameters(parser)
    parser = add_okada_parameters(parser)
    parser = add_sampling_parameters(parser)
    parser = add_coordinates_parameters(parser)

    # Parse arguments
    inps = parser.parse_args()

    inps.folder_path = inps.path if (SCRATCHDIR in inps.path or (os.path.isabs(inps.path))) else os.path.join(SCRATCHDIR, inps.path)

    if not inps.size:
        inps.size = 20 * (inps.zoom ** 3)

    if inps.satellite and inps.weight_sar == 0.0:
        inps.weight_sar = 1.0

    if inps.subset:
        inps.subset = inps.subset.replace(',',' ').replace(':',' ').split(' ')
        e, n = convert_to_utm(longitude=[float(inps.subset[1]), float(inps.subset[3])], latitude=[float(inps.subset[0]), float(inps.subset[2])])
        inps.subset = [min(e), max(e), min(n), max(n)]

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

    for attr in ['x_range', 'y_range', 'z_range']:
        while len(inps.model) > len(getattr(inps, attr)):
            getattr(inps, attr).append(None)

    return inps


def extract_model_parameters(inps):
    model_dict = []

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

        # Initialize model_dict as a list

        # Add a new group with "name" and "params"
        model_dict.append({
            "id": model_id,
            'name': model,
            'params': param_values
        })

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
        z_range=inps.z,
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
    if len(input_sar.split()) >= 2:
        for i, path in enumerate(input_sar.split()):
            matching_files = [os.path.join(os.path.dirname(path), f) for f in os.listdir(os.path.dirname(path)) if 'velocity_msk.h5' in f]
            sar_dict[f"VSM_synth_sar{i+1}.csv"] = matching_files[0] if matching_files else None
    else:
        matching_files = [os.path.join(os.path.dirname(input_sar), f) for f in os.listdir(os.path.dirname(input_sar)) if 'velocity_msk.h5' in f]
        sar_dict["VSM_synth_sar.csv"] = matching_files[0] if matching_files else None

    return sar_dict


def plot_results(inps, output_folder, period=None, file_dictionary=None):
    if os.path.exists(os.path.join(output_folder, 'VSM_best.csv')):
        sources = read_best_values(os.path.join(output_folder, 'VSM_best.csv'))
    elif os.path.exists(os.path.join(output_folder, 'VSM_mean.csv')):
        sources = read_best_values(os.path.join(output_folder, 'VSM_mean.csv'))
    else:
        # raise FileNotFoundError(f"VSM_best.csv not found in {output_folder}")
        print(f"VSM_best.csv not found in {output_folder}")
        sources = None
    figures = []
    for file in os.listdir(output_folder):
        if 'VSM_synth' in file and file.endswith('.csv'):
            east, north, data, synth = results_csv(os.path.join(output_folder, file))
            deformation, metadata = readfile.read(file_dictionary[file]) #Full resolution

            for f in os.listdir(os.path.dirname(file_dictionary[file])):
                if 'downsampled' in f and 'velocity' in f:
                    v = readfile.read(os.path.join(os.path.dirname(file_dictionary[file]), f))[0] #Downsampled
                    inps.length, inps.width  = v.shape
                    inps.mask = ~np.isnan(v)
                    break

            lat, lon = get_bounding_box(metadata)

            if inps.period:
                temp = os.path.dirname(file_dictionary[file]).replace(inps.period[0].replace(':','_'),'')
            else:
                temp = os.path.dirname(file_dictionary[file])

            geometry_file = None
            geometry_data = None
            for f in os.listdir(temp):
                 if 'geometryRadar.h5' in f:
                    geometry_file = os.path.join(temp, f)
                    break

            if geometry_file:
                geometry_data = readfile.read(geometry_file, datasetName='height')[0]
                lat_arr = readfile.read(geometry_file, datasetName='latitude')[0]
                lon_arr = readfile.read(geometry_file, datasetName='longitude')[0]
                lat = (np.nanmin(lat_arr), np.nanmax(lat_arr))
                lon = (np.nanmin(lon_arr), np.nanmax(lon_arr))

            plotter = InversionPlotter(metadata, inps, east, north, data, geometry_data, synth, deformation, inps.model, sources=sources, period=period, latitude=lat, longitude=lon, bbox=inps.bbox)

            fig = plotter.plot()
            figures.append(fig)

    return figures


def gather_input_sar(inps, base_folder, match_str=None):
    def get_range(inps, values, coordinate):
        range = define_range([float('inf'), float('-inf')], values)
        range_width = (range[1] - range[0])/2
        array = []
        for i in coordinate:
            array.append(range if not i else ([i - (range_width * inps.scaling_box), i + (range_width * inps.scaling_box)]))

        return array

    input_sar = ''
    if os.path.isfile(base_folder):
        input_sar = base_folder
        df = pd.read_csv(base_folder)

        inps.x = get_range(inps, df['xx'], inps.x_range)
        inps.y = get_range(inps, df['yy'], inps.y_range)
        inps.z = []

        Z_MIN, Z_MAX = 500, 20000   # meters
        inflate = 2             # >1 enlarges the box

        for z in inps.z_range:
            if z:
                dz = z * inps.scaling_box * inflate
                z0 = max(z - dz, Z_MIN)
                z1 = min(z + dz, Z_MAX)
                inps.z.append([z0, z1])
            else:
                inps.z.append([1000, 9000])

        if len(inps.z) < len(inps.x):
            for i in range(len(inps.x) - len(inps.z)):
                inps.z.append([1000, 10000])
    else:
        for f in os.listdir(base_folder):
            if f.endswith('.csv') and match_str in f:
                input_sar += os.path.join(base_folder, f) + ' '
                df = pd.read_csv(os.path.join(base_folder, f))

                inps.x = get_range(inps, df['xx'], inps.x_range)
                inps.y = get_range(inps, df['yy'], inps.y_range)
                inps.z = []

                Z_MIN, Z_MAX = 500, 20000   # meters
                inflate = 2.0              # >1 enlarges the box

                for z in inps.z_range:
                    if z:
                        dz = z * inps.scaling_box * inflate
                        z0 = max(z - dz, Z_MIN)
                        z1 = min(z + dz, Z_MAX)
                        inps.z.append([z0, z1])
                    else:
                        inps.z.append([1000, 9000])

    if input_sar == '':
        raise FileNotFoundError(f"No matching CSV files found in {base_folder} for pattern {match_str}")

    return input_sar


def process_folder(inps, input_sar, period=None):
    """Run inversion and plotting for a given folder."""
    models = '_'.join(inps.model)
    if not inps.output_folder:
        output_folder = os.path.join(inps.folder_path, period, models) if period else os.path.join(inps.folder_path, models)
    else:
        output_folder = os.path.join(inps.output_folder, models)
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


def configure_logging(inps):
    """Configure root logging to write to console and log the command (basename only) to a file."""
    os.makedirs(inps.process_folder, exist_ok=True)
    log_path = os.path.join(inps.process_folder, 'log')

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # remove existing handlers to avoid duplicate logging
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d')

    # Console handler for general logs (do not write these to file)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Dedicated logger + file handler for the single command line entry
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    cmd_logger = logging.getLogger('command_logger')
    cmd_logger.setLevel(logging.INFO)
    # Prevent propagation so other handlers (root) don't also write this to stdout/file
    cmd_logger.propagate = False
    # Remove any existing handlers on cmd_logger (defensive)
    for h in list(cmd_logger.handlers):
        cmd_logger.removeHandler(h)
    cmd_logger.addHandler(fh)

    # Log only the command (basename + args) to the file via dedicated logger
    script_name = os.path.basename(sys.argv[0]) if len(sys.argv) > 0 else ''
    rest = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''
    cmd_command = f"{script_name} {rest}".strip()
    cmd_logger.info(cmd_command)


def main(iargs=None):
    inps = create_parser() if not isinstance(iargs, argparse.Namespace) else iargs

    pattern = f"({'|'.join([f'{inps.satellite}[AD]T?'])})\\d+"
    regex = re.compile(pattern)
    if os.path.isfile(inps.folder_path):
        match = regex.search(inps.folder_path)
        if match:
            inps.process_folder = inps.folder_path.split(match.group(0))[0]
    else:
        inps.process_folder = inps.folder_path

    configure_logging(inps)

    print("-" * 50)
    print("Starting Inversion Module...")
    print("-" * 50)
    print()

    if os.path.isfile(inps.folder_path):
        input_sar = gather_input_sar(inps, inps.folder_path)
        process_folder(inps, input_sar)
    else:
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