import os
import glob
import numpy as np
from pyproj import Transformer

SCRATCHDIR = os.getenv('SCRATCHDIR')

MODEL_DEFS = {
    'mogi': {
        'id': '0',
        'params': ['volume'],
    },
    'point': {
        'id': '1',
        'params': ['volume'],
    },
    'penny': {
        'id': '2',
        'params': ['radius', 'dP_mu'],
    },
    'spheroid': {
        'id': '3',
        'params': ['strike', 'dip', 'ratio', 'semi_axis', 'dp_mu'],
    },
    'moment': {
        'id': '4',
        'params': ['Mxx', 'Myy', 'Mzz', 'Mxy', 'Mxz', 'Myz'],
    },
    'okada': {
        'id': '5 R',
        'params': ['length', 'width', 'strike', 'dip', 'slip', 'rake', 'opening'],
    }
}


def extent2meshgrid(extent: tuple, ds_shape: list):
    """Get mesh grid coordinates for a given extent and shape.
    Parameters: extent - tuple of float for (left, right, bottom, top) in data coordinates
                shape  - list of int for [length, width] of the data
    Returns:    xx/yy  - 1D np.ndarray of the data coordinates
    """
    height, width = ds_shape
    x = np.linspace(extent[0], extent[2], width)
    y = np.linspace(extent[3], extent[1], height)[::-1]  # reverse the Y-axis
    xx, yy = np.meshgrid(x, y)
    return xx.flatten(), yy.flatten()


def get_file_names(path):
    """gets the youngest eos5 file. Path can be:
    MaunaLoaSenAT124
    MaunaLoaSenAT124/mintpy/S1_qq.he5
    ~/onedrive/scratch/MaunaLoaSenAT124/mintpy/S1_qq.he5'
    """
    from mintpy.utils import readfile

    scratch = os.getenv('SCRATCHDIR')
    if os.path.isfile(glob.glob(path)[0]):
        eos_file = glob.glob(path)[0]

    elif os.path.isfile(os.path.join(scratch, path)):
        eos_file = scratch + '/' + path

    else:
        if 'mintpy' in path or 'network' in path :
            files = glob.glob(path + '/*.he5')

        else:
            files = glob.glob( path + '/mintpy/*.he5' )

        if len(files) == 0:
            raise Exception('USER ERROR: No HDF5EOS files found in ' + path)

        eos_file = max(files, key=os.path.getctime)

    print('HDF5EOS file used:', eos_file)

    metadata = readfile.read(eos_file)[1]
    velocity_file = 'geo/geo_velocity.h5'
    geometryRadar_file = 'geo/geo_geometryRadar.h5'

    # Check if geocoded
    if 'Y_STEP' not in metadata:
        velocity_file = (velocity_file.split(os.sep)[-1]).replace('geo_', '')
        geometryRadar_file = geometryRadar_file.split(os.sep)[-1].replace('geo_', '')

    keywords = ['SenD','SenA','SenDT', 'SenAT', 'CskAT', 'CskDT']
    elements = path.split(os.sep)
    project_dir = None
    for element in elements:
        for keyword in keywords:
            if keyword in element:
                project_dir = element
                project_base_dir = element.split(keyword)[0]
                track_dir = keyword + element.split(keyword)[1]
                break

    project_base_dir = os.path.join(scratch, project_base_dir)
    vel_file = os.path.join(eos_file.rsplit('/', 1)[0], velocity_file)
    geometry_file = os.path.join(eos_file.rsplit('/', 1)[0], geometryRadar_file)

    inputs_folder = os.path.join(scratch, project_dir)
    out_vel_file = os.path.join(project_base_dir, track_dir, velocity_file.split(os.sep)[-1])

    return eos_file, vel_file, geometry_file, project_base_dir, out_vel_file, inputs_folder


def convert_to_utm(longitude, latitude):
    """
    Converts latitude and longitude to UTM coordinates.

    Parameters:
        longitude (array-like): Array of longitude values.
        latitude (array-like): Array of latitude values.

    Returns:
        tuple: Arrays of UTM Eastings (x) and Northings (y).
    """
    # Calculate the UTM zone based on the longitude
    utm_zone = int((np.nanmean(longitude) + 180) // 6) + 1

    # Determine the hemisphere based on latitude
    hemisphere = 'north' if np.nanmean(latitude) >= 0 else 'south'

    # Determine the EPSG code based on the UTM zone and hemisphere
    epsg_code = f"326{utm_zone:02d}" if hemisphere == 'north' else f"327{utm_zone:02d}"

    # Create a Transformer object for WGS84 to UTM
    transformer = Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)

    # Convert to UTM coordinates (Eastings, Northings)
    x, y = transformer.transform(longitude, latitude)

    return x, y


def inversion_template(txt_file,output_folder,input_sar=None,input_gps=None,shear=None,poisson=None,x_range=None,y_range=None,z_range=None,models=None,sampling_id='0',weight_sar=0.0,weight_gps=0.0,p1='1000',p2='300',p3='12'):
    """
    Write VSM inversion template with multiple source models and shared x/y/z ranges.

    Parameters
    ----------
    models : dict
        Dictionary with source_id as keys, and values as dicts with:
        - 'name': model name (str)
        - 'params': list of model-specific parameter ranges
    """
    lines = [
        f'{output_folder}',
        f'{input_sar}',
        f'{input_gps}',
        'None',
        'None',
        'None',
        'None',
        f'{weight_sar}',
        f'{weight_gps}',
        '0.0',
        '0.0',
        '0.0',
        '0.0',
        f'{shear}',
        f'{poisson}',
        str(len(models)),
    ]
    for (source_id, info), x, y, z in zip(models.items(), x_range, y_range, z_range):
        lines.append(f'{source_id}')  # model ID

        # # Add shared spatial parameters first
        lines.append(f'{x[0]}\t{x[1]}')
        lines.append(f'{y[0]}\t{y[1]}')
        lines.append(f'{z[0]}\t{z[1]}')

        # Add model-specific parameters
        param_names = MODEL_DEFS[info['name']]['params']
        for val_range, param_name in zip(info['params'], param_names):
            lines.append(f'{val_range[0]}\t{val_range[1]}\t{param_name}')

    # Sampling algorithm & params
    # p1 = '1000'   #p1
    # p2 = '300'    #p2
    # p3 = '40'     #p3 or BI steps
    lines.append(str(sampling_id))       # 0 for NA, 1 for BI
    if(sampling_id == '0'):
        lines.append(f"{p1}\t{p2}")
        lines.append(f"{p3}")
    else:
        lines.append(p1)
        lines.append(p2)

    lines.append('2000')                 # burn-in

    # Write to file
    with open(txt_file, 'w') as f:
        f.write('\n'.join(lines))


def get_bounding_box(metadata):
    """
    Calculate the bounding box coordinates based on the given metadata.

    Args:
        metadata (dict): A dictionary containing the metadata information.

    Returns:
        tuple: A tuple containing two lists, the first list represents the latitude range and the second list represents the longitude range.
    """
    lat_out = []
    lon_out = []

    length = int(metadata['LENGTH'])
    width = int(metadata['WIDTH'])

    for y_i, x_i in zip([0, length], [0, width]):
        lat_i = None if y_i is None else (y_i + 0.5) * float(metadata['Y_STEP']) + float(metadata['Y_FIRST'])
        lon_i = None if x_i is None else (x_i + 0.5) * float(metadata['X_STEP']) + float(metadata['X_FIRST'])
        lat_out.append(lat_i)
        lon_out.append(lon_i)

    return [min(lat_out), max(lat_out)], [min(lon_out), max(lon_out)]