import numpy as np
from mintpy.utils import (
    readfile,
    utils as ut,
)
from mintpy import subset
from kite import Scene
from src.shared.helper_functions import extent2meshgrid, convert_to_utm


class Downsample:
    def __init__(self, velocity_file=None, kite_file=None, geometry_file=None):
        self.velocity_file = velocity_file
        self.geometry_file = geometry_file
        self.velocity, self.metadata = readfile.read(self.velocity_file)
        self.kite_file = kite_file

        print("#" * 50)
        print(f"Loading {self.velocity_file}.\n")

    def uniform(self, reduction=3):
        """Downsample the velocity data using a mask and geometry file.
        Parameters: velocity_file - path to the velocity data file
                    mask_file     - path to the mask file
                    geometry_file  - path to the geometry file
        Returns:    z_flat       - flattened velocity data
                    x_flat       - flattened x-coordinates
                    y_flat       - flattened y-coordinates
                    z_downsampled- downsampled velocity data
                    xx           - meshgrid x-coordinates
                    yy           - meshgrid y-coordinates
        """
        # Skip value every 'skip' step
        skip = reduction
        # coord = ut.coordinate(metadata)
        pix_box, geo_box = subset.subset_input_dict2box({"subset_lon": None,
                                                        "subset_lat": None,
                                                        "subset_x": None,
                                                        "subset_y": None}, self.metadata)
        print("#" * 50)
        print(f"Reducing {self.velocity_file} by a factor of {reduction}.\n")

        z = self.velocity[:: skip, ::skip]
        x, y = extent2meshgrid(extent=geo_box, ds_shape=z.shape)

        z = z.flatten()
        mask = np.isnan(z)
        self.length = len(z[~mask])

        x, y = convert_to_utm(longitude=x, latitude=y)

        self.x = x[~mask]
        self.y = y[~mask]
        self.z = z[~mask]

        self._LOS()


    def quadtree(self, epsilon=0.0029, tile_size_max=0.02, tile_size_min=0.002, nan_allowed=0.9):
        sc = Scene.load(self.kite_file)

        print("#" * 50)
        print(f"Reducing {self.kite_file} with Quadtree.\n")

        qt = sc.quadtree

        # Parametrisation of the quadtree
        qt.epsilon = epsilon             # Variance threshold
        qt.nan_allowed = nan_allowed     # Percentage of NaN values allowed per tile/leave

        # Be careful here, if you scene is referenced in degree use decimal values!
        qt.tile_size_max = tile_size_max  # Maximum leave edge length in [m] or [deg]
        qt.tile_size_min = tile_size_min   # Minimum leave edge length in [m] or [deg]

        coords = qt.leaf_focal_points
        self.z = qt.leaf_medians
        self.length = len(qt.leaf_eastings)

        self.x, self.y = convert_to_utm(longitude=qt.leaf_coordinates[:, 0] + sc.frame.llLon, latitude=qt.leaf_coordinates[:, 1] + sc.frame.llLat)

        self._LOS()


    def _LOS(self):
        self.los_az_angle = float(self.metadata['HEADING'])
        if 'CENTER_INCIDENCE_ANGLE' in self.metadata:
            self.los_inc_angle = float(self.metadata['CENTER_INCIDENCE_ANGLE'])
        else:
            data = readfile.read(self.geometry_file, datasetName='/incidenceAngle')[0]
            self.los_inc_angle = np.nanmedian(data)

        self.ref_lat = float(self.metadata['REF_LAT'])
        self.ref_lon = float(self.metadata['REF_LON'])

        LOS_E = -np.sin(np.deg2rad(self.los_inc_angle)) * np.cos(np.deg2rad(self.los_az_angle))
        LOS_N = np.sin(np.deg2rad(self.los_inc_angle)) * np.sin(np.deg2rad(self.los_az_angle))
        LOS_Z = np.cos(np.deg2rad(self.los_inc_angle))

        self.lose = [LOS_E for _ in range(self.length)]
        self.losn = [LOS_N for _ in range(self.length)]
        self.losz = [LOS_Z for _ in range(self.length)]

        self.err = np.full(self.length, 0.1)