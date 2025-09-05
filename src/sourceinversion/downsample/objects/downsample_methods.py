import numpy as np
from kite import Scene
from scipy.ndimage import zoom
from mintpy.utils import readfile
from sourceinversion.shared.helper_functions import extent2meshgrid, convert_to_utm


class Downsample:
    def __init__(self, velocity_file=None, kite_file=None, geometry_file=None):
        self.velocity_file = velocity_file
        self.geometry_file = geometry_file
        self.velocity, self.metadata = readfile.read(self.velocity_file)


        self.incident_angle = readfile.read(self.geometry_file, datasetName='/incidenceAngle')[0]
        self.azimuth_angle = readfile.read(self.geometry_file, datasetName='/azimuthAngle')[0]
        self.latitude = readfile.read(self.geometry_file, datasetName='latitude')[0]
        self.longitude = readfile.read(self.geometry_file, datasetName='longitude')[0]
        self.kite_file = kite_file

        self._resize()

        print("#" * 50)
        print(f"Loading {self.velocity_file}.\n")

    def _resize(self):
        if self.incident_angle.shape != self.velocity.shape:
            if all(dim > 0 for dim in self.incident_angle.shape):
                zoom_factors = (
                    self.velocity.shape[0] / self.incident_angle.shape[0],
                    self.velocity.shape[1] / self.incident_angle.shape[1],
                )
                self.incident_angle = zoom(self.incident_angle, zoom_factors, order=1)
            else:
                raise ValueError("Invalid shape for incident_angle: {}".format(self.incident_angle.shape))

        if self.azimuth_angle.shape != self.velocity.shape:
            if all(dim > 0 for dim in self.azimuth_angle.shape):
                zoom_factors = (
                    self.velocity.shape[0] / self.azimuth_angle.shape[0],
                    self.velocity.shape[1] / self.azimuth_angle.shape[1],
                )
                self.azimuth_angle = zoom(self.azimuth_angle, zoom_factors, order=1)  # Linear interpolation
            else:
                raise ValueError("Invalid shape for azimuth_angle: {}".format(self.azimuth_angle.shape))

        if self.latitude.shape != self.velocity.shape:
            if all(dim > 0 for dim in self.latitude.shape):
                zoom_factors = (
                    self.velocity.shape[0] / self.latitude.shape[0],
                    self.velocity.shape[1] / self.latitude.shape[1],
                )
                self.latitude = zoom(self.latitude, zoom_factors, order=1)
            else:
                raise ValueError("Invalid shape for latitude: {}".format(self.latitude.shape))

        if self.longitude.shape != self.velocity.shape:
            if all(dim > 0 for dim in self.longitude.shape):
                zoom_factors = (
                    self.velocity.shape[0] / self.longitude.shape[0],
                    self.velocity.shape[1] / self.longitude.shape[1],
                )
                self.longitude = zoom(self.longitude, zoom_factors, order=1)
            else:
                raise ValueError("Invalid shape for longitude: {}".format(self.longitude.shape))

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

        print("#" * 50)
        print(f"Reducing {self.velocity_file} by a factor of {reduction}.\n")

        # Slice and flatten arrays
        self.imshow = self.velocity[::skip, ::skip]
        sliced_x = self.longitude[::skip, ::skip]
        sliced_y = self.latitude[::skip, ::skip]
        sliced_incident_angle = self.incident_angle[::skip, ::skip]
        sliced_azimuth_angle = self.azimuth_angle[::skip, ::skip]

        z = self.imshow.flatten()
        x = sliced_x.flatten()
        y = sliced_y.flatten()
        incident_angle = sliced_incident_angle.flatten()
        azimuth_angle = sliced_azimuth_angle.flatten()

        # Apply mask to remove NaN values
        mask = ~np.isnan(z)
        self.length = np.sum(mask)

        # Convert coordinates to UTM and apply mask
        x, y = convert_to_utm(longitude=x[mask], latitude=y[mask])

        # Assign filtered values to instance variables
        self.z = z[mask]
        self.x = x
        self.y = y
        self.incident = incident_angle[mask]
        self.azimuth = azimuth_angle[mask]

        self._LOS()


    def quadtree(self, epsilon=0.0029, tile_size_max=0.02, tile_size_min=0.002, nan_allowed=0.9):
        sc = Scene.load(self.kite_file)

        print("#" * 50)
        print(f"Reducing {self.kite_file} with Quadtree.\n")

        self.qt = sc.quadtree

        # Parametrisation of the quadtree
        self.qt.epsilon = epsilon             # Variance threshold
        self.qt.nan_allowed = nan_allowed     # Percentage of NaN values allowed per tile/leave

        # Be careful here, if you scene is referenced in degree use decimal values!
        self.qt.tile_size_max = tile_size_max  # Maximum leave edge length in [m] or [deg]
        self.qt.tile_size_min = tile_size_min   # Minimum leave edge length in [m] or [deg]

        self.z = self.qt.leaf_medians
        self.length = len(self.qt.leaf_eastings)
        shape = self.incident_angle.shape

        qt_lons = self.qt.leaf_coordinates[:, 0] + sc.frame.llLon
        qt_lats = self.qt.leaf_coordinates[:, 1] + sc.frame.llLat

        self.x, self.y = convert_to_utm(longitude=self.qt.leaf_coordinates[:, 0] + sc.frame.llLon, latitude=self.qt.leaf_coordinates[:, 1] + sc.frame.llLat)

        lat_min, lat_max = qt_lats.min(), qt_lats.max()
        lon_min, lon_max = qt_lons.min(), qt_lons.max()

        self.incident, self.azimuth = self._extract_geometry_values(
            lats=qt_lats,
            lons=qt_lons,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            shape=shape,
            incident_angle=self.incident_angle,
            azimuth_angle=self.azimuth_angle
        )

        self._LOS()


    def _extract_geometry_values(self, lats, lons, lat_min, lat_max, lon_min, lon_max, shape, incident_angle, azimuth_angle):
        """Extract geometry values from regular lat/lon grid at given coordinates."""
        n_rows, n_cols = shape
        lat_step = (lat_max - lat_min) / n_rows
        lon_step = (lon_max - lon_min) / n_cols

        row_idx = ((lat_max - lats) / lat_step).astype(int)
        col_idx = ((lons - lon_min) / lon_step).astype(int)

        row_idx = np.clip(row_idx, 0, n_rows - 1)
        col_idx = np.clip(col_idx, 0, n_cols - 1)

        return incident_angle[row_idx, col_idx], azimuth_angle[row_idx, col_idx]


    def _LOS(self):
        self.ref_lat = float(self.metadata['REF_LAT'])
        self.ref_lon = float(self.metadata['REF_LON'])

        # Calculate LOS components using metadata values
        self.lose = -np.sin(np.deg2rad(self.incident)) * np.cos(np.deg2rad(self.azimuth)-np.pi/2)
        self.losn = np.sin(np.deg2rad(self.incident)) * np.sin(np.deg2rad(self.azimuth)-np.pi/2)
        self.losz = np.cos(np.deg2rad(self.incident))


        self.err = np.full(len(self.z), 0.2)