import os
import numpy as np
from kite import Scene
from datetime import datetime
from mintpy.utils import readfile, writefile
from sourceinversion.shared.helper_functions import convert_to_utm


class Downsample():
    def __init__(self, data: object, kite_file=None):
        for attr in dir(data):
            if not attr.startswith('__') and not callable(getattr(data, attr)):
                setattr(self, attr, getattr(data, attr))
        start, end = self.period.split(':')
        self.days = (datetime.strptime(end, '%Y%m%d') - datetime.strptime(start, '%Y%m%d')).days
        self.kite_file = kite_file

    def uniform(self, reduction=3):
        """Downsample the data using a mask and geometry file.
        Args:
            reduction (int): The factor by which to reduce the data.
        """

        # Skip value every 'skip' step
        skip = reduction

        print("-" * 50)
        print(f"Reducing {self.velocity_file} by a factor of {reduction}.\n")

        # Slice and flatten arrays
        self.imshow = self.velocity[::skip, ::skip]
        self.sliced_x = self.longitude[::skip, ::skip]
        self.sliced_y = self.latitude[::skip, ::skip]
        self.sliced_incident_angle = self.incident_angle[::skip, ::skip]
        self.sliced_azimuth_angle = self.azimuth_angle[::skip, ::skip]
        self.sliced_height = self.height[::skip, ::skip]

        if self.temporal_coherence is not None:
            self.sliced_temporal_coherence = self.temporal_coherence[::skip, ::skip]
            self.sliced_temporal_coherence = self.sliced_temporal_coherence.flatten()

        z = self.imshow.flatten()
        x = self.sliced_x.flatten()
        y = self.sliced_y.flatten()
        incident_angle = self.sliced_incident_angle.flatten()
        azimuth_angle = self.sliced_azimuth_angle.flatten()

        # Apply mask to remove NaN values
        mask = ~np.isnan(z)
        self.length = np.sum(mask)

        # Convert coordinates to UTM and apply mask
        x, y = convert_to_utm(longitude=x, latitude=y)

        # Velocity to displacement conversion
        z = z / 365.2 * self.days

        # Assign filtered values to instance variables
        self.z = z
        self.x = x
        self.y = y
        self.incident = incident_angle
        self.azimuth = azimuth_angle

        self._LOS()
        self._write_file()
        self._sigma()


    def quadtree(self, epsilon=0.0029, tile_size_max=0.02, tile_size_min=0.002, nan_allowed=0.9):
        sc = Scene.load(self.kite_file)

        print("-" * 50)
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

    # TODO Change to adapt to quadtree and others
    def _write_file(self):
        """Write downsampled data to a file using mintpy writefile utility."""
        new_shape = tuple(self.imshow.shape)
        if len(new_shape) >= 2:
            nx, ny = int(new_shape[-1]), int(new_shape[-2])
            for k in ('width','WIDTH'):
                if k in self.metadata:
                    self.metadata[k] = np.int64(nx)
                if k in self.geometry_metadata:
                    self.geometry_metadata[k] = np.int64(nx)
            for k in ('length', 'LENGTH'):
                if k in self.metadata:
                    self.metadata[k] = np.int64(ny)
                if k in self.geometry_metadata:
                    self.geometry_metadata[k] = np.int64(ny)
        # write velocity
        file_path = self.velocity_file.replace('.h5', '_downsampled.h5')
        datasetDict = {'velocity': self.imshow}
        self.metadata['FILE_PATH'] = file_path
        writefile.write(datasetDict, metadata=self.metadata, out_file=file_path)
        # write geometry (use self.geometry_file, not self.geom)
        file_path = self.geometry_file.replace('.h5', '_downsampled.h5')
        datasetDict = {
            'incidenceAngle': self.sliced_incident_angle,
            'azimuthAngle': self.sliced_azimuth_angle,
            'latitude': self.sliced_y,
            'longitude': self.sliced_x,
            'height': self.sliced_height,
        }
        self.geometry_metadata['FILE_PATH'] = file_path
        writefile.write(datasetDict, metadata=self.geometry_metadata, out_file=file_path)


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


    def _sigma(self):
        """Calculate the standard deviation of the downsampled velocity data."""
        if self.sliced_temporal_coherence is not None:
            sigma_min = 0.002
            sigma_max = 0.02
            p = 2
            sigma = sigma_min + (sigma_max - sigma_min) * (1 - self.sliced_temporal_coherence) ** p
            self.err = sigma
        else:
            self.err = np.full(len(self.z), 0.005)