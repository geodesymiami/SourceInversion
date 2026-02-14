import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LightSource
from matplotlib.transforms import Affine2D
from sourceinversion.shared.helper_functions import convert_to_utm, resize_to_match


class InversionPlotter:
    def __init__(self, metadata, inps, east, north, data, geometry, synth, deformation, model, sources=None, period=None, latitude=None, longitude=None, bbox=None):
        self.metadata = metadata
        self.east = east
        self.north = north
        self.data = data
        self.geometry = geometry
        self.synth = synth
        self.deformation = deformation
        self.model = model
        self.sources = sources
        self.period = period
        self.latitude = latitude
        self.longitude = longitude
        self.bbox = bbox
        self.inps = inps

        self.properties = {
            "Data": {"cmap": "jet",
                     "data": self.data,},
            "Model": {"cmap": "jet",
                      "data": self.synth},
            "Residual": {"cmap": "bwr",
                         "data": self.data - self.synth},
        }


    def plot(self):
        # choose layout dynamically
        if self.inps.fullres and self.deformation is not None:
            fig, axes = plt.subplots(2, 3, figsize=(20, 7), constrained_layout=True)
            top_axes = axes[0, :]
            bottom_axes = axes[1, :]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(20, 4), constrained_layout=True)
            top_axes = axes
            bottom_axes = None

        fig.suptitle(f"Model: {', '.join(map(str, self.model))}"+ (f", Period: {self.period.replace('_', ' ')}" if self.period else ""),fontsize=10,)

        for i, line in enumerate(self.properties.keys()):
            self._create_panel(top_axes[i], self.east, self.north, self.properties[line]["data"], line, self.properties[line]["cmap"], sources=self.sources, size=self.inps.size)

        for ax in top_axes[1 :]:
            ax.set_yticklabels([])

        for ax in top_axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            # ax.set_aspect('auto')
            ax.set_box_aspect(0.5)

        # optional deformation row
        if bottom_axes is not None:
            self._plot_deformation(bottom_axes, sources=self.sources)
            for ax in bottom_axes[1 :]:
                ax.set_yticklabels([])

            for ax in bottom_axes:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.set_box_aspect(0.5)
                # ax.set_aspect('auto')

            for ax in top_axes:
                ax.set_xticklabels([])

        self._plot_bbox(top_axes[0])

        # Apply zoom/subsection if requested in inputs
        # inps.subsection may be a 4-tuple/list (xmin, xmax, ymin, ymax) or a
        # string like 'xmin,xmax,ymin,ymax' or 'xmin:xmax,ymin:ymax'.
        try:
            self._apply_zoom(top_axes)
            if bottom_axes is not None:
                self._apply_zoom(bottom_axes)
        except Exception:
            # keep plotting even if zoom fails
            pass

        return fig

    def _plot_bbox(self, ax):
        if getattr(self.inps, "bbox", False):
            for x, y in zip(self.inps.x, self.inps.y):
                x_min, x_max = x
                y_min, y_max = y
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="black", facecolor="none", alpha=0.3)
                ax.add_patch(rect)

    def _plot_deformation(self, axes, sources=None):
        xx, yy = convert_to_utm(longitude=[np.nanmin(self.longitude), np.nanmax(self.longitude)], latitude=self.latitude)
        x = np.linspace(np.min(xx), np.max(xx), self.deformation.shape[1])
        y = np.linspace(np.max(yy), np.min(yy), self.deformation.shape[0])
        grid_x, grid_y = np.meshgrid(x, y)

        # interpolate synth on same grid and compute diff
        synth_interp = griddata((self.east, self.north), self.synth, (grid_x, grid_y), method="linear")
        diff_grid = self.deformation - synth_interp
        synth_interp = np.where((~np.isnan(self.deformation)), synth_interp, np.nan)
        # extent for imshow (half-pixel correction)
        nrows, ncols = self.deformation.shape
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        dx = (x[1] - x[0]) if ncols > 1 else (xmax - xmin)
        dy = (y[1] - y[0]) if nrows > 1 else (ymax - ymin)
        extent = (xmin - 0.5 * dx, xmax + 0.5 * dx, ymin - 0.5 * dy, ymax + 0.5 * dy)

        style = getattr(self.inps, "style", "scatter")

        # list of panels: (axis_index, 2D_array, cmap, vmin, vmax)
        panels = [
            (0, self.deformation, "jet", 1),
            (1, synth_interp, "jet", 1),
            (2, diff_grid, "bwr" , 1),
        ]

        for idx, arr2d, cmap, factor in panels:
            ax = axes[idx]
            # draw DEM and isolines underneath
            self._plot_dem(ax, grid_x.flatten(), grid_y.flatten())
            self._plot_isolines(ax)

            if style == "image":
                a = np.array(arr2d, copy=True)
                a[~np.isfinite(a)] = np.nan
                img = ax.imshow(a, origin="upper", cmap=cmap, extent=extent, alpha=0.8, interpolation="bilinear", vmin=self.inps.vlim[0]/factor, vmax=self.inps.vlim[1]/factor,)
            else:
                # scatter: flatten and mask invalid cells
                vals = arr2d.ravel()
                xs = grid_x.ravel()
                ys = grid_y.ravel()
                mask = np.isfinite(vals)
                img = ax.scatter(xs[mask], ys[mask], c=vals[mask], cmap=cmap, vmin=self.inps.vlim[0]/factor, vmax=self.inps.vlim[1]/factor, s=1,)

            img.set_alpha(0.7)
            self._plot_source(ax, sources)

    def _create_panel(self, ax, x, y, values, title, cmap, size=15, sources=None):
        """Draw a scatter panel with optional sources overlay, using self.sources by default."""
        sources = sources if sources is not None else self.sources
        longitude, latitude = convert_to_utm(longitude=self.longitude, latitude=self.latitude)

        if self.geometry is not None:
           self._plot_dem(ax, longitude, latitude)
           self._plot_isolines(ax)

        vmax  = np.nanmax(np.abs(self.data))
        self.inps.vlim  = [-vmax, vmax] if self.inps.vlim is None else self.inps.vlim

        if self.inps.style=='image':
            nrows, ncols = int(self.inps.length), int(self.inps.width)

            xx = np.linspace(np.min(longitude), np.max(longitude), ncols)
            yy = np.linspace(np.min(latitude), np.max(latitude), nrows)

            Xi, Yi = np.meshgrid(xx, yy)

            grid = griddata((x, y), values, (Xi, Yi), method="nearest")
            data = np.where(np.flipud(self.inps.mask), grid, np.nan) if hasattr(self.inps, 'mask') else grid
            img = ax.imshow(data, origin='lower', cmap=cmap, vmin=self.inps.vlim[0], vmax=self.inps.vlim[1] , extent=(np.min(longitude), np.max(longitude), np.min(latitude), np.max(latitude)), alpha=0.8)

        elif self.inps.style=='scatter':
            img = ax.scatter(x, y, s=size, c=values, cmap=cmap, vmin=self.inps.vlim[0], vmax=self.inps.vlim[1], edgecolors='none',)

        label = self.metadata['passDirection'] if 'passDirection' in self.metadata else None
        if label is None:
            label = 'DESCENDING' if 'SenD' in self.metadata['FILE_PATH'] else 'ASCENDING' if 'SenA' in self.metadata['FILE_PATH'] else ''
        if label:
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=5, fontweight='bold', color='white', verticalalignment='top', bbox=dict(facecolor='black', edgecolor='none', pad=3, alpha=0.5))

        cbar = plt.colorbar(img, orientation='horizontal', ax=ax, shrink=0.4)
        img.set_alpha(0.8)
        cbar.set_ticks([self.inps.vlim[0], (self.inps.vlim[0] + self.inps.vlim[1]) / 2, self.inps.vlim[1]])
        cbar.set_label("LOS (m)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(title, fontsize=16, pad=10)

        self._plot_source(ax, sources)

        return ax

    def _plot_source(self, ax, sources):
        if sources:
            source_type = {
                "mogi": {"class": Mogi, "attributes": ["xcen", "ycen"]},
                "spheroid": {"class": Spheroid, "attributes": ["xcen", "ycen", "s_axis_max", "ratio", "strike", "dip"]},
                "penny": {"class": Penny,  "attributes": ["xcen", "ycen", "radius"]},
                "okada": {"class": Okada,  "attributes": ["ytlc", "xtlc", "length", "width", "strike", "dip"]},
            }
            for s in sources:
                s_keys = set(sources[s].keys())

                for key, value in source_type.items():
                    if set(value["attributes"]) == s_keys:
                        model = value["class"]
                        model(ax, **sources[s])

    def _plot_dem(self, ax, x, y):
        self.lat = np.linspace(np.min(y), np.max(y), self.geometry.shape[0]) if not hasattr(self, 'lat') else self.lat
        self.lon = np.linspace(np.min(x), np.max(x), self.geometry.shape[1]) if not hasattr(self, 'lon') else self.lon
        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        dlon, dlat = self.lat[1] - self.lat[0], self.lon[1] - self.lon[0]

        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(self.geometry, vert_exag=0.5, dx=dlon, dy=dlat)

        # Use pcolormesh to plot hillshade using real coordinates
        ax.pcolormesh(lon2d,lat2d,np.flipud(hillshade),cmap='gray',shading='auto', zorder=0)

    def _plot_isolines(self, ax):
        grid_np = self.geometry

        # Remove negative values
        grid_np[grid_np < 0] = 0

        # Convert the numpy array back to a DataArray
        lines = xr.DataArray(grid_np, dims=["lat", "lon"], coords={"lat": self.lat, "lon": self.lon},)

        # Extract coordinates and elevation values
        lon = lines.coords["lon"].values
        lat = lines.coords["lat"].values
        z = np.flipud(lines.values)

        #Plot the isolines
        cont = ax.contour(lon, lat, z, levels=10, colors='black', linewidths=1, alpha=0.3)

        if hasattr(self, 'inline') and self.inline:
            self.ax.clabel(cont, inline=self.inline, fontsize=8)

    def _apply_zoom(self, axes):
        """Apply zoom or subsection to a set of axes.

        - If `self.inps.zoom` is provided (float > 1), zoom in by that factor around data center.
        - If `self.inps.subsection` is provided, it can be a tuple/list of 4 floats
          (xmin, xmax, ymin, ymax) or a comma/colon separated string. We apply
          those limits to all axes passed in `axes`.
        """
        # normalize axes list
        if not hasattr(axes, '__iter__'):
            axes = [axes]

        # helper to set limits for each axis
        def set_limits(ax , x0, x1, y0, y1):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

        # 1) subsection has priority over zoom
        sub = getattr(self.inps, 'subset', None)
        if sub:
            # accept tuple/list or string
            if isinstance(sub, (list, tuple)) and len(sub) == 4:
                xmin, xmax, ymin, ymax = map(float, sub)
            else:
                # parse strings like 'xmin,xmax,ymin,ymax' or 'xmin:xmax,ymin:ymax'
                s = str(sub)
                s = s.replace(':', ',')
                parts = [p for p in re.split('[,\s]+', s) if p]
                if len(parts) != 4:
                    raise ValueError('Invalid subsection format; expected 4 values')
                xmin, xmax, ymin, ymax = map(float, parts)

            for ax in axes:
                set_limits(ax, xmin, xmax, ymin, ymax)
            return

        # 2) zoom factor
        z = getattr(self.inps, 'zoom', None)
        if z:
            try:
                factor = float(z)
            except Exception:
                return

            # compute data center from east/north
            cx = 0.5 * (np.nanmin(self.east) + np.nanmax(self.east))
            cy = 0.5 * (np.nanmin(self.north) + np.nanmax(self.north))
            full_w = np.nanmax(self.east) - np.nanmin(self.east)
            full_h = np.nanmax(self.north) - np.nanmin(self.north)
            w = full_w / factor
            h = full_h / factor

            for ax in axes:
                set_limits(ax, cx - 0.5 * w, cx + 0.5 * w, cy - 0.5 * h, cy + 0.5 * h)


class Mogi():
    def __init__(self, ax, xcen, ycen):
        self.x = xcen
        self.y = ycen
        self._plot_source(ax)

    def _plot_source(self, ax):
        ax.scatter(self.x, self.y, s=15, color="black", linewidth=2, marker="x")


class Spheroid():
    def __init__(self, ax, xcen, ycen, s_axis_max, ratio, strike, dip):
        self.x = xcen
        self.y = ycen
        self.s_axis = s_axis_max
        self.ratio = ratio
        self.strike = strike
        self.dip = dip
        self._plot_source(ax)

    def _plot_source(self, ax):
        # Calculate semi-minor axis
        s_minor = self.s_axis * self.ratio

        # Convert angles to radians
        strike_rad = np.radians(self.strike - 90)
        dip_rad = np.radians(self.dip)

        # Adjust the semi-major axis length for the dip projection
        s_axis_projected = self.s_axis * np.sin(dip_rad)

        # Calculate endpoints of the major axis (with dip projection)
        dx_major = s_axis_projected * np.cos(strike_rad)
        dy_major = s_axis_projected * np.sin(strike_rad)
        x_major = [self.x - dx_major, self.x + dx_major]
        y_major = [self.y - dy_major, self.y + dy_major]

        # Calculate endpoints of the minor axis (without dip projection)
        dx_minor = s_minor * np.sin(strike_rad)
        dy_minor = s_minor * -np.cos(strike_rad)
        x_minor = [self.x - dx_minor, self.x + dx_minor]
        y_minor = [self.y - dy_minor, self.y + dy_minor]

        ax.plot(x_major, y_major, 'r-', label='Major Axis')  # Major axis in red
        ax.plot(x_minor, y_minor, 'b-', label='Minor Axis')  # Minor axis in blue


class Penny():
    def __init__(self, ax, xcen, ycen, radius):
        self.x = xcen
        self.y = ycen
        self.radius = radius
        self._plot_source(ax)

    def _plot_source(self, ax):
        circle = plt.Circle((self.x, self.y), self.radius, edgecolor='black', color="#7cc0ff", fill=True, alpha=0.7, label='Penny')
        ax.add_patch(circle)


class Okada:
    def __init__(self, ax, xtlc, ytlc, length, width, strike, dip):
        self.xtlc = xtlc
        self.ytlc = ytlc
        self.length = length
        self.width = width
        self.strike = strike
        self.dip = dip
        self._plot_source(ax)

    def _plot_source(self, ax):
        dip_radians = np.radians(self.dip)
        projected_width = self.width * np.cos(dip_radians)
        height = abs(projected_width)

        local_rect = Rectangle((0.0, -height),
                               self.length, height,
                            #    facecolor='black',
                               edgecolor='black',
                               lw=1,
                               alpha=0.2)
        # rotate around local origin (top-left) and translate to (xtlc, ytlc)
        t = Affine2D().rotate_deg(90 - self.strike).translate(self.xtlc, self.ytlc)
        local_rect.set_transform(t + ax.transData)
        ax.add_patch(local_rect)

        # add a single spike/triangle along the length that points in the down-dip direction
        # local coordinates: top edge is at y=0, down-dip is negative y
        try:
            from matplotlib.patches import Polygon
            # main triangle size
            base_half_main = max(0.04 * self.length, 0.01 * self.length)

            tri_color = 'black'
            tri_edge = 'black'
            tri_alpha = 0.3

            # add several smaller triangles along the fault length with same color/alpha
            n_extra = 6
            extra_positions = np.linspace(0.1, 0.9, n_extra)
            base_half_small = base_half_main * 0.5
            tip_offset_small = 0.6
            for pos in extra_positions:
                # skip center position to avoid overlapping the main triangle
                if abs(pos - 0.5) < 1e-6:
                    continue
                left = (pos * self.length - base_half_small, 0.0)
                right = (pos * self.length + base_half_small, 0.0)
                tip = (pos * self.length, -height * tip_offset_small)
                tri_small = Polygon([left, right, tip], closed=True,
                                    facecolor=tri_color, edgecolor=tri_edge, linewidth=0.6,
                                    zorder=29, alpha=tri_alpha)
                tri_small.set_transform(t + ax.transData)
                ax.add_patch(tri_small)
        except Exception:
            # non-fatal: continue without spikes if something goes wrong
            pass

        ax.set_aspect('equal', adjustable='datalim')