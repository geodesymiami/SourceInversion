import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LightSource
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
        high_val = max(np.abs(self.data)) * 1.1
        color_min, color_max = -high_val, high_val

        # choose layout dynamically
        if self.inps.fullres and self.deformation is not None:
            fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
            top_axes = axes[0, :]
            bottom_axes = axes[1, :]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(20, 4), constrained_layout=True)
            top_axes = axes
            bottom_axes = None

        fig.suptitle(f"Model: {', '.join(map(str, self.model))}"+ (f", Period: {self.period.replace('_', ' ')}" if self.period else ""),fontsize=10,)

        for i, line in enumerate(self.properties.keys()):
            self._create_panel(top_axes[i], self.east, self.north, self.properties[line]["data"], line, self.properties[line]["cmap"], color_min, color_max, sources=self.sources, size=self.inps.size)

        for ax in top_axes[1 :]:
            ax.set_yticklabels([])

        for ax in top_axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_box_aspect(0.5)

        # optional deformation row
        if bottom_axes is not None:
            self._plot_deformation(bottom_axes, color_min, color_max)
            for ax in bottom_axes[1 :]:
                ax.set_yticklabels([])

            for ax in bottom_axes:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.set_box_aspect(0.5)

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

    def _plot_deformation(self, axes, color_min, color_max):
        #Interpolated result
        xx, yy = convert_to_utm(longitude=[np.nanmin(self.longitude), np.nanmax(self.longitude)], latitude=self.latitude)
        x = np.linspace(np.min(xx), np.max(xx), self.deformation.shape[1])
        y = np.linspace(np.max(yy), np.min(yy), self.deformation.shape[0])
        grid_x, grid_y = np.meshgrid(x, y)

        valid_mask = ~np.isnan(self.deformation)
        z_flat = self.deformation.flatten()
        x_flat, y_flat = grid_x.flatten(), grid_y.flatten()

        synth_interp = griddata((self.east, self.north), self.synth, (grid_x, grid_y), method="linear")
        synth_masked = synth_interp[valid_mask]

        diff = z_flat - synth_interp.flatten()

        axes[0].scatter(x_flat, y_flat, c=z_flat, cmap="jet", vmin=color_min, vmax=color_max, s=1)

        axes[1].scatter(grid_x[valid_mask], grid_y[valid_mask], c=synth_masked, cmap="jet", vmin=color_min, vmax=color_max, s=1)

        axes[2].scatter(x_flat, y_flat, c=diff, cmap="bwr", vmin=color_min/5, vmax=color_max/5, s=1)

    def _create_panel(self, ax, x, y, values, title, cmap, vmin, vmax, size=15, sources=None):
        """Draw a scatter panel with optional sources overlay, using self.sources by default."""
        sources = sources if sources is not None else self.sources
        longitude, latitude = convert_to_utm(longitude=self.longitude, latitude=self.latitude)

        if self.geometry is not None:
           self._plot_dem(ax, longitude, latitude)

        if self.inps.style=='image':
            nrows, ncols = int(self.inps.length), int(self.inps.width)

            xx = np.linspace(np.min(longitude), np.max(longitude), ncols)
            yy = np.linspace(np.min(latitude), np.max(latitude), nrows)

            Xi, Yi = np.meshgrid(xx, yy)

            grid = griddata((x, y), values, (Xi, Yi), method="linear")
            data = np.where(np.flipud(self.inps.mask), grid, np.nan) if hasattr(self.inps, 'mask') else grid
            img = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax , extent=(np.min(longitude), np.max(longitude), np.min(latitude), np.max(latitude)), alpha=0.8)

        elif self.inps.style=='scatter':
            img = ax.scatter(x, y, s=size, c=values, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='none',)

        cbar = plt.colorbar(img, orientation='horizontal', ax=ax, shrink=0.4)
        img.set_alpha(1)
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.set_label("LOS (m)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(title, fontsize=16, pad=10)

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

        return ax

    def _plot_dem(self, ax, x, y):
        lat = np.linspace(np.min(y), np.max(y), self.geometry.shape[0])
        lon = np.linspace(np.min(x), np.max(x), self.geometry.shape[1])
        lon2d, lat2d = np.meshgrid(lon, lat)
        dlon, dlat = lat[1] - lat[0], lon[1] - lon[0]

        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(self.geometry, vert_exag=1, dx=dlon, dy=dlat)

        # Use pcolormesh to plot hillshade using real coordinates
        ax.pcolormesh(lon2d,lat2d,np.flipud(hillshade),cmap='gray',shading='auto', zorder=0)

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

        rectangle = Rectangle(
            (self.xtlc, self.ytlc),         # Bottom-left corner
            self.length,                    # Length of the rectangle
            projected_width,                     # Width of the rectangle
            angle=self.strike - 90,         # Rotation angle (strike)
            # edgecolor='black',              # Edge color
            facecolor='black',               # Transparent fill
            lw=1,                           # Line width
            alpha=0.5
        )
        ax.add_patch(rectangle)
        ax.set_aspect('equal', adjustable='datalim')