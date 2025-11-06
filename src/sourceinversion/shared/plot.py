import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from sourceinversion.shared.helper_functions import convert_to_utm

def create_panel(ax, x, y, values, title, cmap, vmin, vmax, size=15, sources=None):
    """Draw a scatter panel with optional sources overlay."""
    img = ax.scatter(x, y, size, values, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img, orientation='horizontal', ax=ax)
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


class InversionPlotter:
    def __init__(self, inps, east, north, data, synth, deformation, model, sources=None, period=None, latitude=None, longitude=None, bbox=None):
        self.east = east
        self.north = north
        self.data = data
        self.synth = synth
        self.deformation = deformation
        self.model = model
        self.sources = sources
        self.period = period
        self.latitude = latitude
        self.longitude = longitude
        self.bbox = bbox
        self.inps = inps

    def _plot_bbox(self, ax):
        if getattr(self.inps, "bbox", False):
            for x, y in zip(self.inps.x, self.inps.y):
                x_min, x_max = x
                y_min, y_max = y
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="black", facecolor="none", alpha=0.3)
                ax.add_patch(rect)

    def plot(self):
        residuals = self.data - self.synth
        high_val = max(np.abs(self.data)) * 1.1
        color_min, color_max = -high_val, high_val

        # choose layout dynamically
        if self.inps.fullres and self.deformation is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            top_axes = axes[0, :]
            bottom_axes = axes[1, :]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            top_axes = axes
            bottom_axes = None

        fig.suptitle(f"Model: {', '.join(map(str, self.model))}"+ (f", Period: {self.period.replace('_', ' ')}" if self.period else ""),fontsize=10,)

        # top row
        create_panel(top_axes[0], self.east, self.north, self.data, "Data", "jet", color_min, color_max, sources=self.sources)
        self._plot_bbox(top_axes[0])

        create_panel(top_axes[1], self.east, self.north, self.synth, "Model", "jet", color_min, color_max, sources=self.sources)

        create_panel(top_axes[2], self.east, self.north, residuals, "Residual", "bwr", color_min/4, color_max/4, sources=self.sources)

        # optional deformation row
        if bottom_axes is not None:
            self._plot_deformation(bottom_axes, color_min, color_max)

        return fig

    def _plot_deformation(self, axes, color_min, color_max):
        #Interpolated result
        xx, yy = convert_to_utm(longitude=self.longitude, latitude=self.latitude)
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
        ax.set_aspect('equal', adjustable='datalim')


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