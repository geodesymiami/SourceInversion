import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from sourceinversion.shared.helper_functions import convert_to_utm

def create_panel(ax, x, y, values, title, cmap, vmin, vmax, size=15, sources_center=None):
    """Draw a scatter panel with optional sources overlay."""
    img = ax.scatter(x, y, size, values, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img, orientation='horizontal', ax=ax)
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_label("LOS (m)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(title, fontsize=16, pad=10)

    if sources_center:
        xs, ys = zip(*sources_center)
        ax.scatter(xs, ys, s=15, c=xs, cmap="Set3", edgecolors="black", linewidth=0.5, marker="8")
    return ax


class InversionPlotter:
    def __init__(self, inps, east, north, data, synth, deformation, model, sources_center=None, period=None, latitude=None, longitude=None, bbox=None):
        self.east = east
        self.north = north
        self.data = data
        self.synth = synth
        self.deformation = deformation
        self.model = model
        self.sources_center = sources_center
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
        high_val = max(np.abs(self.data))
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
        create_panel(top_axes[0], self.east, self.north, self.data, "Data", "jet", color_min, color_max, sources_center=self.sources_center)
        self._plot_bbox(top_axes[0])

        create_panel(top_axes[1], self.east, self.north, self.synth, "Model", "jet", color_min, color_max, sources_center=self.sources_center)

        create_panel(top_axes[2], self.east, self.north, residuals, "Residual", "bwr", color_min/5, color_max/5, sources_center=self.sources_center)

        # optional deformation row
        if bottom_axes is not None:
            self._plot_deformation(bottom_axes, color_min, color_max)

        return fig

    def _plot_deformation(self, axes, color_min, color_max):
        xx, yy = convert_to_utm(longitude=self.longitude, latitude=self.latitude)
        x = np.linspace(np.min(xx), np.max(xx), self.deformation.shape[1])
        y = np.linspace(np.max(yy), np.min(yy), self.deformation.shape[0])
        grid_x, grid_y = np.meshgrid(x, y)

        valid_mask = ~np.isnan(self.deformation)
        z_flat = self.deformation.flatten()
        x_flat, y_flat = grid_x.flatten(), grid_y.flatten()

        axes[0].scatter(x_flat, y_flat, c=z_flat, cmap="jet", vmin=color_min, vmax=color_max, s=1)

        synth_interp = griddata((self.east, self.north), self.synth, (grid_x, grid_y), method="linear")
        synth_masked = synth_interp[valid_mask]

        axes[1].scatter(grid_x[valid_mask], grid_y[valid_mask], c=synth_masked, cmap="jet", vmin=color_min, vmax=color_max, s=1)

        diff = z_flat - synth_interp.flatten()
        axes[2].scatter(x_flat, y_flat, c=diff, cmap="bwr", vmin=color_min/5, vmax=color_max/5, s=1)

# def plot_results(inps, east, north, data, synth, sources_center, model, period=None, deformation=None, latitude=None, longitude=None):
#     """
#     Plot the results of the inversion.
#     """
#     def create_panel(ax, x, y, values, title, cmap, vmin, vmax, size=15, sources_center=None):
#         img = ax.scatter(x, y, size, values, cmap=cmap, vmin=vmin, vmax=vmax)
#         cbar = plt.colorbar(img, orientation='horizontal', ax=ax)
#         cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax]) 
#         cbar.set_label('LOS (m)')
#         ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
#         ax.set_title(title, fontsize=16, pad=10)
#         # Plot sources if available
#         if sources_center:
#             x_values = [sublist[0] for sublist in sources_center]
#             y_values = [sublist[1] for sublist in sources_center]
#             ax.scatter(x_values, y_values, s=15, c=x_values, cmap='Set3', edgecolors='black', linewidth=0.5, marker='8')

#     residuals = data - synth
#     fig = plt.figure(figsize=(15, 10))

#     fig.suptitle(f"Model: {', '.join(str(m) for m in model)}" + (f", Period: {period.replace('_', ' ')}" if period else ""), fontsize=10)

#     # Calculate color limits based on the data panel
#     color_min = -max(data)
#     color_max = max(data)

#     # Panel Data
#     ax = plt.subplot(231)
#     create_panel(ax, east, north, data, 'Data', 'jet', color_min, color_max, sources_center=sources_center)

#     if inps.bbox:
#         for x, y in zip(inps.x, inps.y):
#             x_min, x_max = x
#             y_min, y_max = y
#             rect = Rectangle((x_min, y_min), (x_max-x_min), (y_max-y_min), linewidth=2, edgecolor='black', facecolor='none', alpha=0.3)
#             ax.add_patch(rect)

#     # Panel Model
#     ax1 = plt.subplot(232)
#     create_panel(ax1, east, north, synth, 'Model', 'jet', color_min, color_max, sources_center=sources_center)

#     # Panel Residuals
#     ax2 = plt.subplot(233)
#     create_panel(ax2, east, north, residuals, 'Residual', 'bwr', color_min/5, color_max/5, sources_center=sources_center)

#     ########################### FULL RESOLUTION ###########################
#     # Create a grid matching the resolution of deformation
#     xx, yy = convert_to_utm(longitude=longitude, latitude=latitude)
#     x = np.linspace(np.min(xx), np.max(xx), deformation.shape[1])
#     y = np.linspace(np.max(yy), np.min(yy), deformation.shape[0])
#     grid_x, grid_y = np.meshgrid(x, y)

#     # Flatten the grid and deformation data
#     x_flat = grid_x.flatten()
#     y_flat = grid_y.flatten()
#     z_flat = deformation.flatten()

#     # Plot using scatter
#     ax3 = plt.subplot(234)
#     ax3.scatter(x_flat, y_flat, c=z_flat, cmap='jet', vmin=color_min, vmax=color_max, s=1)
#     ax3.set_xlim(np.min(east), np.max(east))
#     ax3.set_ylim(np.min(north), np.max(north))

#     # Ensure valid_mask matches the shape of deformation
#     valid_mask = ~np.isnan(deformation)

#     # Flatten and mask grid_x, grid_y, and deformation
#     x_flat_mskd = grid_x[valid_mask]
#     y_flat_mskd = grid_y[valid_mask]

#     # Interpolate synth data to match the deformation grid
#     synth_interpolated = griddata(points=(east, north), values=synth, xi=(grid_x, grid_y), method='linear')

#     # Apply the same mask to synth_interpolated
#     synth_interpolated_masked = synth_interpolated[valid_mask]

#     # Plot using scatter
#     ax4 = plt.subplot(235)
#     ax4.scatter(x_flat_mskd, y_flat_mskd, c=synth_interpolated_masked, cmap='jet', vmin=color_min, vmax=color_max, s=1)

#     # Compute the difference
#     difference = (z_flat - synth_interpolated.flatten())

#     # Plot the scatter plot
#     ax5 = plt.subplot(236)
#     ax5.scatter(x_flat, y_flat, c=difference, cmap='bwr', vmin=color_min/5, vmax=color_max/5, s=1)

#     ax.tick_params(axis='both', which='minor', direction='out', length=5, width=2, grid_color='b', grid_alpha=0.5)
#     ###########################################################################

#     return fig