import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata


def plot_results(inps, east, north, data, synth, sources_center, model, period=None, deformation=None):
    """
    Plot the results of the inversion.
    """
    def create_panel(ax, x, y, values, title, cmap, vmin, vmax, size=15, sources_center=None):
        img = ax.scatter(x, y, size, values, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img, orientation='horizontal', ax=ax)
        cbar.set_label('LOS (m)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(title, fontsize=16, pad=10)
        # Plot sources if available
        if sources_center:
            x_values = [sublist[0] for sublist in sources_center]
            y_values = [sublist[1] for sublist in sources_center]
            ax.scatter(x_values, y_values, s=15, c=x_values, cmap='Set3', edgecolors='black', linewidth=0.5, marker='8')

    residuals = data - synth
    fig = plt.figure(figsize=(15, 10))

    fig.suptitle(f"Model: {', '.join(str(m) for m in model)}" + (f", Period: {period.replace('_', ' ')}" if period else ""), fontsize=10)

    # Calculate color limits based on the data panel
    color_min = -max(data)
    color_max = max(data)

    # Panel Data
    ax = plt.subplot(231)
    create_panel(ax, east, north, data, 'Data', 'jet', color_min, color_max, sources_center=sources_center)

    if inps.bbox:
        x_min, x_max = inps.x_range
        y_min, y_max = inps.y_range
        rect = Rectangle((x_min, y_min), (x_max-x_min), (y_max-y_min), linewidth=2, edgecolor='black', facecolor='none', alpha=0.3)
        ax.add_patch(rect)

    # Panel Model
    ax1 = plt.subplot(232)
    create_panel(ax1, east, north, synth, 'Model', 'jet', color_min, color_max, sources_center=sources_center)

    # Panel Residuals
    ax2 = plt.subplot(233)
    create_panel(ax2, east, north, residuals, 'Residual', 'bwr', color_min/5, color_max/5, sources_center=sources_center)

    ########################### FULL RESOLUTION ###########################
    ax3 = plt.subplot(234)
    ax3.imshow(deformation, cmap='jet', vmin=color_min, vmax=color_max)

    mask = ~np.isnan(deformation)
    deformation_shape = deformation.shape

    # Create a grid matching the resolution of deformation
    x = np.linspace(np.min(east), np.max(east), deformation_shape[1])
    y = np.linspace(np.min(north), np.max(north), deformation_shape[0])
    grid_x, grid_y = np.meshgrid(x, y)

    # Interpolate synth data to match the deformation grid
    synth_interpolated = np.flipud(griddata(points=(east, north), values=synth, xi=(grid_x, grid_y),method='linear'))
    synth_interpolated_masked = np.where(mask, synth_interpolated, np.nan)
    # Use the interpolated data in ax3.imshow
    ax4 = plt.subplot(235)
    ax4.imshow(synth_interpolated_masked, cmap='jet', vmin=color_min, vmax=color_max)

    ax.tick_params(axis='both', which='minor', direction='out', length=5, width=2, grid_color='b', grid_alpha=0.5)

    ax5 = plt.subplot(236)
    ax5.imshow((deformation - synth_interpolated_masked), cmap='bwr', vmin=color_min/5, vmax=color_max/5)

    return fig