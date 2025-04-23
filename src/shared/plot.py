import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_results(east, north, data, synth):
    """
    Plot the residualsults of the inversion.
    """
    residuals = data - synth

    fig=plt.figure(figsize=(15,6))

    ## PANEL DATA ##########
    ax = plt.subplot(131)
    img = ax.scatter(east, north,5, data ,cmap='viridis', vmin= -max(data), vmax = max(data))
    #palette
    cbar = plt.colorbar(img,orientation='horizontal')
    cbar.set_label('LOS (m)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


    plt.title('Data',fontsize = 16, pad=10)

    ## PANEL MODEL ##########
    ax1 = plt.subplot(132)
    img = ax1.scatter(east, north,5, synth ,cmap='viridis', vmin= -max(synth), vmax = max(synth))
    #palette
    cbar = plt.colorbar(img,orientation='horizontal')
    cbar.set_label('LOS (m)')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))

    plt.title('Model',fontsize = 16, pad=10)

    ## PANEL residualsIDUALS ##########
    ax2 = plt.subplot(133)
    img = ax2.scatter(east, north,5, residuals,cmap="bwr",vmin= -0.1, vmax = 0.1)
    #palette
    cbar = plt.colorbar(img,orientation='horizontal')
    cbar.set_label('LOS (m)')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Title for plot
    plt.title('residual',fontsize = 16, pad=10)

    index = np.where(synth == max(synth))
    ax1.scatter(east[index], north[index],s=50, c='black', marker='x')

    ax.tick_params(axis='both', which='minor', direction='out', length=5, width=2, grid_color='b', grid_alpha=0.5)

    plt.show()