import numpy as np


def spatial_correlation_maps(map1, map2, mask=None):
    """
    Compare two deformation maps (same area, different periods)
    to measure if the Spatial pattern of deformation stayed the same.

    Parameters
    ----------
    map1, map2 : 2D np.ndarray
        Deformation maps (e.g., LOS velocity, cumulative displacement)
    mask : 2D np.ndarray, optional
        Boolean mask to exclude invalid pixels

    Returns
    -------
    corr : float
        Spatial Pearson correlation (-1 to 1)

    Interpretation
    --------------c
    High corr (≥ 0.7): similar geometry → same source
    Low corr (< 0.5): pattern changed → possible new/migrating source
    """

    if mask is not None:
        map1, map2 = map1[mask], map2[mask]
    valid = np.isfinite(map1) & np.isfinite(map2)
    corr = np.corrcoef(map1[valid], map2[valid])[0, 1]

    print('-' * 50)
    if corr >= 0.7:
        print("High correlation (≥ 0.7): similar geometry → same source\n")
    elif corr >= 0.5:
        print("Moderate correlation (0.5-0.7): partially similar geometry → evolving or mixed source\n")
    else:
        print("Low correlation (< 0.5): pattern changed → possible new/migrating source\n")

    return corr


def centroid_shift_maps(map1, map2, x, y, mask=None):
    """
    Compute the centroid shift of deformation between two maps (same area, different periods).

    Parameters
    ----------
    map1, map2 : 2D np.ndarray
        Deformation maps (same shape, aligned)
    x, y : 2D np.ndarray
        Coordinate grids (same size as maps)
    mask : 2D np.ndarray, optional
        Boolean mask to exclude invalid pixels

    Returns
    -------
    shift : float
        Distance between centroids (adimensional)
    centroids : tuple
        ((x1, y1), (x2, y2)) centroids of each map

    Interpretation
    --------------
    Small shift (< few hundred units): same source
    Large shift (> 1 unit): source migration or multiple sources
    """

    if mask is not None:
        map1, map2, x, y = map1[mask], map2[mask], x[mask], y[mask]
    valid = np.isfinite(map1) & np.isfinite(map2)
    w1, w2 = np.abs(map1[valid]), np.abs(map2[valid])
    cx1, cy1 = np.average(x[valid], weights=w1), np.average(y[valid], weights=w1)
    cx2, cy2 = np.average(x[valid], weights=w2), np.average(y[valid], weights=w2)
    shift = np.hypot(cx2 - cx1, cy2 - cy1)

    if shift < 100:  # assuming adimensional units
        print("Small shift (< few hundred units): same source")
    else:
        print("Large shift (> 1 unit): source migration or multiple sources")

    return shift, ((cx1, cy1), (cx2, cy2))