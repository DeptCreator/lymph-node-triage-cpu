"""Singleton suppression: drop isolated hot tiles (connectivity filter)."""
import numpy as np
from scipy.ndimage import label as clabel


def filter_hot(grid, thr=0.0962, min_size=2):
    """grid: 2D array of probabilities. Returns (filtered_bool, n_components)."""
    hot = grid >= thr
    lab, ncomp = clabel(hot, structure=np.ones((3, 3)))
    keep = {c + 1 for c, sz in enumerate([(lab == c).sum() for c in range(1, ncomp + 1)]) if sz >= min_size}
    filt = np.zeros_like(hot)
    for c in keep:
        filt |= lab == c
    return filt, ncomp
