"""Tissue masking (Otsu on HSV saturation) + tile grid. Same math as validated cutter.py."""
import numpy as np
from scipy.ndimage import binary_closing


def otsu_mask(thumb):
    """thumb: PIL RGB. Returns (bool tissue mask at thumbnail res, threshold)."""
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32)
    s = hsv[:, :, 1].ravel() / 255.0
    hist, _ = np.histogram(s, bins=256, range=(0, 1))
    hist = hist.astype(np.float64)
    total = hist.sum()
    best, best_t = -1.0, 0.5
    for t in range(256):
        w0, w1 = hist[:t + 1].sum(), hist[t + 1:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:t + 1] * np.arange(t + 1)).sum() / w0 / 255.0
        m1 = (hist[t + 1:] * np.arange(t + 1, 256)).sum() / w1 / 255.0
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, t / 255.0
    mask = (hsv[:, :, 1] / 255.0 > best_t)
    return binary_closing(mask, structure=np.ones((5, 5))), round(float(best_t), 4)


def tissue_cells(pil_img, tile_px=224, frac=0.30, thumb_w=256):
    """Return [(ix, jy), ...] cells with tissue fraction >= frac. Pure function of the image."""
    W, H = pil_img.size
    tw, th = thumb_w, max(1, round(thumb_w * H / W))
    thumb = pil_img.resize((tw, th))
    mask, thr = otsu_mask(thumb)
    nx, ny = W // tile_px, H // tile_px
    cells = []
    for jy in range(ny):
        for ix in range(nx):
            mx0, my0 = int(ix * tile_px * tw / W), int(jy * tile_px * th / H)
            mx1, my1 = int((ix + 1) * tile_px * tw / W), int((jy + 1) * tile_px * th / H)
            if my1 > my0 and mx1 > mx0 and mask[max(my0, 0):my1, max(mx0, 0):mx1].mean() >= frac:
                cells.append((ix, jy))
    return cells, (nx, ny), thr
