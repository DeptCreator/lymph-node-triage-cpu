"""Heatmap overlay rendering."""
import numpy as np
from PIL import Image


def render_overlay(pil_img, grid, alpha=0.55):
    """grid: (ny, nx) probabilities in [0, 1]. Returns PIL RGB overlay."""
    ny, nx = grid.shape
    ox = Image.new("L", (nx, ny))
    ox.putdata([int(255 * min(1.0, max(0.0, v))) for v in grid.ravel()])
    ox = ox.resize(pil_img.size, Image.NEAREST)
    red = Image.new("RGBA", pil_img.size, (255, 0, 0, 0))
    red.putalpha(ox.point(lambda v: int(v * alpha)))
    return Image.alpha_composite(pil_img.convert("RGBA"), red).convert("RGB")
