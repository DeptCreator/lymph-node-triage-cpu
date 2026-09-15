"""Batch CLI: image -> heatmap + JSON report. Works fully offline after first weights fetch."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tiling import tissue_cells
from filters import filter_hot
from heatmap import render_overlay
from inference import predict_tiles
from PIL import Image
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Lymph-node triage screening (RUO, CPU-only)")
    ap.add_argument("image", help="input PNG/JPG scan crop")
    ap.add_argument("--out", default="heatmap_out.png", help="output overlay path")
    ap.add_argument("--thr", type=float, default=0.0962)
    ap.add_argument("--tile", type=int, default=224)
    args = ap.parse_args()
    t0 = time.time()
    img = Image.open(args.image).convert("RGB")
    cells, (nx, ny), _ = tissue_cells(img, tile_px=args.tile)
    tiles = [img.crop((ix * args.tile, jy * args.tile, (ix + 1) * args.tile, (jy + 1) * args.tile))
             for ix, jy in cells]
    probs = predict_tiles(tiles) if tiles else []
    grid = np.zeros((ny, nx))
    for (ix, jy), p in zip(cells, probs):
        grid[jy, ix] = p
    hot = grid >= args.thr
    filt, ncomp = filter_hot(grid, args.thr, 2)
    render_overlay(img, grid).save(args.out)
    rep = {"image": args.image, "size": list(img.size), "grid": [nx, ny],
           "tiles_total": nx * ny, "tiles_kept": len(cells),
           "hot": int(hot.sum()), "components": int(ncomp),
           "max_p": round(float(grid.max()), 3) if grid.size else 0.0,
           "thr": args.thr, "wall_s": round(time.time() - t0, 1),
           "note": "RUO triage aid. Verdict by pathologist. Scale unverified for arbitrary uploads."}
    print(json.dumps(rep, indent=1))
    Path(args.out).with_suffix(".json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
