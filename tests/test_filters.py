"""Unit test: singleton suppression against the REAL src.filters (no duplicated logic)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from filters import filter_hot

grid = np.full((6, 6), 0.01)
grid[0, 0] = 0.95
grid[3:5, 3:5] = 0.80
grid[0, 5] = 0.50
filt, ncomp = filter_hot(grid)
assert filt[0, 0] == False, "singleton survived!"
assert filt[0, 5] == False, "second singleton survived!"
assert filt[3:5, 3:5].all(), "cluster damaged!"
assert filt.sum() == 4, f"unexpected kept count {filt.sum()}"
print(f"singleton-test PASS: components={ncomp}, kept={int(filt.sum())}/36")
