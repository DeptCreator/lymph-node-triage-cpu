"""Fetch versioned weights into ./models (first run needs internet; afterwards fully offline)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from inference import ensure_weights

if __name__ == "__main__":
    d = ensure_weights()
    print("models ready in", d)
