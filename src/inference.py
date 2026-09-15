"""Model bootstrap: downloads versioned weights on first run, then runs offline.
Backbone: CTransPath (community mirror, GPLv3 — see WEIGHTS.md, NOT Apache-2.0).
Probe: logistic regression trained by us (Apache-2.0, same as this code).
"""
import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models"
RELEASE_BASE = "https://github.com/DeptCreator/lymph-node-triage-cpu/releases/download/v0.1.0"
FILES = ["ctranspath_ov.xml", "ctranspath_ov.bin", "probe_m2.pkl"]
_compiled = None
_clf = None


def ensure_weights():
    import urllib.request
    MODEL_DIR.mkdir(exist_ok=True)
    for fn in FILES:
        dst = MODEL_DIR / fn
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"downloading {fn} ...", flush=True)
        urllib.request.urlretrieve(f"{RELEASE_BASE}/{fn}", dst)
    return MODEL_DIR


def get_stack():
    """Returns (compiled OpenVINO model, sklearn probe). Loads once."""
    global _compiled, _clf
    if _compiled is None:
        import openvino as ov
        ensure_weights()
        core = ov.Core()
        xml = MODEL_DIR / "ctranspath_ov.xml"
        _compiled = core.compile_model(core.read_model(str(xml)), "CPU")
        with open(MODEL_DIR / "probe_m2.pkl", "rb") as f:
            _clf = pickle.load(f)
    return _compiled, _clf


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(pil_tiles, size=224):
    import numpy as np
    arr = [np.asarray(t.resize((size, size)), dtype=np.float32) for t in pil_tiles]
    X = np.stack(arr).astype(np.float32) / 255.0
    m = np.array(IMAGENET_MEAN, np.float32).reshape(1, 1, 1, 3)
    s = np.array(IMAGENET_STD, np.float32).reshape(1, 1, 1, 3)
    return ((X - m) / s).transpose(0, 3, 1, 2).astype(np.float32)


def predict_tiles(pil_tiles, batch=8):
    """pil_tiles: list of PIL RGB. Returns list[float] P(tumor)."""
    compiled, clf = get_stack()
    import numpy as np
    out = []
    for i in range(0, max(len(pil_tiles), 1), batch):
        fe = compiled(preprocess(pil_tiles[i:i + batch]))[0]
        out.extend(float(p) for p in clf.predict_proba(fe)[:, 1])
    return out
