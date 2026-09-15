"""Gradio demo: upload scan -> threshold slider -> heatmap + honest stats.
Backbone: validated PCam probe. RUO banner is hard-coded visible.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tiling import tissue_cells
from filters import filter_hot
from heatmap import render_overlay
from inference import predict_tiles
import numpy as np

THR_DEFAULT = 0.0962
TILE = 224

RUO_BANNER = (
    "### ⚠️ RESEARCH USE ONLY — не для клинических решений\n"
    "Прототип triage-ассистента: подсвечивает подозрительные зоны, вердикт ставит врач. "
    "AUC 0.937, чувствительность 0.95 / специфичность 0.67 (рабочая точка 0.0962). "
    "Детали валидации и ограничения — в разделе ниже."
)

VALIDATION_DETAILS = (
    "Валидация (frozen, PCam test 1k): AUC 0.937 [95% CI 0.922–0.951]; "
    "sens 0.947 [0.925–0.968] / spec 0.671 при thr 0.0962.\n\n"
    "ITC/micro: регионы подсвечены 35/35, но пик тепловой карты лежит точно на очаге "
    "лишь в 11/35 — это инструмент триажа зон, а не указатель на клетки.\n\n"
    "Чистые слайды: количественная FP-оценка — открытая задача валидации; "
    "подозрительные зоны в любом случае требуют сверки с тканью.\n\n"
    "Масштаб входа для произвольных загрузок не верифицирован "
    "(валидировано на патчах 96px@10x-класс)."
)


def analyze(pil_img, thr=THR_DEFAULT):
    t0 = time.time()
    W, H = pil_img.size
    cells, (nx, ny), _ = tissue_cells(pil_img, tile_px=TILE)
    if nx < 1 or ny < 1:
        return pil_img, pil_img, {"error": "image smaller than one 224 tile"}, 0.0
    tiles = [pil_img.crop((ix * TILE, jy * TILE, (ix + 1) * TILE, (jy + 1) * TILE)) for ix, jy in cells]
    probs = predict_tiles(tiles) if tiles else []
    grid = np.zeros((ny, nx))
    for (ix, jy), p in zip(cells, probs):
        grid[jy, ix] = p
    hot = grid >= thr
    filt, ncomp = filter_hot(grid, thr, 2)
    combo = render_overlay(pil_img, grid)
    stats = {"tiles_total": nx * ny, "tiles_kept": len(cells), "hot": int(hot.sum()),
             "components": int(ncomp), "max_p": round(float(grid.max()), 3) if grid.size else 0.0,
             "thr": thr, "infer_note": "input scale UNVERIFIED for arbitrary uploads (demo only)"}
    return pil_img, combo, stats, time.time() - t0


def build_ui():
    import gradio as gr
    with gr.Blocks(title="MedSafety triage demo (RUO)") as app:
        gr.Markdown(RUO_BANNER)
        with gr.Row():
            inp = gr.Image(type="pil", label="Скан / вырезка (PNG/JPG)")
            thr = gr.Slider(0.01, 0.9, value=THR_DEFAULT, step=0.005, label="Порог P(tumor)")
        btn = gr.Button("Анализировать")
        with gr.Row():
            out_orig = gr.Image(label="Оригинал")
            out_heat = gr.Image(label="Heatmap (красный = подозрение)")
        out_json = gr.JSON(label="Статистика прогона")
        with gr.Accordion("Детали валидации и ограничения", open=False):
            gr.Markdown(VALIDATION_DETAILS)

        def run(img, t):
            if img is None:
                return None, None, {"error": "no image"}
            o, h, st, _ = analyze(img, float(t))
            return o, h, st

        btn.click(run, [inp, thr], [out_orig, out_heat, out_json])
    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
