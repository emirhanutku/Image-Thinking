from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import re

NUM_RE = re.compile(r'^[+-]?(\d+(\.\d+)?|\.\d+)$')

def paddle_extract_y_ticks(pil_img, ocr, crop_box=None, keep_left_frac=0.35):

    W, H = pil_img.size
    if crop_box is None:
        x0, y0, x1, y1 = 0, 0, int(W * 0.35), H
    else:
        x0, y0, x1, y1 = crop_box

    crop = pil_img.crop((x0, y0, x1, y1))
    arr  = np.array(crop.convert("RGB"))

    results = ocr.predict(arr)
    ticks = []

    def try_add(text, poly):
        s = re.sub(r'[^0-9.\-]', '', text.strip())
        if not s or not NUM_RE.match(s):
            return
        try:
            val = float(s)
        except:
            return

        poly = np.array(poly, dtype=float)
        y_c = float(poly[:, 1].mean())  
        x_c = float(poly[:, 0].mean())  

        y_full = y0 + y_c
        x_full = x0 + x_c

        if x_full > x0 + keep_left_frac * (x1 - x0):
            return

        ticks.append((y_full, val))

    for res in results:
        jd = getattr(res, "json", None)
        if jd and "res" in jd:
            r = jd["res"]
            texts = r.get("rec_texts", [])

            polys = r.get("det_polys") or r.get("boxes") or r.get("polys") or r.get("rec_polys")
            rects = r.get("rects")

            if polys:
                for t, p in zip(texts, polys):
                    try_add(t, p)
            elif rects:
                for t, (xx, yy, ww, hh) in zip(texts, rects):
                    try_add(t, [(xx, yy), (xx+ww, yy), (xx+ww, yy+hh), (xx, yy+hh)])
        else:

            if isinstance(res, list):
                for box, (t, _score) in res:
                    try_add(t, box)


    ticks.sort(key=lambda kv: kv[0])
    merged = []
    tol = max(4, 0.006 * H)  
    for ypx, v in ticks:
        if not merged or abs(ypx - merged[-1][0]) > tol:
            merged.append([ypx, [v]])
        else:
            merged[-1][1].append(v)
    return [(y, float(np.median(vals))) for y, vals in merged]

def fit_pixel_to_value(ticks):
    ys = np.array([t[0] for t in ticks], float)
    vs = np.array([t[1] for t in ticks], float)
    if len(ys) < 2:
        raise ValueError("Need at least two y-axis ticks.")
    if len(ys) == 2:
        m = (vs[1] - vs[0]) / (ys[1] - ys[0])
        c = vs[0] - m * ys[0]
    else:
        m, c = np.polyfit(ys, vs, 1)
    return float(m), float(c)

def ypx_to_value(y_px, m, c):
    return int(round(m * y_px + c))-1
