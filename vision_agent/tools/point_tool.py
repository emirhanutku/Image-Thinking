from typing import Dict, Tuple, Optional, Any, List
from functools import lru_cache
import re

import torch
from PIL import Image , ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import numpy as np
from vision_agent.tools.point_tool_qwen import qwen_point_2b 

from vision_agent.utils.get_value_from_pixel import (paddle_extract_y_ticks, fit_pixel_to_value, ypx_to_value)
from paddleocr import PaddleOCR

ocr = PaddleOCR(device="cpu",
    use_textline_orientation=True,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,)

_num_pat = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)%?(?!\w)")
_year_pat = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")  

# -------- HELPERS ---------
def _rgb_dist2(target: tuple, pixels: np.ndarray) -> np.ndarray:
    tr, tg, tb = int(target[0]), int(target[1]), int(target[2])

    diff_r = pixels[..., 0].astype(np.int16) - tr
    diff_g = pixels[..., 1].astype(np.int16) - tg
    diff_b = pixels[..., 2].astype(np.int16) - tb

    dist2 = (
        np.square(diff_r, dtype=np.int32) +
        np.square(diff_g, dtype=np.int32) +
        np.square(diff_b, dtype=np.int32)
    )
    return dist2  


def _find_y_match_along_x(
    img: Image.Image,
    x: int,
    target_rgb: tuple,
    y_min: int = 0,
    y_max: Optional[int] = None,
    x_window: int = 2,
    dist_threshold: Optional[float] = None,
) -> Tuple[Optional[int], Optional[float]]:

    W, H = img.size
    if W <= 0 or H <= 0:
        return None, None

    x = max(0, min(int(round(x)), W - 1))
    x0 = max(0, x - int(abs(x_window)))
    x1 = min(W - 1, x + int(abs(x_window)))

    if y_max is None:
        y_max = H
    y0 = max(0, int(y_min))
    y1 = min(H, int(y_max))
    if y0 >= y1:
        return None, None

    arr = np.array(img.convert("RGB"))  

    col_block = arr[y0:y1, x0:x1 + 1, :]


    dist2 = _rgb_dist2(target_rgb, col_block)  # (Y, Xw)

    min_dist2_across_x = dist2.min(axis=1)  # (Y,)

    best_idx = int(min_dist2_across_x.argmin())
    best_y = y0 + best_idx
    min_dist = float(np.sqrt(float(min_dist2_across_x[best_idx])))

    if dist_threshold is not None and min_dist > dist_threshold:
        return None, None

    return best_y, min_dist

def mark_pixel(img: Image.Image, x: int, y: int,
               radius: int = 5,
               crosshair: bool = True,
               color: tuple = (255, 0, 0),
               width: int = 2) -> Image.Image:

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    w, h = annotated.size
    x_clamped = max(0, min(x, w - 1))
    y_clamped = max(0, min(y, h - 1))

    bbox = [
        x_clamped - radius, y_clamped - radius,
        x_clamped + radius, y_clamped + radius
    ]
    draw.ellipse(bbox, outline=color, width=width)

    if crosshair:
        draw.line([(x_clamped - radius*2, y_clamped),
                   (x_clamped + radius*2, y_clamped)], fill=color, width=width)
        draw.line([(x_clamped, y_clamped - radius*2),
                   (x_clamped, y_clamped + radius*2)], fill=color, width=width)

    draw.ellipse([x_clamped-1, y_clamped-1, x_clamped+1, y_clamped+1],
                 fill=color)

    return annotated


def _is_near_white(pixel, white_thresh: int = 230) -> bool:
    if isinstance(pixel, int):
        return pixel >= white_thresh

    if isinstance(pixel, (tuple, list)):
        r, g, b = pixel[:3]
        return (r >= white_thresh) and (g >= white_thresh) and (b >= white_thresh)

def _coerce_number(s):
    s = str(s).strip().replace(",", "")
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None
        
def _fmt_like_axis(v):
    try:
        vf = float(v)
        if vf.is_integer():
            return str(int(vf))
        return str(vf)
    except Exception:
        return str(v)
#------------ HELPERS END -----------
def _qwen_initial_point(image: Image.Image, phrase: str, W: int, H: int):


    qwen_query = (
        f'Locate: {phrase}.'
    )

    qout = qwen_point_2b(image, qwen_query)  

    px = qout["point_px"]

    cx_det = int(round(float(px[0])))
    cy_det = int(round(float(px[1])))

    # clamp to image bounds
    cx_det = max(0, min(cx_det, W - 1))
    cy_det = max(0, min(cy_det, H - 1))

    detections = [{
        "center_xy": (float(cx_det), float(cy_det)),
        "bbox_xyxy": (float(cx_det), float(cy_det), float(cx_det), float(cy_det)),
        "score": 1.0,          
        "label": phrase,       
    }]

    return cx_det, cy_det, detections

def _refine_point_upwards_if_white(
    image: Image.Image,
    x: int,
    y: int,
    window_half_width: int = 3,
    white_thresh: int = 230,
    max_scan_pixels: int = None,
) -> int:

    W, H = image.size
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))

    px = image.load()
    if not _is_near_white(px[x, y], white_thresh=white_thresh):
        return (x,y)  

    if max_scan_pixels is None:
        max_scan_pixels = y  

    steps = 0
    for yy in range(y - 5, -1, -1):
        if steps >= max_scan_pixels:
            break

        x0 = max(0, x - window_half_width)
        x1 = min(W - 1, x + window_half_width)

        non_white_found = False
        for xx in range(x0, x1 + 1):
            if not _is_near_white(px[xx, yy], white_thresh=white_thresh):
                non_white_found = True
                break

        if non_white_found:
            return (xx,yy)

        steps += 1

    return (x,y)


def infer_device():
    if torch.backends.mps.is_available():   # Apple Silicon GPU (Metal)
        return torch.device("mps")
    if torch.cuda.is_available():           # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")   



def _ocr_words(img_rgb: Image.Image) -> List[Dict[str, Any]]:
    arr = np.array(img_rgb.convert("RGB"))
    outputs = ocr.predict(arr)

    words: List[Dict[str, Any]] = []

    def add_item(text, conf, poly):
        if not text:
            return

        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        words.append({
            "text": str(text).strip(),
            "conf": float(conf) if conf is not None else 0.0,
            "bbox": (x0, y0, x1, y1),
            "center": (cx, cy),
        })


    for res in outputs:
        jd = getattr(res, "json", None)
        if jd and "res" in jd:
            r = jd["res"]
            texts = r.get("rec_texts", []) or []
            rec_scores = r.get("rec_scores", []) or []
            polys = r.get("det_polys") or r.get("boxes") or r.get("polys") or r.get("rec_polys")
            rects = r.get("rects")


            if polys:
                for i, t in enumerate(texts):
                    poly = polys[i]
                    score = rec_scores[i] if i < len(rec_scores) else None
                    add_item(t, score, poly)
            elif rects:
                for i, t in enumerate(texts):
                    x, y, w, h = rects[i]
                    poly = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                    score = rec_scores[i] if i < len(rec_scores) else None
                    add_item(t, score, poly)

        else:

            if isinstance(res, list):
                for poly, (t, score) in res:
                    add_item(t, score, poly)

    #print("---------------------------")
    return words


def _find_x_from_label_or_interpolate(
    words: List[Dict[str, Any]],
    wanted_label: str,
    w: int, h: int,
    *,
    step=1,
    max_neighbor_gap=5,
):

    x = _find_x_from_label(words, str(wanted_label), w=w, h=h)
    if x is not None:
        return int(x)


    v = _coerce_number(wanted_label)
    if v is None:
        return None

    left = right = None  
    for k in range(1, max_neighbor_gap + 1):

        lv = v - k * step
        lx = _find_x_from_label(words, _fmt_like_axis(lv), w,h)
        if lx is not None and left is None:
            left = (float(lv), int(lx))


        rv = v + k * step
        rx = _find_x_from_label(words, _fmt_like_axis(rv), w,h)
        if rx is not None and right is None:
            right = (float(rv), int(rx))

        if left and right:
            break

    if left and right:
        (lv, lx), (rv, rx) = left, right
        if rv == lv:
            return None
        ratio = (float(v) - lv) / (rv - lv)
        xi = int(round(lx + ratio * (rx - lx)))
        return xi

    return None

def _find_x_from_label(words: List[Dict[str, Any]], label: str, w: int, h: int) -> Optional[int]:
    candidates = []
    for wobj in words:
        if wobj["text"] == label:

            cx, cy = wobj["center"]

            candidates.append((cy, int(round(cx))))
            break
        elif label in wobj["text"]:
            #print("partial word found", wobj["text"])
            text = wobj["text"]
            start = text.find(label)
            end = start + len(label)

            x1, y1, x2, y2 = wobj["bbox"]
            cy = (y1 + y2) / 2.0


            text_len = max(len(text), 1)
            char_w = (x2 - x1) / float(text_len)

            sub_cx = x1 + ((start + end) / 2.0) * char_w
            bbox_cx = (x1 + x2) / 2.0

            margin = max(3, int(0.005 * w))

            if sub_cx >= bbox_cx:
                pick_x = min(w - 1, x2 - margin)
            else:
                pick_x = max(0, x1 + margin)

            candidates.append((cy, int(pick_x)))

    if not candidates:

        return None
    candidates.sort(key=lambda t: (abs(t[0] - (h * 0.9)), ))  # near bottom
    return candidates[0][1]


def _chart_point(image: Image.Image, param: str) -> Optional[int]:
    W, H = image.size
    words = _ocr_words(image)

    m_year = _year_pat.search(param)
    
    x_label = m_year.group(0) if m_year else None

    if not x_label:

        nums = _num_pat.findall(param)
        if nums:
            x_label = nums[0]
    if not x_label:
        return None
    x_px = _find_x_from_label_or_interpolate(words, x_label, W, H)
    if x_px is None:
        return None

    return x_px


def locate_phrase_with_groundingdino(
    image: Image.Image,
    phrase: str,
    *,
    prefer: str = 'auto',
) -> Optional[Dict[str, Any]]:
    
    W, H = image.size

    #print("phrase : ", phrase)
    cx_det, cy_det, detections = _qwen_initial_point(image, phrase, W, H)

    _refined_x, _refined_y = _refine_point_upwards_if_white(
        image=image,
        x=cx_det,
        y=cy_det,
        window_half_width=25,
        white_thresh=230,
    max_scan_pixels=50,  
    )

    if _refined_y != cy_det:
        cy_det = _refined_y

        if detections:
            detections[0]["center_xy"] = (float(cx_det), float(cy_det))
            detections[0]["bbox_xyxy"] = (float(cx_det), float(cy_det), float(cx_det), float(cy_det))

    out_pixel = mark_pixel(image, cx_det, cy_det,1,True,(255,0,0))
    #out_pixel.show()


    if prefer == 'only_phrase':
        return {'x': cx_det, 'y': cy_det, 'value': None}

    else:
        pixel_value = image.getpixel((cx_det, cy_det))
        if _is_near_white(pixel_value, white_thresh=230):
            pixel_value = image.getpixel((_refined_x,cy_det))
        
        label_location_x = _chart_point(image, phrase)

        best = max(detections, key=lambda d: d["score"])  
        best["sampled_pixel_value"] = tuple(int(v) for v in pixel_value) if isinstance(pixel_value, tuple) else pixel_value
        best["label_location_x"] = int(label_location_x) if label_location_x is not None else None

        if label_location_x is not None:

            y_min = int(H * 0.25)
            y_max = int(H * 0.75)
            match_y, _ = _find_y_match_along_x(
                image,
                x=int(label_location_x),
                target_rgb=pixel_value,
                y_min=y_min,
                y_max=y_max,
                x_window=2,
            )

        if label_location_x is not None and match_y is not None:
            value = None
            if 'value' in phrase.lower():
                crop_box = (0, int(H*0.15), W, int(H*0.60))  
                ticks = paddle_extract_y_ticks(image, ocr, crop_box=crop_box, keep_left_frac=0.35)
                m, c = fit_pixel_to_value(ticks)
                value = ypx_to_value(match_y, m, c)

            out_pixel = mark_pixel(image, int(label_location_x), int(match_y),1,True,(0,0,255))
            #out_pixel.show()
            return {'x': int(label_location_x), 'y': int(match_y), 'value': float(value) if value is not None else None}
        else:
            out_pixel = mark_pixel(image, cx_det, cy_det,1,True,(0,0,255))
            #out_pixel.show()
            return {'x': cx_det, 'y': cy_det , 'value': None}

     
class PointTool:
    def __init__(self, prefer:str = 'auto'):
        self.prefer = prefer
    
    def __call__(self, image: Image.Image, param: str) -> Dict[str, Any]:
        res = None
        is_chart_like = bool(_year_pat.search(param) or re.search(r"\bx\s*=\s*\d", param.lower()) or "value at" in param.lower() or _num_pat.search(param))
        if self.prefer in ("auto", "chart-first"):
            if is_chart_like:
                res = locate_phrase_with_groundingdino(image, param, prefer=self.prefer)
        if res is None:
            res = locate_phrase_with_groundingdino(image, param , prefer='only_phrase')
        if 'value' in param.lower():
            if res['value'] is not None:
                result_message = f"OBSERVATION:\nPoint model outputs: <point x=\"{res['x']}\" y=\"{res['y']}\" value=\"{res['value']}\" alt=\"{param}\">{param}</point>\nPlease summarize the model outputs and answer my first question."
            else : 
                result_message = f"OBSERVATION:\nPoint model outputs: <point x=\"{res['x']}\" y=\"{res['y']}\" alt=\"{param}\">{param}</point>\nPlease summarize the model outputs and answer my first question."
            return result_message

            
        result_message = f"OBSERVATION:\nPoint model outputs: <point x=\"{res['x']}\" y=\"{res['y']}\" alt=\"{param}\">{param}</point>\nPlease summarize the model outputs and answer my first question."
        return result_message




        


