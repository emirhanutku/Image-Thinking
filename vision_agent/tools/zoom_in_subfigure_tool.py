from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from PIL import Image

from rapidfuzz import fuzz

from openai import OpenAI
import base64, io, json, re
from typing import Dict, Any, Tuple, List
from PIL import Image, ImageDraw
from rapidfuzz import fuzz
from paddleocr import PaddleOCR

from vision_agent.utils.image_io import  pil_to_data_url


_openai_client = OpenAI()  # picks up OPENAI_API_KEY from env





@dataclass
class TextBox:
    text: str
    bbox: Tuple[int, int, int, int]  
    score: float

@dataclass
class PanelCandidate:
    panel_id: int
    bbox: Tuple[int, int, int, int]  
    title: str
    context: str         
    ocr_tokens: List[TextBox]
    pre_score: float     



def _clip_bbox(x1, y1, x2, y2, w, h):
    return (max(0, min(x1, w-1)),
            max(0, min(y1, h-1)),
            max(0, min(x2, w-1)),
            max(0, min(y2, h-1)))

def _expand_bbox(b, w, h, frac=0.06):
    x1, y1, x2, y2 = b
    dx, dy = int((x2 - x1) * frac), int((y2 - y1) * frac)
    return _clip_bbox(x1 - dx, y1 - dy, x2 + dx, y2 + dy, w, h)

def _to_rgb_np(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"))

def _poly_to_bbox(poly: np.ndarray) -> Tuple[int, int, int, int]:
    xs, ys = poly[:, 0], poly[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def run_paddle_ocr(pil: Image.Image, _paddle_ocr: PaddleOCR) -> List[TextBox]:
    arr = _to_rgb_np(pil)
    results = _paddle_ocr.predict(arr)
    out: List[TextBox] = []

    for r in results:
        jd = getattr(r, "json", None)
        if jd is None:
            continue
        res = jd.get("res", {})
        texts = res.get("rec_texts", []) or []
        scores = res.get("rec_scores", []) or []
        boxes = res.get("rec_boxes", []) or []


        if (not boxes) and res.get("rec_polys") is not None:
            polys = res["rec_polys"]
            for i, t in enumerate(texts):
                try:
                    poly = np.array(polys[i])
                    x1, y1, x2, y2 = _poly_to_bbox(poly)
                    s = float(scores[i]) if i < len(scores) else 1.0
                    out.append(TextBox(text=t, bbox=(x1, y1, x2, y2), score=s))

                except Exception:
                    pass
        else:
            for i, t in enumerate(texts):
                try:

                    bx = boxes[i]
                    x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                    s = float(scores[i]) if i < len(scores) else 1.0
                    #print(f"Anchor candidate: '{t}' score={s}")
                    out.append(TextBox(text=t, bbox=(x1, y1, x2, y2), score=s))
                except Exception:
                    pass

    return out



def _draw_anchor_overlay(pil: Image.Image, anchor_bbox: Tuple[int,int,int,int]) -> Image.Image:
    over = pil.copy()
    d = ImageDraw.Draw(over)
    x1,y1,x2,y2 = anchor_bbox

    d.rectangle([x1,y1,x2,y2], outline=(255,255,0), width=5)
    d.rectangle([x1,y1,x2,y2], fill=(255,0,0,40))
    return over

def _pick_best_anchor(ocr_boxes, query: str) -> Dict[str, Any]:

    best = None
    for tb in ocr_boxes:
        t = tb.text.strip()
        if not t: 
            continue
        s = max(
            fuzz.token_set_ratio(t, query),
            fuzz.token_sort_ratio(t, query),
            fuzz.partial_ratio(t, query),
        )
        cand = {"text": t, "bbox": tb.bbox, "score": float(s)}
        if (best is None) or (s > best["score"]):
            best = cand
    #print(f"Best anchor: {best}")
    return best

def _llm_bbox_from_image_and_anchor(
    pil: Image.Image,
    anchor_bbox: Tuple[int,int,int,int],
    query: str,
    model: str = "gpt-5-2025-08-07"
) -> Tuple[int,int,int,int]:

    if _openai_client is None:
        raise RuntimeError("OpenAI client not available.")

    W, H = pil.size
    overlay = _draw_anchor_overlay(pil, anchor_bbox)
    overlay.show(title="overlay")

    # Two images: original + overlay (overlay makes it trivial for the model)
    img_url = pil_to_data_url(pil)
    ovl_url = pil_to_data_url(overlay)

    system = (
        "You see a scientific figure. We highlighted (with a red rectangle) the anchor text "
        "that best matches the user's request. Return the bounding box of the ENTIRE subfigure/panel "
        "that the anchor belongs to. Output strictly JSON with integer pixels [x1,y1,x2,y2], "
        "top-left origin, inclusive-exclusive semantics are not required (just standard pixel bounds)."
    )

    user_text = (
        f"User request: {query}\n"
        f"Image size: {W}x{H}\n"
        f"Anchor bbox (x1,y1,x2,y2): {list(map(int, anchor_bbox))}\n\n"
        "Please return JSON ONLY in the form:\n"
        '{ "bbox": [x1,y1,x2,y2], "reason": "...", "confidence": 0.0..1.0 }\n'
    )

    # Prefer Responses API; if your stack uses chat.completions with image parts, adapt accordingly.
    resp = _openai_client.chat.completions.create(
        model=model,
        temperature=1,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "image_url", "image_url": {"url": ovl_url}},
                ],
            },
        ],
    )

    txt = resp.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise RuntimeError("LLM did not return JSON.")
    data = json.loads(m.group(0))
    bx = data.get("bbox", None)
    if not (isinstance(bx, list) and len(bx) == 4):
        raise RuntimeError("Invalid bbox from LLM.")

    x1,y1,x2,y2 = map(int, bx)
    return _clip_bbox(x1,y1,x2,y2, W, H)

def zoom_in_subfigure_llm_only(
    pil: Image.Image,
    description: str,
    llm_model: str = "gpt-5",
    margin_frac: float = 0.06,
    _paddle_ocr: PaddleOCR = None,
) -> Dict[str, Any]:

    W, H = pil.size
    ocr_boxes = run_paddle_ocr(pil,_paddle_ocr)
    anchor = _pick_best_anchor(ocr_boxes, description)

    if not anchor:
        # fallback: whole image (rare if OCR sees nothing)
        anchor_bbox = (0,0,W,H)
    else:
        anchor_bbox = anchor["bbox"]

    # Ask GPT-5 for the panel box
    x1,y1,x2,y2 = _llm_bbox_from_image_and_anchor(pil, anchor_bbox, description, model=llm_model)

    # Small margin so we don't clip axes/titles
    x1,y1,x2,y2 = _expand_bbox((x1,y1,x2,y2), W, H, frac=margin_frac)
    crop = pil.crop((x1,y1,x2,y2))
    crop.show(title="last crop")

    return {
        "crop": crop,
        "bbox": (x1,y1,x2,y2),
        "meta": {
            "mode": "llm_only",
            "anchor_text": anchor.get("text") if anchor else "",
            "anchor_bbox": anchor_bbox,
        }
    }

class ZoomInSubfigureTool:
    def __init__(self,image : Image.Image, llm_model :str ,margin_frac : float,_paddle_ocr : PaddleOCR):
        self._image = image
        self._llm_model = llm_model
        self._margin_frac = margin_frac
        self._paddle_ocr = _paddle_ocr

    def __call__(self, description: str) -> Dict[str, Any]:
        return zoom_in_subfigure_llm_only(self._image, description, llm_model=self._llm_model, margin_frac=self._margin_frac, _paddle_ocr=self._paddle_ocr)