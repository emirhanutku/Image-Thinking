from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import cv2
import torch

from transformers import Sam2Processor, Sam2Model

@dataclass
class SeedClicks:
    positive: List[Tuple[int, int]]
    negative: List[Tuple[int, int]]


_SAM2_MODEL = None
_SAM2_PROC = None
_SAM2_DEVICE = "cpu"

def _choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _ensure_sam2(model_id: str = "facebook/sam2.1-hiera-large"):
    global _SAM2_MODEL, _SAM2_PROC, _SAM2_DEVICE
    if _SAM2_MODEL is None or _SAM2_PROC is None:
        _SAM2_DEVICE = _choose_device()
        _SAM2_MODEL = Sam2Model.from_pretrained(model_id).to(_SAM2_DEVICE).eval()
        _SAM2_PROC = Sam2Processor.from_pretrained(model_id)
    return _SAM2_MODEL, _SAM2_PROC, _SAM2_DEVICE

# --------- Helpers ---------
def _pil_to_rgb(pil: Image.Image) -> Image.Image:
    return pil.convert("RGB")

def _mask_to_bbox_and_polygon(mask: np.ndarray) -> Tuple[Tuple[int,int,int,int], List[List[int]]]:
    ys, xs = np.where(mask > 0)  # ys: row indices, xs: column indices
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0), []
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    bbox = (x1, y1, x2, y2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bbox, []
    cnt = max(cnts, key=cv2.contourArea)
    poly = cnt[:, 0, :].tolist()
    return bbox, poly

def _make_overlay(pil: Image.Image, mask: np.ndarray, alpha: int = 130) -> Image.Image:
    rgb = _pil_to_rgb(pil)
    r, g, b = rgb.split()
    A = Image.fromarray((mask * alpha).astype(np.uint8))
    return Image.merge("RGBA", (r, g, b, A))

from PIL import Image
import numpy as np

def paste_icon_on_segment(
    base_img: Image.Image,
    mask_img: Image.Image,
    icon_img: Image.Image,
    scale: float = 0.25,
    at: str = "centroid"
) -> Image.Image:

    mask = (np.array(mask_img) > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return base_img  

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max() + 1, ys.max() + 1
    bw, bh = (x2 - x1), (y2 - y1)

    if at == "centroid":
        cx, cy = int(xs.mean()), int(ys.mean())
    elif at == "bbox_top":
        cx, cy = x1 + bw // 2, y1  
    else:
        cx, cy = int(xs.mean()), int(ys.mean())

    icon = icon_img.convert("RGBA").copy()
    target = int(min(bw, bh) * scale)
    target = max(target, 8)  
    icon.thumbnail((target, target), Image.LANCZOS)
    iw, ih = icon.size

    x = int(cx - iw // 2)
    y = int(cy - ih // 2)

    W, H = base_img.size
    x = max(min(x, W - iw), 0)
    y = max(min(y, H - ih), 0)

    # 4) yapıştır (alpha korunur)
    out = base_img.convert("RGBA").copy()
    out.alpha_composite(icon, dest=(x, y))

    
    return out



def _make_inputs(processor: Sam2Processor, pil: Image.Image, seeds: SeedClicks) -> Dict[str, torch.Tensor]:
    pos = list(seeds.positive or [])
    neg = list(seeds.negative or [])

    if not pos and not neg:
        raise ValueError("At least one positive point is required.")

    pts = pos + neg
    points_list = [[float(x), float(y)] for (x, y) in pts]
    labels_list = [1] * len(pos) + [0] * len(neg)


    return processor(
        images=pil.convert("RGB"),
        input_points=[[points_list]],
        input_labels=[[labels_list]],
        return_tensors="pt",
    )


def _pick_best_mask(masks_img_space: torch.Tensor) -> np.ndarray:
    m = masks_img_space
    if m.ndim == 4:
        m = m[0]
    best = None
    best_area = -1
    for i in range(m.shape[0]):
        binm = (m[i] > 0.5).cpu().numpy().astype(np.uint8)
        area = int(binm.sum())
        if area > best_area:
            best, best_area = binm, area
    return best if best is not None else np.zeros_like(m[0].cpu().numpy(), dtype=np.uint8)

def segment_region_around_point(
    pil: Image.Image,
    seeds: SeedClicks,
    model_id: str = "facebook/sam2.1-hiera-large",
) -> Dict[str, Any]:
    model, processor, device = _ensure_sam2(model_id)
    inputs = _make_inputs(processor, pil, seeds).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]
    bin_mask = _pick_best_mask(masks)
    bbox, polygon = _mask_to_bbox_and_polygon(bin_mask)
    area = int(bin_mask.sum())
    return {
        "mask": Image.fromarray((bin_mask * 255).astype(np.uint8), mode="L"),
        "overlay": _make_overlay(pil, bin_mask),
        "bbox": bbox,
        "polygon": polygon,
        "area": area,
        "scores": {"num_masks": int(masks.shape[0])},
    }


class SegmentTool :
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large"):
        self.model_id = model_id

    def __call__(self, pil: Image.Image, param: Any) -> Dict[str, Any]:
        if isinstance(param, (list, tuple)) and len(param) == 2:
            x, y = param
            seeds = SeedClicks(positive=[(int(round(float(x))), int(round(float(y))))], negative=[])
        elif isinstance(param, dict) and "x" in param and "y" in param:
            x, y = param["x"], param["y"]
            seeds = SeedClicks(positive=[(int(round(float(x))), int(round(float(y))))], negative=[])
        else:
            raise ValueError("param must be [x, y] or {'x': ..., 'y': ...}")

        out = segment_region_around_point(pil, seeds, model_id=self.model_id)

        star = Image.open("/Users/emirhan/Desktop/Jotform Projects/Visual Reasoning/vision_agent/images/greenStar.png").convert("RGBA")

        img_with_star = paste_icon_on_segment(
            base_img=pil,
            mask_img=out["mask"],
            icon_img=star,
            scale=0.09,
            at="centroid"
        )
        #img_with_star.show()

        return img_with_star