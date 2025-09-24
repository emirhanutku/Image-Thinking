from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple
import io
import os
import re

from datasets import load_dataset
from PIL import Image
import requests


# ---------------------------
# Split selection
# ---------------------------

_PREFERRED_SPLITS: Tuple[str, ...] = ("test", "validation", "train", "dev", "val")

def _pick_split(dataset_name: str, preferred: Tuple[str, ...] = _PREFERRED_SPLITS) -> str:
    last_err: Optional[Exception] = None
    for sp in preferred:
        try:
            load_dataset(dataset_name, split=sp, streaming=True)
            return sp
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No available split found for '{dataset_name}'. Tried: {preferred}. Last error: {last_err}")


# ---------------------------
# Question / Label extraction
# ---------------------------

def _extract_question(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get("question"), str) and sample["question"].strip():
        return sample["question"].strip()
    return str(sample)

def _extract_label(sample: Dict[str, Any]) -> Optional[str]:
    for k in ("label", "answer", "gt", "gold", "target"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

        if isinstance(v, (list, tuple)) and v:
            return ", ".join(map(str, v))
    return None


# ---------------------------
# Image extraction
# ---------------------------

def _open_url_as_pil(url: str, timeout: int = 30) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def _open_path_as_pil(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    return Image.open(path).convert("RGB")

def _extract_pil_image(sample: Dict[str, Any]) -> Image.Image:
    if "image" in sample:
        im = sample["image"]
        if isinstance(im, Image.Image):
            return im.convert("RGB")
        if isinstance(im, dict):
            if im.get("bytes") is not None:
                return Image.open(io.BytesIO(im["bytes"])).convert("RGB")
            if im.get("path"):
                return _open_path_as_pil(im["path"])

    raise RuntimeError("Could not extract an image from sample (no known image fields).")


# ---------------------------
# Public API
# ---------------------------

def yield_examples(
    dataset_name: str,
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:

    chosen_split = split or _pick_split(dataset_name)
    ds = load_dataset(dataset_name, split=chosen_split)

    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        sample = ds[i]
        img = _extract_pil_image(sample)
        q = _extract_question(sample)
        y = _extract_label(sample)
        yield {
            "image": img,
            "question": q,
            "label": y,
            "index": i,
            "raw": sample,
        }

def fetch_examples(
    dataset_name: str,
    split: Optional[str] = None,
    limit: int = 1,
) -> List[Dict[str, Any]]:

    if limit <= 0:
        return []
    return list(yield_examples(dataset_name=dataset_name, split=split, limit=limit))
