from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict
from io import BytesIO

from datasets import load_dataset, Dataset  # type: ignore

from PIL import Image  # type: ignore


__all__ = ["PointSFTDatasetBuilder"]


POINT_TAG = re.compile(
    r'<point\s+[^>]*x="([\d\.]+)"\s+y="([\d\.]+)"[^>]*alt="([^"]+)"[^>]*>',
    re.IGNORECASE
)

def _norm_text(s: str) -> str:
    return " ".join(s.strip().lower().split())

@dataclass
class _PointHit:
    param: str
    x_px: float
    y_px: float
    image_id: Optional[str]
    msg_idx: int
    source_id: Optional[str] = None


class PointSFTDatasetBuilder:
    def __init__(
        self,
        *,
        normalize: bool = True,
        images_root: Optional[str] = None,
        strict_match: bool = False,
        consistency_radius: Optional[float] = None,
        bbox_radius: float = 0.01,
        default_wh: Optional[Tuple[int, int]] = None,
        dump_images_dir: Optional[str] = None,  
        prefer_saved_paths: bool = True,               
        sequential_naming: bool = True,                
        sequential_prefix: str = "sample",
        coord_source: str = "percent"           
    ):

        self.normalize = normalize
        self.images_root = images_root
        self.strict_match = strict_match
        self.consistency_radius = consistency_radius
        self.bbox_radius = bbox_radius
        self.default_wh = default_wh
        self.dump_images_dir = dump_images_dir
        self.prefer_saved_paths = prefer_saved_paths
        self.sequential_naming = sequential_naming
        self.sequential_prefix = sequential_prefix
        self.coord_source = coord_source

        self._samples: List[Dict[str, Any]] = []
        if self.dump_images_dir:
            os.makedirs(self.dump_images_dir, exist_ok=True)

        self._seq_counter = 0
        if self.dump_images_dir and self.sequential_naming:
            try:
                pat = re.compile(rf"^{re.escape(self.sequential_prefix)}_(\d+)\.png$")
                nums = []
                for fn in os.listdir(self.dump_images_dir):
                    m = pat.match(fn)
                    if m:
                        try:
                            nums.append(int(m.group(1)))
                        except ValueError:
                            pass
                if nums:
                    self._seq_counter = max(nums)
            except Exception:
                # klasör okunamazsa sessizce devam
                pass

    # ---------- Loading ----------

    def load_from_hf(self, dataset_name: str, *, split: str = "train") -> "PointSFTDatasetBuilder":
        if load_dataset is None:
            raise ImportError("datasets not installed. `pip install datasets`")
        ds = load_dataset(dataset_name, split=split)
        self._samples = [dict(x) for x in ds]
        return self

    def load_from_local(self, path: str) -> "PointSFTDatasetBuilder":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text.startswith("["):
                self._samples = json.loads(text)  # JSON array
            else:
                self._samples = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        return self

    def add_samples(self, samples: List[Dict[str, Any]]) -> "PointSFTDatasetBuilder":
        self._samples.extend(samples)
        return self

    # ---------- Build ----------

    def build(
        self,
        *,
        max_samples: Optional[int] = None,
        also_bbox: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:


        sft_records: List[Dict[str, Any]] = []
        bbox_records: List[Dict[str, Any]] = []
        stats = dict(
            total=len(self._samples),
            with_points=0,
            emitted=0,
            normalized=0,
            pixel=0,
            skipped_no_image=0
        )

        for i, sample in enumerate(self._samples):
            if max_samples and i >= max_samples:
                break

            conv = sample.get("conversations") or []
            hits = self._parse_conversation_for_points(conv)
            if not hits:
                continue

            stats["with_points"] += 1
            grouped: Dict[Tuple[str, Optional[str]], List[_PointHit]] = defaultdict(list)
            for h in hits:
                grouped[(_norm_text(h.param), h.image_id)].append(h)

            for (_param_norm, _imgid), plist in grouped.items():
                keep = plist[0]

                img_path, wh = self._resolve_image_and_wh(sample, keep.image_id, idx_fallback=0)

                if wh is None and self.default_wh is not None:
                    wh = self.default_wh

                x_out, y_out, was_norm = self._xy_for_output(keep.x_px, keep.y_px, wh)

                if was_norm:
                    stats["normalized"] += 1
                else:
                    stats["pixel"] += 1
                    if img_path is None and self.normalize:
                        stats["skipped_no_image"] += 1

                rec = self._emit_sft_record(
                    param=keep.param,
                    image_path=img_path,
                    x=x_out, y=y_out,
                    normalized=was_norm,
                    source_point=(keep.x_px, keep.y_px),
                    source_units=("percent" if self.coord_source == "percent" else "pixel"),                    
                    source_id=str(sample.get("id") or sample.get("_id") or i),
                    extra_meta={
                        "image_path": img_path,
                        "image_wh": wh,
                        "param": keep.param,
                        "image_id": keep.image_id,
                        "sample_index": i,
                    }
                )
                sft_records.append(rec)

                if also_bbox:
                    bbox = self._make_bbox_from_point(x_out, y_out, wh, was_norm, self.bbox_radius)
                    bbox_rec = {
                        "conversations": [
                            {"from": "user",
                             "value": (
                                 f'Locate: {keep.param}. '
                                 f'Return JSON {{"bbox_2d":[x1,y1,x2,y2]}} '
                                 f'with coords normalized to [0,1].' if was_norm else
                                 f'Locate: {keep.param}. '
                                 f'Return JSON {{"bbox_2d":[x1,y1,x2,y2]}} in raw pixels.'
                             ),
                             "images": [img_path] if img_path else []},
                            {"from": "assistant",
                             "value": json.dumps({"bbox_2d": [round(bbox[0], 6), round(bbox[1], 6),
                                                              round(bbox[2], 6), round(bbox[3], 6)]},
                                                 ensure_ascii=False)}
                        ],
                        "task": "PointBBox",
                        "meta": {
                            "from_point": True,
                            "normalized_target": was_norm,
                            "image_path": img_path,
                            "image_wh": wh,
                            "param": keep.param,
                            "source_id": str(sample.get("id") or sample.get("_id") or i),
                        }
                    }
                    bbox_records.append(bbox_rec)

        stats["emitted"] = len(sft_records)
        return sft_records, bbox_records, stats

    # ---------- Save / Export ----------

    @staticmethod
    def save_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def to_hf_dataset(records: List[Dict[str, Any]]):
        if Dataset is None:
            raise ImportError("datasets not installed. `pip install datasets`")
        return Dataset.from_list(records)

    # ---------- Internals ----------

    def _parse_conversation_for_points(self, conversations: List[Dict[str, Any]]) -> List[_PointHit]:
        hits: List[_PointHit] = []
        pending: Optional[Tuple[str, Optional[str]]] = None  # (param, image_id)

        for idx, turn in enumerate(conversations):
            src = turn.get("from", "")
            val = turn.get("value", "")

            if src == "gpt":
                try:
                    obj = json.loads(val)
                except Exception:
                    continue
                for act in obj.get("actions", []):
                    if act.get("name") == "Point":
                        args = act.get("arguments", {}) or {}
                        param = args.get("param") or args.get("text") or ""
                        image_id = args.get("image")
                        if isinstance(param, str) and param.strip():
                            pending = (param, image_id)

            elif src == "human" and "OBSERVATION" in val and pending is not None:
                for m in POINT_TAG.finditer(val):
                    x_str, y_str, _ = m.groups()
                    x, y = float(x_str), float(y_str)
                    param, image_id = pending
                    hits.append(_PointHit(param=param.strip(), x_px=x, y_px=y,
                                          image_id=image_id, msg_idx=idx))
                    break
                pending = None  
        return hits

    def _resolve_image_and_wh(
        self,
        sample: Dict[str, Any],
        image_id: Optional[str],
        *,
        idx_fallback: int = 0
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:

        imgs = sample.get("images") or sample.get("image") or []
        if isinstance(imgs, str):
            imgs = [imgs]


        index = idx_fallback
        if image_id:
            try:
                index = int(str(image_id).split("_")[-1]) - 1  # "img_1"->0
            except Exception:
                index = idx_fallback

        img_obj = None
        if isinstance(imgs, list) and len(imgs) > 0:
            if 0 <= index < len(imgs):
                img_obj = imgs[index]
            else:
                img_obj = imgs[0]


        if isinstance(img_obj, str):
            path = (os.path.join(self.images_root, img_obj)
                    if self.images_root and not os.path.isabs(img_obj) else img_obj)
            wh = self._probe_wh_from_path(path)
            return path, wh


        if Image is not None and isinstance(img_obj, Image.Image):
            wh = (img_obj.width, img_obj.height)
            path = self._maybe_save_image(img_obj, sample, index)
            return path, wh


        if isinstance(img_obj, dict):

            if "path" in img_obj and isinstance(img_obj["path"], str):
                path = (os.path.join(self.images_root, img_obj["path"])
                        if self.images_root and not os.path.isabs(img_obj["path"]) else img_obj["path"])
                wh = self._probe_wh_from_path(path)
                return path, wh

            if Image is not None and "bytes" in img_obj and isinstance(img_obj["bytes"], (bytes, bytearray)):
                try:
                    im = Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
                    wh = (im.width, im.height)
                    path = self._maybe_save_image(im, sample, index)
                    return path, wh
                except Exception:
                    pass

            if Image is not None and "array" in img_obj:
                try:
                    import numpy as np  # lazy
                    arr = img_obj["array"]
                    if isinstance(arr, np.ndarray):
                        im = Image.fromarray(arr)
                        wh = (im.width, im.height)
                        path = self._maybe_save_image(im, sample, index)
                        return path, wh
                except Exception:
                    pass

        return None, None

    def _maybe_save_image(self, im, sample: Dict[str, Any], index: int) -> Optional[str]:
        if self.dump_images_dir is None or Image is None:
            return None

        if self.sequential_naming:
            n = self._seq_counter + 1
            while True:
                fname = f"{self.sequential_prefix}_{n}.png"
                out_path = os.path.join(self.dump_images_dir, fname)
                if not os.path.exists(out_path):
                    break
                n += 1
            try:
                im.save(out_path)
                self._seq_counter = n  
                return out_path
            except Exception:
                return None

        source_id = sample.get("id") or sample.get("_id") or "sample"
        fname = f"{str(source_id)}_{index}.png"
        out_path = os.path.join(self.dump_images_dir, fname)
        if self.prefer_saved_paths and os.path.exists(out_path):
            return out_path
        try:
            im.save(out_path)
            return out_path
        except Exception:
            return None

    @staticmethod
    def _probe_wh_from_path(path: Optional[str]) -> Optional[Tuple[int, int]]:
        if not path or Image is None:
            return None
        try:
            with Image.open(path) as im:
                return im.width, im.height
        except Exception:
            return None

    def _xy_for_output(
        self,
        x_tag: float,
        y_tag: float,
        wh: Optional[Tuple[int, int]]
    ) -> Tuple[float, float, bool]:
        if self.coord_source == "percent":
            return x_tag / 100.0, y_tag / 100.0, True


        if self.normalize and wh:
            w, h = wh
            if w > 0 and h > 0:
                return x_tag / float(w), y_tag / float(h), True
        return x_tag, y_tag, False

    @staticmethod
    def _emit_sft_record(
        *,
        param: str,
        image_path: Optional[str],
        x: float,
        y: float,
        normalized: bool,
        source_point: Optional[Tuple[float, float]],
        source_units: str,
        source_id: Optional[str],
        extra_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        user_msg = (
            f'Locate: {param}. Return JSON {{"point_2d":[x,y]}} with x,y normalized to [0,1].'
            if normalized else
            f'Locate: {param}. Return JSON {{"point_2d":[x,y]}} in raw pixel space.'
        )
        conv = [
            {"from": "user", "value": user_msg, "images": [image_path] if image_path else []},
            {"from": "assistant", "value": json.dumps({"point_2d": [round(x, 6), round(y, 6)]}, ensure_ascii=False)}
        ]
        meta = {
            "normalized_target": normalized,
            "source_point": {"x": source_point[0], "y": source_point[1]} if source_point else None,
            "source_units": source_units,
            "source_id": source_id,
        }
        meta.update(extra_meta or {})
        return {"conversations": conv, "task": "Point", "meta": meta}

    @staticmethod
    def _make_bbox_from_point(
        x: float, y: float,
        wh: Optional[Tuple[int, int]],
        normalized: bool,
        bbox_radius: float
    ) -> List[float]:
        if normalized:
            r = float(bbox_radius)
            return [max(0.0, x - r), max(0.0, y - r),
                    min(1.0, x + r), min(1.0, y + r)]
        # pixel coords
        if wh:
            w, h = wh
            r = bbox_radius * min(w, h) if bbox_radius < 1.0 else bbox_radius
        else:
            r = bbox_radius if bbox_radius >= 1.0 else 4.0
        return [x - r, y - r, x + r, y + r]