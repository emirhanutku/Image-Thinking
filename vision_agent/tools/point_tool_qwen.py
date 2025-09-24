# Qwen2-VL 2B "Point" tool — LoRA adapter (macOS/MPS friendly)

import os, json, torch
from typing import Dict, Any
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Paths / IDs
ADAPTER_DIR = os.environ.get(
    "QWEN_ADAPTER_DIR",
    "/Users/emirhan/Desktop/checkpoint-420"
)
BASE_ID = os.environ.get("QWEN_BASE_ID", "Qwen/Qwen2-VL-2B-Instruct")

USE_FAST_PROCESSOR = False

MIN_PIXELS = 1            # never force an upsample
MAX_PIXELS = 10_000_000   # ~10 MP cap; raise if your images are bigger

DEFAULT_MAX_NEW_TOKENS = 32

SYSTEM = (
    'You are the Point tool. Given an image and a textual query, '
    'return only a strict JSON object: {"point_2d":[x, y]}. '
    'x = fraction of ORIGINAL image width, y = fraction of ORIGINAL image height; both in [0,1].'
)
_PROCESSOR = None
_MODEL = None
_DEVICE = None
_DTYPE = None

def _setup_device():
    global _DEVICE, _DTYPE
    force_cpu = os.environ.get("QWEN_FORCE_CPU", "0") == "1"
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if use_mps and not force_cpu:
        _DEVICE, _DTYPE = torch.device("mps"), torch.float16
    else:
        _DEVICE, _DTYPE = torch.device("cpu"), torch.float32

def _load_once():
    global _PROCESSOR, _MODEL
    if _MODEL is not None:
        return
    if _DEVICE is None:
        _setup_device()

    proc_src = ADAPTER_DIR if os.path.exists(
        os.path.join(ADAPTER_DIR, "preprocessor_config.json")
    ) else BASE_ID
    _PROCESSOR = AutoProcessor.from_pretrained(
        proc_src, use_fast=USE_FAST_PROCESSOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_ID, dtype=_DTYPE, attn_implementation="sdpa"
    )
    _MODEL = PeftModel.from_pretrained(base, ADAPTER_DIR)
    _MODEL.to(_DEVICE); _MODEL.eval()
    if hasattr(_MODEL.config, "use_cache"):
        _MODEL.config.use_cache = False

def _parse_point_json(s: str):
    s = s.strip().strip("`").strip()
    try:
        return json.loads(s)["point_2d"]
    except Exception:
        import re
        m = re.search(r'"point_2d"\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]', s)
        if not m:
            raise
        return [float(m.group(1)), float(m.group(2))]

def _denorm_point(norm_xy, wh):
    w, h = wh
    x = max(0.0, min(1.0, float(norm_xy[0])))
    y = max(0.0, min(1.0, float(norm_xy[1])))
    return [x * w, y * h]

def qwen_point_2b(pil_img: Image.Image, query: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Dict[str, Any]:
    _load_once()
    if not isinstance(pil_img, Image.Image):
        pil_img = Image.open(pil_img).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": query},
        ]},
    ]
    gen_text = _PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = _PROCESSOR(text=[gen_text], images=image_inputs, return_tensors="pt")
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(_DEVICE)

    with torch.inference_mode():
        try:
            out_ids = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)
        except RuntimeError as e:
            if "MPS" in str(e):
                _MODEL.to("cpu")
                for k, v in list(inputs.items()):
                    if hasattr(v, "to"):
                        inputs[k] = v.to("cpu")
                out_ids = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)
            else:
                raise
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        out_txt = _PROCESSOR.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    w, h = pil_img.size
    norm_xy = _parse_point_json(out_txt)
    px_xy = _denorm_point(norm_xy, (w, h))
    return {"point_2d": norm_xy, "point_px": px_xy, "image_wh": [w, h], "raw": out_txt}
