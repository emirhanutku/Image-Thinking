import io, base64
from typing import Any
from PIL import Image

def increment_image_number(image_str: str) -> str:
    prefix, num_str = image_str.split("_")
    num = int(num_str)
    return f"{prefix}_{num+1}"
def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def ensure_pil(x: Any) -> Image.Image:

    if isinstance(x, Image.Image):
        return x
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    if isinstance(x, str):
        return Image.open(x)
    if isinstance(x, dict):

        if x.get("path"):
            return Image.open(x["path"])
        if x.get("bytes") is not None:
            return Image.open(io.BytesIO(x["bytes"]))
        if x.get("array") is not None:
            try:
                import numpy as np
                return Image.fromarray(np.asarray(x["array"]))
            except Exception:
                pass
    raise TypeError(f"Unsupported image type: {type(x)}")
