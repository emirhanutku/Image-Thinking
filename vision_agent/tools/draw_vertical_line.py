from __future__ import annotations
from typing import Any, Dict, Tuple
from PIL import Image, ImageDraw, ImageColor, ImageFont

def _to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")

def _parse_color(c: Any, alpha: int) -> Tuple[int,int,int,int]:
    if isinstance(c, tuple):
        if len(c) == 4: return tuple(c)  # type: ignore
        if len(c) == 3: return (c[0], c[1], c[2], alpha)  # type: ignore
    try:
        r,g,b = ImageColor.getrgb(str(c))
        return (r,g,b,alpha)
    except Exception:
        return (255,45,85,alpha)  

def _draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int,
                          color: Tuple[int,int,int,int], width: int, dash: int, gap: int):
    y = y1
    while y < y2:
        y_end = min(y + dash, y2)
        draw.line([(x, y), (x, y_end)], fill=color, width=width)
        y += dash + gap

def draw_vertical_line_by_x(
    pil: Image.Image,
    x: float,
    *,
    color: Any = "#FF2D55",
    thickness: int = 3,
    dashed: bool = True,
    dash: int = 10,
    gap: int = 6,
    alpha: int = 255,
    normalized: bool = False,
    margin: int = 6,
) -> Dict[str, Any]:
    
    im = _to_rgba(pil)
    W, H = im.size

    if normalized:
        x_px = int(round(max(0.0, min(1.0, float(x))) * (W - 1)))
    else:
        x_px = int(round(float(x)))
    x_px = max(0, min(W - 1, x_px))

    draw = ImageDraw.Draw(im, "RGBA")
    col = _parse_color(color, int(max(0, min(255, alpha))))
    y1, y2 = 0 + margin, H - 1 - margin

    if dashed:
        _draw_dashed_vertical(draw, x_px, y1, y2, col, max(1, int(thickness)), max(1, int(dash)), max(1, int(gap)))
    else:
        draw.line([(x_px, y1), (x_px, y2)], fill=col, width=max(1, int(thickness)))

    return im


class VerticalLineTool:
    def __init__(self, image: Image.Image , param: str): 
        self.image = image
        self.param = param
    def draw_line(self) -> Dict[str, Any]:
        if isinstance(self.param, (int, float)):
            x = float(self.param)
            opts = {}
        elif isinstance(self.param, dict):
            if "x" not in self.param:
                raise ValueError("param must be a number or an object containing 'x'.")
            x = float(self.param["x"])
            opts = {k: v for k, v in self.param.items() if k != "x"}
        else:
            raise ValueError("param must be a number or an object with {'x': ...}.")

        return draw_vertical_line_by_x(self.image, x, **opts)