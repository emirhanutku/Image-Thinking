from __future__ import annotations
from typing import Any, Dict, Tuple
from PIL import Image, ImageDraw, ImageColor, ImageFont

def _to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")

def _parse_color(c: Any, alpha: int) -> Tuple[int,int,int,int]:
    if isinstance(c, tuple):
        if len(c) == 4: return tuple(c)
        if len(c) == 3: return (c[0], c[1], c[2], alpha)
    try:
        r,g,b = ImageColor.getrgb(str(c))
        return (r,g,b,alpha)
    except Exception:
        return (255,45,85,alpha)
    
def _draw_dashed(draw: ImageDraw.ImageDraw, y: int, x1: int, x2: int,
                 color: Tuple[int,int,int,int], width: int, dash: int, gap: int):
    x = x1
    while x < x2:
        x_end = min(x + dash, x2)
        draw.line([(x, y), (x_end, y)], fill=color, width=width)
        x += dash + gap
    

def draw_horizontal_line_by_y(
    pil: Image.Image,
    y: float,
    *,
    color: Any = "#FF2D55",
    thickness: int = 3,
    dashed: bool = True,
    dash: int = 10,
    gap: int = 6,
    alpha: int = 255,
    normalized: bool = False,
    margin: int = 15,
) -> Dict[str, Any]:

    im = _to_rgba(pil)
    W, H = im.size

    # y: normalize if requested
    if normalized:
        y_px = int(round(max(0.0, min(1.0, float(y))) * (H - 1)))
    else:
        y_px = int(round(float(y)))
    y_px = max(0, min(H - 1, y_px))

    draw = ImageDraw.Draw(im, "RGBA")
    col = _parse_color(color, int(max(0, min(255, alpha))))
    x1, x2 = 0 + margin, W - 1 - margin

    if dashed:
        _draw_dashed(draw, y_px, x1, x2, col, max(1, int(thickness)), max(1, int(dash)), max(1, int(gap)))
    else:
        draw.line([(x1, y_px), (x2, y_px)], fill=col, width=max(1, int(thickness)))

    #im.show()

    return im


class HorizontalLineTool:
    def __init__(self, image: Image.Image , param: str):
        self.image = image
        self.param = param

    def draw_line(self) -> Image.Image:

        if isinstance(self.param, (int, float)):
            y = float(self.param)
            opts = {}
        elif isinstance(self.param, dict):
            if "y" not in self.param:
                raise ValueError("param must be a number or an object containing 'y'.")
            y = float(self.param["y"])
            opts = {k: v for k, v in self.param.items() if k != "y"}
        else:
            raise ValueError("param must be a number or an object with {'y': ...}.")

        return draw_horizontal_line_by_y(self.image, y, **opts)

        
