from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from PIL import Image
from .utils.image_io import ensure_pil, pil_to_data_url,increment_image_number
from PIL import Image
from vision_agent.tools.point_tool import PointTool
from vision_agent.tools.ocr_tool import OCRTool
from vision_agent.tools.zoom_in_subfigure_tool import ZoomInSubfigureTool
from paddleocr import PaddleOCR
from vision_agent.tools.segment_region_around_point_tool import  SegmentTool
from vision_agent.tools.draw_horizontal_line import HorizontalLineTool 
from vision_agent.tools.draw_vertical_line import VerticalLineTool
    

_paddle = PaddleOCR(
    device="cpu",
    use_textline_orientation=True,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
# --- Image store ---
@dataclass
class ImageRef:
    pil: Image.Image
    url: str  # Data url

class ImageStore:
    def __init__(self):
        self._images: Dict[str, ImageRef] = {}
        self._counter = 1 

    def add(self, pil: Image.Image, name: Optional[str] = None) -> str:
        if name is None:
            name = f"img_{self._counter}"
            self._counter += 1
            
        
        pil = ensure_pil(pil)
        url = pil_to_data_url(pil)
        self._images[name] = ImageRef(pil=pil, url=url)
        return name

    def get_pil(self, name: str) -> Image.Image:
        return self._images[name].pil

    def get_url(self, name: str) -> str:
        return self._images[name].url


def tool_OCR(image_store: ImageStore, image: str) -> Dict[str, Any]:
    ocr_tool = OCRTool(_paddle, image_store.get_pil(image), crop_rate_bottom_right=0.95)

    return ocr_tool()

def tool_Point(image_store: ImageStore, image: str, param: str) -> Dict[str, Any]:
    pil_img: Image.Image = image_store.get_pil(image)
    _point = PointTool(prefer='auto')

    return _point(pil_img, param)

def tool_ZoomInSubfigure(image_store: ImageStore, image: str, param: str) -> str:
    pil = image_store.get_pil(image)
    _zoom = ZoomInSubfigureTool(pil, llm_model='gpt-5-2025-08-07', margin_frac=0.1, _paddle_ocr=_paddle)
    out = _zoom(param)
    new_image = out["crop"]
    image_store.add(new_image,increment_image_number(image))
    return increment_image_number(image)

def tool_SegmentRegionAroundPoint(image_store: ImageStore, image: str, param: Any) -> str:
    pil = image_store.get_pil(image)
    segment_tool = SegmentTool("facebook/sam2.1-hiera-large")
    out = segment_tool(pil, param)
    image_store.add(out, increment_image_number(image))
    return increment_image_number(image)

def tool_DrawHorizontalLineByY(image_store: ImageStore, image: str, param: Any) -> str:
    pil = image_store.get_pil(image)
    _line_tool = HorizontalLineTool(pil, param)
    out =  _line_tool.draw_line()
    image_store.add(out, increment_image_number(image))
    return increment_image_number(image)

def tool_DrawVerticalLineByX(image_store: ImageStore, image: str, param: Any) -> str:
    pil = image_store.get_pil(image)
    _line_tool = VerticalLineTool(pil, param)
    out =  _line_tool.draw_line()
    image_store.add(out, increment_image_number(image))
    return increment_image_number(image)