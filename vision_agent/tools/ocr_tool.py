from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

class OCRTool:
    def __init__(self, paddle_ocr: PaddleOCR , image : Image.Image , crop_rate_bottom_right: float):
        self._paddle_ocr = paddle_ocr
        self._image = image
        self._crop_rate_bottom_right = crop_rate_bottom_right

    def __call__(self):                       
        left = 0
        right = self._image.width * self._crop_rate_bottom_right
        top = 0
        bottom = self._image.height * self._crop_rate_bottom_right
        cropped_image = self._image.copy().crop((left, top, right, bottom))
        arr = np.array(cropped_image.convert("RGB"))

        outputs = self._paddle_ocr.predict(arr)
        words = []
        for res in outputs:
            jd = res.json  # {'res': {...}}
            words.extend([t for t in jd["res"].get("rec_texts", []) if t])
        text = ",".join([f"'{w}'" for w in words])

        output = f"OBSERVATION:\nOCR model outputs: [{text}]\nPlease summarize the model outputs and answer my first question."

        return output
            


