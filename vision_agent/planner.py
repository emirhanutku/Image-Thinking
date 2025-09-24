import json
from typing import Dict, Any, List
from PIL import Image
from openai import OpenAI
from .config import MODEL_NAME, TEMPERATURE
from .schema import PLANNING_SCHEMA
from .prompt import PromptBuilder
from .utils.image_io import pil_to_data_url

class VisionPlanner:
    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI()
        self.builder = PromptBuilder()

    def _examples_to_dicts(self, examples) -> List[dict]:
        return [
            {
                "user_text": e.user_text,
                "image_data_urls": e.image_data_urls,
                "assistant_json": e.assistant_json,

                "tools_seq": e.tools_seq,
                "final_ans": e.final_ans,
                "thought": e.thought,
            }
            for e in examples
        ]

    def plan(self, image_path: str, question: str, examples) -> Dict[str, Any]:
        pil = Image.open(image_path)
        
        data_url = pil_to_data_url(pil)
        msgs = self.builder.build_messages(self._examples_to_dicts(examples), question, data_url)


        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            response_format={"type": "json_schema", "json_schema": PLANNING_SCHEMA},
            messages=msgs,
        )
        
        content = resp.choices[0].message.content
        return json.loads(content)
