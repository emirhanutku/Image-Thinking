from __future__ import annotations
from .schema import RATIONALE_SCHEMA
from typing import Dict, Any, List
from openai import OpenAI
from base64 import b64encode
import mimetypes
import json
import hashlib
import os

MAX_IMAGES = 4

TRANSLATOR_SYSTEM = (
    "You convert tool-using trajectories into a SHORT, structured rationale.\n"
    "Explain reasons per tool and ordering at a high level, teaching another model HOW to act.\n"
    "Do not include chain-of-thought or step-by-step derivations; keep each field concise.\n"
    "Use the provided JSON schema exactly."
)
def _hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _to_image_block(url_or_path: str) -> dict:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return {"type": "image_url", "image_url": {"url": url_or_path}}

    if not os.path.isfile(url_or_path):
        return {
            "type": "text",
            "text": f"[warn] image not found: {url_or_path}"
        }

    mime, _ = mimetypes.guess_type(url_or_path)
    mime = mime or "image/png"
    with open(url_or_path, "rb") as f:
        b64 = b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class RationaleTranslator:
    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14", temperature: float = 0.3):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self._cache: dict[str, dict] = {}

    def _build_steps_preview(self, assistant_json: str) -> List[Dict[str, Any]]:

        try:
            traj = json.loads(assistant_json or "{}")
        except Exception:
            traj = {}

        steps_preview: List[Dict[str, Any]] = []
        for i, a in enumerate(traj.get("actions", [])):
            nm = a.get("name", "")
            if nm == "Terminate":
                continue
            steps_preview.append({
                "step": i + 1,
                "tool": nm,
                "args": a.get("arguments", {})
            })
        return steps_preview

    def translate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        
        key = _hash({
            "u": example.get("user_text", ""),
            "a": example.get("assistant_json", ""),
            "imgs": example.get("image_urls") or example.get("image_data_urls") or example.get("local_image_paths") or []
        })
        if key in self._cache:
            return self._cache[key]

        steps_preview = self._build_steps_preview(example.get("assistant_json", ""))

        # Build a multimodal USER message: text + up to MAX_IMAGES
        content_blocks: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                "Summarize WHY each tool was used and WHY this order makes sense.\n"
                "Generalize for similar charts. Keep each field concise.\n\n"
                f"User question:\n{example.get('user_text','')}\n\n"
                f"Observed tool steps (tool + args):\n{json.dumps(steps_preview, ensure_ascii=False)}\n"
                f"Final answer (context): {example.get('final_ans','')}\n"
                "Rules:\n"
                "- Use logical image ids (img_1, ...) only as examples.\n"
                "- Do NOT output chain-of-thought.\n"
            )
        }]


        # Gather images from the three optional sources, in priority order
        imgs: List[str] = []
        for key_src in ("image_urls", "image_data_urls", "local_image_paths"):
            vals = example.get(key_src)
            if isinstance(vals, list):
                imgs.extend(vals)

        # Clip and append images
        for u in imgs[:MAX_IMAGES]:
            if u.startswith("data:"):
                # already a data URL
                content_blocks.append({"type": "image_url", "image_url": {"url": u}})
            else:
                content_blocks.append(_to_image_block(u))

        messages = [
            {"role": "system", "content": TRANSLATOR_SYSTEM},
            {"role": "user", "content": content_blocks}
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": RATIONALE_SCHEMA},
        )
        content = (resp.choices[0].message.content or "").strip()

        try:
            data = json.loads(content)
        except Exception:
            # Minimal robust fallback to avoid breaking the pipeline
            data = {
                "problem_signature": (example.get("user_text", "") or "")[:200],
                "tools_sequence": [
                    {"tool": s.get("tool", ""), "purpose": "", "why_order": ""}
                    for s in steps_preview
                ],
                "global_rules": []
            }

        self._cache[key] = data
        return data
    


