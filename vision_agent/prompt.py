from typing import List, Dict, Any
from .tools_def import TOOLS_DEFINITION_TEXT
from .rationale_translator import RationaleTranslator
import json
from typing import List, Dict, Any, Optional


SYSTEM_CORE = """
[BEGIN OF TASK INSTRUCTIONS]
1. Select the appropriate action(s) from the list of tools provided (# ACTIONS #).
2. Actions are combined logically to solve the problem, with each action building upon the previous.
3. Call one action at a time, and ensure the output from one action informs the next step.
4. If no action is required, leave the "actions" array empty (e.g., "actions": []).
5. The output of the "Point" action will not be a new image but the coordinates of the identified point.
6. After modifying an image, label the new image (e.g., img2) based on the previous image (e.g., img1).
7. Always refer to images in tool arguments by logical ids like "img_1", "img_2", ... Never pass raw URLs to tools.
8. Always include a call to "Terminate" with the final answer when the task is completed.
[END OF TASK INSTRUCTIONS]
"""




class PromptBuilder:
    def __init__(self):
        pass

    def build_messages(self, exemplars: List[dict], user_question: str, user_image_data_url: str) -> List[Dict[str, Any]]:
          
        msgs: List[Dict[str, Any]] = [{
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_CORE},
                {"type": "text", "text": TOOLS_DEFINITION_TEXT},
            ],
        }]
        for i, ex in enumerate(exemplars, start=1):
            tools_hint = ", ".join(ex.get("tools_seq", [])) if ex.get("tools_seq") else "—"
            final_ans  = ex.get("final_ans", "")
            short_thought = ex.get("thought", "")
            
            content_blocks = [
                {
                    "type": "text",
                    "text": (
                        f"Example {i}\n"
                        f"User question:\n{ex['user_text']}\n\n"
                        f"Prior tools used: {tools_hint}\n"
                        f"(Do not copy) Final answer in that run: {final_ans}\n"
                        f"(Pattern hint) Short rationale from that run: {short_thought}"
                    ),
                }
            ]
            img_urls = ex.get("image_data_urls", [])
            for j, url in enumerate(img_urls, start=1):
                if url:
                    content_blocks.append({
                        "type": "text",
                        "text": f"[EXAMPLE_IMAGE_ID=ex{i}_img_{j}] (Use this id if referring to this example image; do not use the URL in tool arguments.)"
                    })
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })
            msgs.append({
                "role": "user",
                "content": content_blocks,
            })

            # Assistant plan
            msgs.append({
                "role": "assistant",
                "content": ex["assistant_json"],  # {"thought":"...","actions":[...]}
            })

        # Example question
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Current image id: img_1. Use this id in ALL tool arguments. Never pass raw URLs to tools."},
                {"type": "text", "text": user_question},
                {"type": "image_url", "image_url": {"url": user_image_data_url}},
            ],
        })


        return msgs
    

    def build_llm_integrated_messages(
        self,
        examples: List[Dict[str, Any]],
        user_question: Optional[str] = None,
        user_image_url: Optional[str] = None,
        *,
        translator_model: str = "gpt-4.1-mini-2025-04-14",
        translator_temperature: float = 0.3,
        per_example_header: bool = True,
    ) -> List[Dict[str, Any]]:
        
        msgs: List[Dict[str, Any]] = [{
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_CORE},
                {"type": "text", "text": TOOLS_DEFINITION_TEXT},
            ],
        }]
        translator = RationaleTranslator(model=translator_model, temperature=translator_temperature)

        for idx, ex in enumerate(examples):
            try:
                rationale = translator.translate_example(ex)  # dict, schema-constrained
            except Exception as e:
                # Skip bad examples without breaking the prompt
                print(f"Error translating example {idx+1}: {e}")
                continue

            header = f"[TOOL USE RATIONALE SPEC – EXAMPLE {idx+1}]\n" if per_example_header else ""
            msgs.append({
                "role": "assistant",
                "content": f"{header}{json.dumps(rationale, ensure_ascii=False)}"
            })

        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Current image id: img_1. Use this id in ALL tool arguments. Never pass raw URLs to tools."},
                {"type": "text", "text": user_question},
                {"type": "image_url", "image_url": {"url": user_image_url}},
            ],
        })
        return msgs