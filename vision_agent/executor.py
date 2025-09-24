from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
from PIL import Image
from openai import OpenAI

from .config import MODEL_NAME, TEMPERATURE
from .tools_spec import TOOLS_SPEC
from .prompt import PromptBuilder
from .runtime_tools import (
    ImageStore, tool_OCR, tool_Point, tool_ZoomInSubfigure,
    tool_SegmentRegionAroundPoint, tool_DrawHorizontalLineByY, tool_DrawVerticalLineByX
)


TOOL_CHOICE: str = "auto"
MAX_STEPS: int = 10
MAX_NO_TOOL_TURNS: int = 2 

# tool name -> python function
PY_TOOL_IMPLS = {
    "OCR": tool_OCR,
    "Point": tool_Point,
    "ZoomInSubfigure": tool_ZoomInSubfigure,
    "SegmentRegionAroundPoint": tool_SegmentRegionAroundPoint,
    "DrawHorizontalLineByY": tool_DrawHorizontalLineByY,
    "DrawVerticalLineByX": tool_DrawVerticalLineByX,
}

# Tools that *emit* new images in their result
IMAGE_EMITTING_TOOLS = {
    "ZoomInSubfigure",
    "SegmentRegionAroundPoint",
    "DrawHorizontalLineByY",
    "DrawVerticalLineByX",
}
 

def _assistant_message_dict(msg) -> Dict[str, Any]:
    out : Dict[str, Any] = {
        "role":"assistant",
        "content":msg.content or ""
    }
    
    if getattr(msg,"tool_calls",None):
        out["tool_calls"] = []
        for tc in msg.tool_calls:
            out["tool_calls"].append({
                "id": tc.id,
                "type":tc.type,
                "function":{
                    "name":tc.function.name,
                    "arguments":tc.function.arguments or {},
                }
            })
    return out

class VisionExecutor:
    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI()
        self.builder = PromptBuilder()
        self.image_store = ImageStore()

    def _examples_to_dicts(self, examples) -> List[dict]:
        return [
            {
                "user_text": e.user_text,
                "image_data_urls": e.image_data_urls,
                "assistant_json": e.assistant_json,
                "tools_seq": e.tools_seq,
                "final_ans": e.final_ans,
                "thought": e.thought,
            } for e in examples
        ]
    
    def _inject_image_message(self, msgs: List[Dict[str, Any]], image_id: str) -> None:
        """After a tool creates a new image, show it to the model with its logical id."""
        try:
            url = self.image_store.get_url(image_id)
        except KeyError:
            return  # unknown id; skip gracefully
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"New image id: {image_id}. Use this id in ALL tool arguments. Never pass raw URLs."},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        })

    def run(self, image_path: Optional[str], question: str, examples, latency: bool = True) -> Dict[str, Any]:
        if isinstance(image_path, Image.Image):
            pil = image_path.convert("RGB")

        else:
            pil = Image.open(image_path).convert("RGB")
        img_id = self.image_store.add(pil, name="img_1")
        assert img_id == "img_1"
        
        if not latency: 
            msgs = self.builder.build_messages(
                self._examples_to_dicts(examples),
                question,
                self.image_store.get_url(img_id),
            )
        else:
            msgs = self.builder.build_llm_integrated_messages(
                self._examples_to_dicts(examples),
                user_question=question,
                user_image_url=self.image_store.get_url(img_id),
            translator_model='gpt-4.1-mini-2025-04-14',     # or "o4-mini"
            translator_temperature=0.3,
            per_example_header=True,          
            )

        final_answer : str = None
        transcript: List[Dict[str, Any]] = []
        steps = 0
        no_tool_turns = 0

        while steps < MAX_STEPS:
            steps += 1
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                tools=TOOLS_SPEC,
                tool_choice="auto",   
                messages=msgs,
            )
            choice = resp.choices[0]
            msg = choice.message

            assistant_dict = _assistant_message_dict(msg)
            msgs.append(assistant_dict)
            #print("msg : " , msg)

            #If no tool calls
            if not msg.tool_calls:
                content = (msg.content or "").strip()
                no_tool_turns += 1

                msgs.append({
                    "role": "user",
                    "content": (
                        "You wrote a thought but did not call a tool. "
                        "Choose the best next tool (OCR, Point, ZoomInSubfigure, "
                        "SegmentRegionAroundPoint, DrawHorizontalLineByY, DrawVerticalLineByX) "
                        "and continue. When ready to answer, call Terminate. "
                        f"Original question: {question}"
                    )
                })

                if no_tool_turns >= MAX_NO_TOOL_TURNS:
                    # last gentle push to end gracefully
                    msgs.append({
                        "role": "system",
                        "content": (
                            "Reminder: You must call a tool each step and finish with Terminate."
                        )
                    })
                continue

            #If tool calls are present
            no_tool_turns = 0
            for tc in msg.tool_calls:
                name = tc.function.name

                args = json.loads(tc.function.arguments or "{}")


                #Last answer
                if name == "Terminate":
                    final_answer = args.get("ans", "")

                    # Also append a tool result so the log is complete
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps({"ans": final_answer}, ensure_ascii=False),
                    })
                    transcript.append({"tool": name, "args": args, "result": {"ans": final_answer}})
                    break

                py_fn = PY_TOOL_IMPLS.get(name)

                if py_fn is None:
                    tool_result = {"error": f"Unknown tool {name}"}
                else:
                    try:
                        tool_result = py_fn(self.image_store, **args)
                    except TypeError:
                        # Attempt a best-effort call with common arg names
                        call_args = []
                        if "image" in args:
                            call_args.append(args["image"])
                        if "param" in args:
                            call_args.append(args["param"])
                        tool_result = py_fn(self.image_store, *call_args)
                #print("tool_result",tool_result)

                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": tool_result,
                })
                transcript.append({"tool": name, "args": args, "result": tool_result})

                # Inject image message if tool emits a new image
                if name in IMAGE_EMITTING_TOOLS:
                    self._inject_image_message(msgs, tool_result)

            if final_answer is not None:
                break

        
        # If we exit due to MAX_STEPS without a Terminate, attempt graceful end:
        if final_answer is None:
            msgs.append({
                "role": "system",
                "content": "Please call Terminate now with your final short answer."
            })
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                tools=TOOLS_SPEC,
                tool_choice="required",   # force tool call (Terminate)
                messages=msgs,
            )
            choice = resp.choices[0]
            msg = choice.message
            assistant_dict = _assistant_message_dict(msg)
            msgs.append(assistant_dict)

            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args_json = tc.function.arguments or "{}"
                    try:
                        args = json.loads(args_json)
                    except Exception:
                        args = {}
                    if name == "Terminate":
                        final_answer = args.get("ans", "")
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps({"ans": final_answer}, ensure_ascii=False),
                        })
                        transcript.append({"tool": name, "args": args, "result": {"ans": final_answer}})
                        break


        return {
            "final_answer": final_answer,
            "transcript": transcript,
        }    

    def run_stream(self, image_path: Optional[str], question: str, examples, latency: bool = True):
        
        try:
            if isinstance(image_path, Image.Image):
                pil = image_path.convert("RGB")
            else:
                pil = Image.open(image_path).convert("RGB")
            img_id = self.image_store.add(pil, name="img_1")

            if not latency:
                msgs = self.builder.build_messages(
                    self._examples_to_dicts(examples),
                    question,
                    self.image_store.get_url(img_id),
                )
            else:
                msgs = self.builder.build_llm_integrated_messages(
                    self._examples_to_dicts(examples),
                    user_question=question,
                    user_image_url=self.image_store.get_url(img_id),
                    translator_model='gpt-4.1-mini-2025-04-14',
                    translator_temperature=0.3,
                    per_example_header=True,
                )

            yield {"event": "start", "question": question}

            final_answer: Optional[str] = None
            transcript: List[Dict[str, Any]] = []
            steps = 0
            no_tool_turns = 0

            while steps < MAX_STEPS:
                steps += 1
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=TEMPERATURE,
                    tools=TOOLS_SPEC,
                    tool_choice="auto",
                    messages=msgs,
                )
                choice = resp.choices[0]
                msg = choice.message

                assistant_dict = _assistant_message_dict(msg)
                msgs.append(assistant_dict)

                if not msg.tool_calls:
                    content = (msg.content or "").strip()
                    no_tool_turns += 1

                    msgs.append({
                        "role": "user",
                        "content": (
                            "You wrote a thought but did not call a tool. "
                            "Choose the best next tool (OCR, Point, ZoomInSubfigure, "
                            "SegmentRegionAroundPoint, DrawHorizontalLineByY, DrawVerticalLineByX) "
                            "and continue. When ready to answer, call Terminate. "
                            f"Original question: {question}"
                        )
                    })

                    if no_tool_turns >= MAX_NO_TOOL_TURNS:
                        msgs.append({
                            "role": "system",
                            "content": (
                                "Reminder: You must call a tool each step and finish with Terminate."
                            )
                        })
                    continue

                no_tool_turns = 0
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")

                    if name == "Terminate":
                        final_answer = args.get("ans", "")
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps({"ans": final_answer}, ensure_ascii=False),
                        })
                        transcript.append({"tool": name, "args": args, "result": {"ans": final_answer}})
                        yield {"event": "final_answer", "answer": final_answer}
                        break

                    # Announce tool start
                    try:
                        yield {"event": "tool_started", "tool": name, "args": args}
                    except Exception:
                        # Streaming should not break execution
                        pass

                    py_fn = PY_TOOL_IMPLS.get(name)
                    if py_fn is None:
                        tool_result: Any = {"error": f"Unknown tool {name}"}
                    else:
                        try:
                            tool_result = py_fn(self.image_store, **args)
                        except TypeError:
                            call_args = []
                            if "image" in args:
                                call_args.append(args["image"])
                            if "param" in args:
                                call_args.append(args["param"])
                            tool_result = py_fn(self.image_store, *call_args)

                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": tool_result,
                    })
                    transcript.append({"tool": name, "args": args, "result": tool_result})

                    payload: Dict[str, Any] = {"event": "tool_result", "tool": name, "result": tool_result}
                    if name in IMAGE_EMITTING_TOOLS:
                        # Make the new image available to the model in subsequent turns
                        try:
                            self._inject_image_message(msgs, tool_result)
                        except Exception:
                            pass
                        # Also surface it to the client
                        try:
                            image_id = tool_result  # per tools, image-emitting result is id string
                            image_url = self.image_store.get_url(image_id)
                            payload["image"] = {"id": image_id, "url": image_url}
                        except Exception:
                            pass
                    try:
                        yield payload
                    except Exception:
                        pass

                if final_answer is not None:
                    break

            if final_answer is None:
                msgs.append({
                    "role": "system",
                    "content": "Please call Terminate now with your final short answer."
                })
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=TEMPERATURE,
                    tools=TOOLS_SPEC,
                    tool_choice="required",
                    messages=msgs,
                )
                choice = resp.choices[0]
                msg = choice.message
                assistant_dict = _assistant_message_dict(msg)
                msgs.append(assistant_dict)

                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        name = tc.function.name
                        args_json = tc.function.arguments or "{}"
                        try:
                            args = json.loads(args_json)
                        except Exception:
                            args = {}
                        if name == "Terminate":
                            final_answer = args.get("ans", "")
                            msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": name,
                                "content": json.dumps({"ans": final_answer}, ensure_ascii=False),
                            })
                            transcript.append({"tool": name, "args": args, "result": {"ans": final_answer}})
                            yield {"event": "final_answer", "answer": final_answer}
                            break

            yield {"event": "done"}
        except Exception as e:
            try:
                yield {"event": "error", "message": str(e)}
            finally:
                return
