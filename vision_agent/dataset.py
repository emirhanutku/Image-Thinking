from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable
import json, random
from .utils.image_io import ensure_pil, pil_to_data_url
from .config import INDEXES_WE_WANT


@dataclass
class Example:
    user_text: str                # ilk soru
    image_data_urls: List[str]    # görsel (data URL)
    assistant_json: str           # {"thought":"...", "actions":[...]} 
    tools_seq: List[str]          # Terminate hariç kullanılan tool adları 
    final_ans: str                # en son görülen Terminate.arguments.ans
    thought: str = ""             # tüm thought'ların kısa birleştirilmiş hali

class OpenThinkDataset:
    def __init__(self, rows: Iterable[Dict[str, Any]], seed: int = 42):
        self.rows = list(rows)
        self.seed = seed

    def sample(self, k: int , scan_limit: Optional[int]) -> List[Example]:
        n = len(self.rows) if scan_limit is None else min(len(self.rows), scan_limit)
        idxs = list(range(n))
        random.Random(self.seed).shuffle(idxs)
        idxs_iwant = INDEXES_WE_WANT
        idxs = idxs_iwant + idxs

        out: List[Example] = []
        for i in idxs_iwant:
            row = self.rows[i]

            ex = self._row_to_example(row)
            if not ex:
                continue
        
            if ex.tools_seq and ex.final_ans and ex.image_data_urls and ex.user_text:
                out.append(ex)
            if len(out) == k:
                break
        return out


    # ---- helpers ----
    @staticmethod
    def _convs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return row.get("conversations", []) or row.get("conversation", []) or []

    @staticmethod
    def _first_human(conv: List[Dict[str, Any]]) -> str:
        for c in conv:
            if c.get("from") in ("human","user"):
                return c.get("value","")
        return ""
    @staticmethod
    def _assistant_msgs(conv: List[Dict[str, Any]]) -> List[str]:
        """All gpt messages in the conversation."""
        msgs = []
        for c in conv:
            if c.get("from") in ("gpt","assistant","model"):
                val = c.get("value","")
                if isinstance(val, str) and val.strip():
                    msgs.append(val)
        return msgs
    
    @staticmethod
    def _parse_actions_from_msg(msg: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse only one assistant message.
        return: (thought, actions)
        """
        try:
            parsed = json.loads(msg)
        except Exception:
            return "", [] 
        

        if not isinstance(parsed, dict):
            return "", []
        
        thought = str(parsed.get("thought",""))
        actions = parsed.get("actions", [])

        if not isinstance(actions, list):
            actions = []

        norm_actions: List[Dict[str, Any]] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            name = a.get("name") or a.get("tool")
            if not name:
                continue
            args = a.get("arguments")
            if args is None:
                args = a.get("input")
            if not isinstance(args, dict):
                args = {"param": args} if args is not None else {}
            norm_actions.append({"name": str(name), "arguments": args})
        return thought, norm_actions

    @staticmethod
    def _merge_all_actions(asst_msgs: List[str]) -> Tuple[str, List[Dict[str, Any]], List[str], str]:
        """
        Return: (merged_thought, merged_actions, tools_seq_wo_terminate, final_ans)
        """
        merged_thoughts: List[str] = []
        merged_actions: List[Dict[str, Any]] = []
        tools_seq: List[str] = []
        final_ans: str = ""

        for msg in asst_msgs:
            t , acts = OpenThinkDataset._parse_actions_from_msg(msg)
            if t:
                merged_thoughts.append(t.strip())
            if acts: 
                merged_actions.extend(acts)
                for a in acts:
                    name = a.get("name")
                    if name == "Terminate":
                        args = a.get("arguments") or {}
                        if isinstance(args,dict) and "ans" in args:
                            final_ans = str(args["ans"])
                    elif name:
                        tools_seq.append(str(name))
        merged_thought = " | ".join(merged_thoughts)
        return merged_thought, merged_actions, tools_seq, final_ans
    

    @staticmethod
    def _image_to_data_url(row: Dict[str, Any]) -> Optional[str]:
        img = row.get("image") or row.get("img")
        if img is None:
            imgs = row.get("images")
            if isinstance(imgs, list) and imgs:
                img = imgs[0]
        if img is None:
            return None
        try:
            pil = ensure_pil(img)
            return pil_to_data_url(pil)
        except Exception:
            return None
    
    @staticmethod
    def _gather_images_from_row(row: Dict[str, Any]) -> List[Any]:
        candidates : List[Any] = []
        v = row.get("image") or row.get("img") or row.get("images")
        if isinstance(v, list):
            candidates.extend(v)
        elif v:
            candidates.append(v)
        return candidates[:4]

    def _row_to_example(self, row: Dict[str, Any]) -> Optional[Example]:
        conv = self._convs(row)
        if not conv:
            return None

        user_text = (self._first_human(conv) or "").strip()
        if not user_text:
            return None
        asst_msgs = self._assistant_msgs(conv)

        merged_thought, merged_actions, tools_seq, final_ans = self._merge_all_actions(asst_msgs)

        raw_images = self._gather_images_from_row(row)
        if raw_images:
            data_urls = [pil_to_data_url(ensure_pil(x)) for x in raw_images]
        else:
            return None
        

        
        assistant_json = json.dumps(
            {"thought": merged_thought, "actions": merged_actions},
            ensure_ascii=False
        )

        return Example(
            user_text=user_text,
            image_data_urls=data_urls,
            assistant_json=assistant_json,
            tools_seq=tools_seq,
            final_ans=final_ans,
            thought=merged_thought
        )


    
