import os
import sys
import time
import math
import re
import argparse
from typing import List, Tuple, Optional

from datasets import load_dataset
from PIL import Image

# Local project imports (do not modify project files)
from vision_agent.config import DATASET, N_FEWSHOTS, MAX_EXAMPLES_SCAN
from vision_agent.dataset import OpenThinkDataset
from vision_agent.executor import VisionExecutor
from openai import OpenAI



def _load_embedder(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_numbers(s: str) -> List[float]:
    nums: List[float] = []
    for m in _NUM_RE.finditer(s):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            continue
    return nums


def _numeric_match(pred: str, label: str, rel_tol: float = 0.05, abs_tol: float = 1e-6) -> bool:
    p_nums = _extract_numbers(pred)
    l_nums = _extract_numbers(label)
    if not p_nums or not l_nums:
        return False
    if len(p_nums) != len(l_nums):
        return False

    remaining = l_nums.copy()
    for x in p_nums:
        found_idx: Optional[int] = None
        for j, y in enumerate(remaining):
            if math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
                found_idx = j
                break
        if found_idx is None:
            return False
        remaining.pop(found_idx)
    return True


def _semantic_match(embedder, pred: str, label: str, threshold: float) -> Tuple[float, bool]:
    def norm(t: str) -> str:
        return (t or "").strip()

    a, b = norm(pred), norm(label) #a = my predicton , b = ground truth
    if not a and not b:
        return 1.0, True
    if not a or not b:
        return 0.0, False

    embs = embedder.encode([a, b], normalize_embeddings=True)
    sim = float((embs[0] * embs[1]).sum())
    return sim, bool(sim >= threshold)


def evaluate(limit: int = 50, threshold: float = 0.50, use_numeric_first: bool = True):

    print(f"Loading test dataset: hitsmy/OpenThinkIMG-Chart-Test-994 [train], limit={limit}")
    test_ds = load_dataset("hitsmy/OpenThinkIMG-Chart-Test-994", split="train")
    n = min(limit, len(test_ds))

    # 2) Build few-shot exemplars from SFT dataset once (reused for all eval samples)
    print(f"Loading few-shots from {DATASET} …")
    sft_rows = load_dataset(DATASET, split="train")
    fewshot_ds = OpenThinkDataset(rows=sft_rows, seed=42)
    exemplars = fewshot_ds.sample(k=N_FEWSHOTS, scan_limit=MAX_EXAMPLES_SCAN)
    print(f"Few-shots prepared: {len(exemplars)}")

    # 3) Init model executor
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. The executor will fail without it.")
    client = OpenAI()
    executor = VisionExecutor(client)

    # 4) Load metric model
    print("Loading evaluation embedder: sentence-transformers/all-mpnet-base-v2 …")
    embedder = _load_embedder("sentence-transformers/all-mpnet-base-v2")

    total = 0
    correct = 0

    per_item = []
    t0 = time.time()

    for i in range(n):
        rec = test_ds[i]
        image: Image.Image = rec["image"].convert("RGB") if isinstance(rec["image"], Image.Image) else rec["image"]
        question: str = rec["question"]
        label: str = str(rec["label"])  

        try:
            result = executor.run(
                image_path=image,  # VisionExecutor accepts PIL.Image
                question=question,
                examples=exemplars,
                latency=False,     # avoid extra translator latency
            )
            pred = str(result.get("final_answer") or "").strip()
        except Exception as e:
            pred = ""
            print(f"[{i:03d}] ERROR during inference: {e}")


        sim = 0.0
        is_ok = False
        if use_numeric_first and _numeric_match(pred, label):
            sim, is_ok = 1.0, True
        else:
            sim, is_ok = _semantic_match(embedder, pred, label, threshold)

        total += 1
        correct += int(is_ok)
        per_item.append((i, question, pred, label, sim, is_ok))

        # Per-sample log
        print("-" * 40)
        print(f"[{i+1:02d}/{n}]\n- Question: {question}\n- Pred: {pred}\n- Label: {label}\n- Similarity: {sim:.3f} | Correct: {is_ok}")
        print("-" * 40)


    dt = time.time() - t0
    acc = (correct / total) if total else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.3%}")
    print(f"Elapsed: {dt:.1f}s")

    return {
        "samples": total,
        "correct": correct,
        "accuracy": acc,
        "elapsed_sec": dt,
        "details": per_item,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate vision agent on OpenThinkIMG-Chart-Test-994")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--threshold", type=float, default=0.50, help="Similarity threshold for correctness")
    parser.add_argument("--no-numeric", action="store_true", help="Disable numeric matching shortcut")
    args = parser.parse_args()

    evaluate(limit=args.limit, threshold=args.threshold, use_numeric_first=not args.no_numeric)


if __name__ == "__main__":
    main()
