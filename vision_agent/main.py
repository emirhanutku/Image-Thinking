import os
import json
from datasets import load_dataset
from PIL import Image

from openai import OpenAI
from vision_agent.config import DATASET, N_FEWSHOTS,MAX_EXAMPLES_SCAN
from vision_agent.dataset import OpenThinkDataset, Example
from vision_agent.planner import VisionPlanner
from .executor import VisionExecutor
from vision_agent.utils.dataset_util import fetch_examples, yield_examples
from datasets import get_dataset_split_names
from vision_agent.tools.point_tool_qwen import qwen_point_2b 

from vision_agent.tools.point_tool import mark_pixel

from vision_agent.point_sft_builder import PointSFTDatasetBuilder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    """in_file  = "/Users/emirhan/Desktop/Jotform Projects/Visual Reasoning/vision_agent/outputs/OpenThinkIMG-Chart-SFT-2942_train_point.jsonl"   # giriş JSONL
    out_file = "/Users/emirhan/Desktop/Jotform Projects/Visual Reasoning/vision_agent/outputs/OpenThinkIMG-Chart-SFT-2942_train_point_fixed.jsonl"  # çıkış JSONL

    local_prefix = "/Users/emirhan/Desktop/Jotform Projects/Visual Reasoning/vision_agent/outputs/images"
    drive_prefix = "/content/drive/MyDrive/point_data/images"

    with open(in_file, "r") as fin, open(out_file, "w") as fout:
        for line in fin:
            rec = json.loads(line)

            # conversations içindeki images
            if "conversations" in rec and rec["conversations"]:
                for conv in rec["conversations"]:
                    if "images" in conv and conv["images"]:
                        conv["images"] = [
                            p.replace(local_prefix, drive_prefix, 1) for p in conv["images"]
                        ]

            # meta içindeki image_path
            if "meta" in rec and "image_path" in rec["meta"]:
                rec["meta"]["image_path"] = rec["meta"]["image_path"].replace(local_prefix, drive_prefix, 1)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")"""

    """# --- Config ---
    hf_dataset = "hitsmy/OpenThinkIMG-Chart-SFT-2942"
    hf_split   = "train"           
    out_dir    = "/Users/emirhan/Desktop/Jotform Projects/Visual Reasoning/vision_agent/outputs"
    os.makedirs(out_dir, exist_ok=True)

    dump_images_dir = os.path.join(out_dir, "images")
    os.makedirs(dump_images_dir, exist_ok=True)

    # --- Build the dataset ---
    builder = PointSFTDatasetBuilder(
        normalize=True,             
        images_root=None,           
        strict_match=False,         
        consistency_radius=None,    
        bbox_radius=0.01,           
        default_wh=None,            
        dump_images_dir=dump_images_dir,  
        prefer_saved_paths=True,
        sequential_naming=True,     
        sequential_prefix="sample" 
    )

    print(f"Loading HF dataset: {hf_dataset} [{hf_split}] …")
    builder.load_from_hf(hf_dataset, split=hf_split)

    print("Building SFT records …")
    sft_records, bbox_records, stats = builder.build(also_bbox=True)

    # --- Save outputs ---
    sft_path  = os.path.join(out_dir, f"{hf_dataset.split('/')[-1]}_{hf_split}_point.jsonl")
    bbox_path = os.path.join(out_dir, f"{hf_dataset.split('/')[-1]}_{hf_split}_point_bbox.jsonl")

    builder.save_jsonl(sft_path, sft_records)
    builder.save_jsonl(bbox_path, bbox_records)

    # Small preview for manual inspection
    preview_path = os.path.join(out_dir, "preview_first5.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(sft_records[:5], f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"SFT JSONL:    {sft_path}")
    print(f"BBox JSONL:   {bbox_path}")
    print(f"Images saved: {dump_images_dir}")
    print(f"Preview:      {preview_path}")
    pil = Image.open("/Users/emirhan/Desktop/Jotform Projects/Visual Reasoning/vision_agent/outputs/images/sample_1.png").convert("RGB")
    out = mark_pixel(pil, x=24.0, y=67.7, radius=5, color=(255,0,0))
    out.show()"""


    """examples = fetch_examples(
        dataset_name="hitsmy/OpenThinkIMG-Chart-Test-994",
        split="train",  
        limit=3
    )
    for ex in examples:
        question = ex['question']
        image_path = ex['image']"""
    

    image_path = "/Users/emirhan/Downloads/image.png"  # <--  test image
    question   = "What's the median value of green graph from 2013 to 2015?"  # <-- sample question


    
    hf = load_dataset(DATASET,split="train")
    ds = OpenThinkDataset(rows=hf, seed=42)
    examples = ds.sample(k=N_FEWSHOTS, scan_limit=MAX_EXAMPLES_SCAN)


    client = OpenAI()  
    executor = VisionExecutor(client)

    # run execution loop
    result = executor.run(
        image_path=image_path,
        question=question,
        examples=examples,
        latency=False,

    )

    # print result nicely
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
