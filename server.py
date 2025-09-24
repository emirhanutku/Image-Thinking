from __future__ import annotations
import io
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Import project executor
from vision_agent.executor import VisionExecutor
from datasets import load_dataset
from vision_agent.dataset import OpenThinkDataset
from vision_agent.config import DATASET, N_FEWSHOTS, MAX_EXAMPLES_SCAN


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        hf = load_dataset(DATASET, split="train")
        ds = OpenThinkDataset(rows=hf, seed=42)
        examples = ds.sample(k=N_FEWSHOTS, scan_limit=MAX_EXAMPLES_SCAN)
        app.state.examples = examples
        print(f"[lifespan] Preloaded {len(examples)} few-shot examples from {DATASET}.")
    except Exception as e:
        app.state.examples = []
        print(f"[lifespan] Failed to preload examples: {e}")

    yield

    try:
        app.state.examples = []
        print("[lifespan] Cleared preloaded examples on shutdown.")
    except Exception:
        pass



app = FastAPI(title="Visual Reasoning Web API", lifespan=lifespan)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="front-end"), name="static")


@app.get("/")
def index():
    return FileResponse("front-end/index.html")


@app.post("/api/ask")
async def ask(question: str = Form(...), image: UploadFile = File(...)):
    try:

        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")


        examples = getattr(app.state, "examples", [])
        executor = VisionExecutor()
        result = executor.run(image_path=pil, question=question, examples=examples, latency=False)


        images: List[Dict[str, Any]] = []
        try:

            store = executor.image_store
            for image_id, ref in getattr(store, "_images", {}).items():
                images.append({"id": image_id, "url": ref.url})
        except Exception:
            pass

        return JSONResponse({
            "success": True,
            "final_answer": result.get("final_answer"),
            "transcript": result.get("transcript", []),
            "images": images,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        )


def run(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


@app.post("/api/ask_stream")
async def ask_stream(question: str = Form(...), image: UploadFile = File(...)):
    try:
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")

        examples = getattr(app.state, "examples", [])

        executor = VisionExecutor()

        def iter_events():
            import json as _json
            try:
                for evt in executor.run_stream(image_path=pil, question=question, examples=examples, latency=False):
                    # Ensure serializable content
                    try:
                        line = _json.dumps(evt, ensure_ascii=False)
                    except TypeError:
                        # Fallback: convert non-serializable values
                        sanitized = {
                            k: (v if isinstance(v, (str, int, float, bool, type(None), dict, list)) else str(v))
                            for k, v in evt.items()
                        }
                        line = _json.dumps(sanitized, ensure_ascii=False)
                    yield line + "\n"
            except Exception as e:
                err = {"event": "error", "message": str(e)}
                yield _json.dumps(err, ensure_ascii=False) + "\n"

        return StreamingResponse(iter_events(), media_type="application/x-ndjson")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    run()
