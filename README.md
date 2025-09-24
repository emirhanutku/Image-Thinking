
# Image Thinking — Visual Reasoning Agent

A fast, tool-augmented visual reasoning system that answers questions about images (charts, figures, UI shots, etc.). It orchestrates multiple vision tools (OCR, pointing, segmentation, zoom, guide-lines) with an LLM to plan multi-step solutions and return visual + textual answers.

- Web UI (FastAPI + static front-end)
- Tool calling via OpenAI Chat Completions
- OCR via PaddleOCR
- Pointing via Qwen2‑VL (with optional LoRA)
- Segmentation via SAM2 (Transformers)
- Guide lines (horizontal/vertical) to visually explain answers
- Few-shot prompting from a small curated dataset

## Quickstart

1) Clone (with submodules)
```bash
git clone --recurse-submodules https://github.com/emirhanutku/Image-Thinking.git
cd Image-Thinking
# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

2) Python env
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

3) Install dependencies (CPU-friendly baseline)
```bash
pip install fastapi uvicorn[pstandard] pillow numpy opencv-python datasets \
            openai transformers peft qwen-vl-utils rapidfuzz

# PyTorch (CPU default on macOS; pick your platform-specific wheel if needed)
pip install torch

# OCR (PaddleOCR + PaddlePaddle). CPU install (works on most machines):
pip install paddleocr paddlepaddle
# If PaddlePaddle fails on your platform, see:
# https://www.paddlepaddle.org.cn/en/install/quick
```

4) Environment variables
```bash
# OpenAI (required)
export OPENAI_API_KEY="YOUR_KEY"

# Optional: Apple Silicon stability
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Qwen2-VL LoRA for the Point tool (optional but expected by default code)
# Point tool loads a PEFT adapter from this folder. Change this to your adapter.
export QWEN_ADAPTER_DIR="/path/to/your/qwen2vl_lora_checkpoint"
# Or to use base model only, edit vision_agent/tools/point_tool_qwen.py accordingly.

# Optional: change base model
# export QWEN_BASE_ID="Qwen/Qwen2-VL-2B-Instruct"
```

5) Run the server
```bash
# From repo root
uvicorn app.server:app --reload
# or
python server.py
```

6) Open the web UI
- http://localhost:8000

Upload an image, ask a question (e.g., “Mark the point for 2019 and draw a vertical guideline”), and watch the model reason with tools.

## API

- POST /api/ask
  - Form fields: `question` (text), `image` (file)
  - Response: `{ success, final_answer, transcript, images }`

Example:
```bash
curl -X POST http://localhost:8000/api/ask \
  -F "question=Mark the point for 2019 and draw a vertical line." \
  -F "image=@/path/to/chart.png"
```

- POST /api/ask_stream
  - Same inputs as `/api/ask`
  - Returns NDJSON stream of events: `start`, `tool_started`, `tool_result`, `final_answer`, `done`

## Features and Tools

The agent plans tool calls and executes them step-by-step until it can “Terminate” with an answer.

- OCR
  - File: `vision_agent/tools/ocr_tool.py`
  - PaddleOCR extracts text (axes, labels, annotations).
- Point
  - File: `vision_agent/tools/point_tool.py`
  - Uses Qwen2‑VL (vision-language model) + heuristics to locate a target point; can read numeric value on charts when possible.
  - Qwen integration: `vision_agent/tools/point_tool_qwen.py` (PEFT LoRA expected via `QWEN_ADAPTER_DIR`).
- ZoomInSubfigure
  - File: `vision_agent/tools/zoom_in_subfigure_tool.py`
  - OCR anchors + LLM to crop a relevant subfigure/panel.
  - Default model name in code is “gpt-5-2025-08-07”; change to a model you have access to.
- SegmentRegionAroundPoint
  - File: `vision_agent/tools/segment_region_around_point_tool.py`
  - Segments around an (x,y) with SAM2 via Transformers (`facebook/sam2.1-hiera-large`).
- DrawHorizontalLineByY / DrawVerticalLineByX
  - Files: `vision_agent/tools/draw_horizontal_line.py`, `vision_agent/tools/draw_vertical_line.py`
  - Draw dashed guide lines to visualize reasoning.

All tool interfaces are declared in: `vision_agent/tools_spec.py`

## How It Works

- Backend: FastAPI (`server.py`)
  - Serves static UI from `front-end/`
  - Endpoints: `/api/ask` and `/api/ask_stream`
  - Preloads few-shot exemplars from the dataset `hitsmy/OpenThinkIMG-Chart-SFT-2942` on startup
- Agent: `vision_agent/executor.py`
  - Builds the prompt (`vision_agent/prompt.py`)
  - Calls OpenAI Chat Completions with tool specs (`vision_agent/tools_spec.py`)
  - Executes Python implementations and feeds outputs back to the model
  - Emits new images via an in-memory `ImageStore` and exposes them to the front-end
- Front-end: `front-end/` (vanilla HTML/CSS/JS)
  - Upload image, ask question, see final answer, transcript, and image outputs

## Repo Structure

- `server.py:8000` — FastAPI app, web + API
- `front-end/` — static assets for the UI
- `vision_agent/executor.py` — tool-using agent loop
- `vision_agent/prompt.py`, `vision_agent/tools_spec.py` — prompt and tool schemas
- `vision_agent/runtime_tools.py` — tool adaptors + image store
- `vision_agent/tools/*.py` — tool implementations
- `vision_agent/dataset.py`, `vision_agent/config.py` — few-shot sampling config
- `GroundingDINO/`, `sam2_repo/` — git submodules (research/reference). Not required at runtime; segmentation uses Transformers SAM2.

## Configuration

- LLM selection: `vision_agent/config.py` (`MODEL_NAME`, `TEMPERATURE`)
- Zoom model: `vision_agent/tools/zoom_in_subfigure_tool.py` (`llm_model` argument)
- SAM2 model ID: `vision_agent/tools/segment_region_around_point_tool.py` (default `facebook/sam2.1-hiera-large`)
- Few-shot dataset: `vision_agent/config.py` (`DATASET`, `N_FEWSHOTS`, `MAX_EXAMPLES_SCAN`, `INDEXES_WE_WANT`)

## Troubleshooting

- “OPENAI_API_KEY not set”
  - Export your key: `export OPENAI_API_KEY=...`
- Qwen Point tool fails to load LoRA
  - Set `QWEN_ADAPTER_DIR` to your LoRA path or edit `point_tool_qwen.py` to use the base model only.
- PaddleOCR/PaddlePaddle install issues
  - Try CPU package: `pip install paddleocr paddlepaddle`, or follow Paddle’s platform-specific install docs.
- SAM2 model download errors
  - Ensure internet access and recent `transformers`. Try `pip install --upgrade transformers`.
- Submodules missing
  - `git submodule update --init --recursive`
- Port already in use
  - Run `uvicorn server:app --port 8001 --reload` and open http://localhost:8001

## Acknowledgments

- OpenAI API for tool-augmented reasoning
- PaddleOCR for robust OCR
- Qwen2‑VL for point localization
- SAM2 (Transformers) for high-quality segmentation

Enjoy Image Thinking!
