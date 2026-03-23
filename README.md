# RES Pipeline Testing Platform

Web-based testing platform for RES (Referring Expression Segmentation) pipelines with dynamic worker registration.

## Quick Start

### Installation

Install base dependencies (server only):
```bash
pip install -e .
```

Install with specific worker support:
```bash
# SAM2 workers (qwen-sam, ollama-sam)
pip install -e ".[sam]"

# Qwen API worker
pip install -e ".[sam,qwen]"

# All workers
pip install -e ".[all]"

# Development
pip install -e ".[all,dev]"
```

### Running

### Running

```bash
# Terminal 1: Start server
python -m server.main

# Terminal 2: Start a worker
python -m worker.mock_worker --name mock-worker-1 --port 8001
# OR
python -m worker.qwen_sam_worker --name qwen-sam-worker --port 8002
# OR
python -m worker.ollama_sam_worker --name ollama-sam-worker --port 8003
# OR
python -m worker.qwen_sam_bbox_worker --name qwen-sam-bbox-worker --port 8004

# Access UI at http://localhost:7000/ui
```

## Architecture

- **Server** (port 7000): FastAPI + Gradio UI, manages workers and tasks
- **Workers** (ports 8001+): Process segmentation requests independently
- **Communication**: REST API with base64-encoded images

## Available Workers

- **mock-worker**: Generates geometric masks based on keywords (no dependencies)
- **qwen-sam-worker**: Qwen3.5-Plus API + SAM2 with point prompts (requires `sam,qwen`)
- **qwen-sam-bbox-worker**: Qwen3.5-Plus API + SAM2 with bbox prompts (requires `sam,qwen`)
- **ollama-sam-worker**: Local Ollama Qwen3-VL + SAM2 (requires `sam`)

## API Endpoints

### Server
- `POST /api/workers/register` - Register worker
- `POST /api/workers/{id}/heartbeat` - Keep-alive (30s interval)
- `GET /api/workers` - List active workers
- `POST /api/segment` - Submit task
- `GET /api/tasks/{id}` - Get task status/result

### Worker
- `POST /process` - Process image+text, return mask
- `GET /health` - Health check

## Testing

```bash
# Run tests (requires server + worker running)
pytest tests/smoke_test.py -v
```

## Mock Worker Behavior

Generates geometric masks based on text keywords:
- "left" → left half
- "right" → right half
- "center" → circular mask
- "top"/"bottom" → horizontal halves

Results shown with red alpha-blended overlay.
