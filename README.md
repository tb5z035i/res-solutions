# RES Pipeline Testing Platform

Web-based testing platform for RES (Referring Expression Segmentation) pipelines with dynamic worker registration.

## Quick Start

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Terminal 1: Start server
python -m server.main

# Terminal 2: Start mock worker
python -m worker.mock_worker --name mock-worker-1 --port 8001

# Access UI at http://localhost:8000
```

## Architecture

- **Server** (port 8000): FastAPI + Gradio UI, manages workers and tasks
- **Workers** (ports 8001+): Process segmentation requests independently
- **Communication**: REST API with base64-encoded images

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
