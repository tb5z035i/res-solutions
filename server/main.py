import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr
from server.worker_registry import WorkerRegistry
from server.task_manager import TaskManager
from server.image_utils import encode_image, decode_image, blend_mask

registry = WorkerRegistry()
task_manager = TaskManager()

async def cleanup_loop():
    while True:
        await asyncio.sleep(30)
        await registry.ping_workers()
        await registry.cleanup_stale_workers()

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class WorkerRegisterRequest(BaseModel):
    worker_id: str
    name: str
    host: str
    port: int
    capabilities: dict = {}

class SegmentRequest(BaseModel):
    image: str
    text: str
    worker_id: str

@app.post("/api/workers/register")
async def register_worker(req: WorkerRegisterRequest):
    worker = await registry.register(req.worker_id, req.name, req.host, req.port, req.capabilities)
    return {"status": "registered", "worker": worker}

@app.post("/api/workers/{worker_id}/heartbeat")
async def heartbeat(worker_id: str):
    success = await registry.heartbeat(worker_id)
    if not success:
        raise HTTPException(404, "Worker not found")
    return {"status": "ok"}

@app.get("/api/workers")
async def list_workers():
    workers = await registry.list_workers()
    return {"workers": workers}

@app.post("/api/segment")
async def segment(req: SegmentRequest):
    worker = await registry.get_worker(req.worker_id)
    if not worker:
        raise HTTPException(404, "Worker not found")

    task_id = await task_manager.create_task(req.image, req.text, req.worker_id)
    worker_url = f"http://{worker['host']}:{worker['port']}"
    asyncio.create_task(task_manager.execute_task(task_id, worker_url))
    return {"task_id": task_id}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task

async def process_image(image, text, worker_name):
    if image is None or not text or not worker_name:
        return None

    workers = await registry.list_workers()
    worker = next((w for w in workers if w["name"] == worker_name), None)
    if not worker:
        return None

    img_encoded = encode_image(image)
    task_id = await task_manager.create_task(img_encoded, text, worker["id"])
    worker_url = f"http://{worker['host']}:{worker['port']}"
    asyncio.create_task(task_manager.execute_task(task_id, worker_url))

    for _ in range(60):
        await asyncio.sleep(0.5)
        task = await task_manager.get_task(task_id)
        if task["status"] == "completed":
            mask = decode_image(task["mask"])
            img_array = decode_image(img_encoded)
            blended = blend_mask(img_array, mask)
            return blended
        elif task["status"] == "failed":
            return None
    return None

def get_worker_names():
    import asyncio
    workers = asyncio.run(registry.list_workers())
    return [w["name"] for w in workers]

with gr.Blocks() as demo:
    gr.Markdown("# RES Pipeline Testing")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image")
            text_input = gr.Textbox(label="Text Description", placeholder="e.g., object on the left")
            worker_dropdown = gr.Dropdown(label="Select Worker", choices=[], allow_custom_value=True)
            refresh_btn = gr.Button("🔄 Refresh Workers")
            submit_btn = gr.Button("Process", variant="primary")
        with gr.Column():
            result_output = gr.Image(label="Result")

    def refresh_workers():
        workers = asyncio.run(registry.list_workers())
        names = [w["name"] for w in workers]
        return gr.Dropdown(choices=names, value=names[0] if names else None)

    # Manual refresh button
    refresh_btn.click(refresh_workers, None, worker_dropdown)

    # Auto-poll every 1 second
    timer = gr.Timer(1)
    timer.tick(refresh_workers, None, worker_dropdown)

    submit_btn.click(process_image, [image_input, text_input, worker_dropdown], result_output)

app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
