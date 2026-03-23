import asyncio
import uuid
import aiohttp
from typing import Dict, Optional
from server.image_utils import encode_image, decode_image

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, dict] = {}
        self.lock = asyncio.Lock()

    async def create_task(self, image: str, text: str, worker_id: str) -> str:
        task_id = str(uuid.uuid4())
        async with self.lock:
            self.tasks[task_id] = {
                "id": task_id,
                "status": "pending",
                "image": image,
                "text": text,
                "worker_id": worker_id,
                "mask": None,
                "point": None,
                "bbox": None,
                "inference_time": None,
                "timings": None,
                "error": None
            }
        return task_id

    async def get_task(self, task_id: str) -> Optional[dict]:
        async with self.lock:
            return self.tasks.get(task_id)

    async def execute_task(self, task_id: str, worker_url: str):
        task = await self.get_task(task_id)
        if not task:
            return

        async with self.lock:
            self.tasks[task_id]["status"] = "processing"

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"image": task["image"], "text": task["text"]}
                async with session.post(f"{worker_url}/process", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        async with self.lock:
                            self.tasks[task_id]["mask"] = result.get("mask")
                            self.tasks[task_id]["point"] = result.get("point")
                            self.tasks[task_id]["bbox"] = result.get("bbox")
                            self.tasks[task_id]["inference_time"] = result.get("inference_time")
                            self.tasks[task_id]["timings"] = result.get("timings")
                            self.tasks[task_id]["status"] = "completed"
                    else:
                        async with self.lock:
                            self.tasks[task_id]["status"] = "failed"
                            self.tasks[task_id]["error"] = f"Worker returned {resp.status}"
        except Exception as e:
            async with self.lock:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = str(e)
