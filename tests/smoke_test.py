import pytest
import asyncio
import httpx
import numpy as np
from server.image_utils import encode_image

BASE_URL = "http://localhost:8000"

@pytest.mark.asyncio
async def test_worker_registration():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/api/workers")
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert len(workers) > 0

@pytest.mark.asyncio
async def test_segment_task():
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_b64 = encode_image(test_img)

    async with httpx.AsyncClient() as client:
        workers_resp = await client.get(f"{BASE_URL}/api/workers")
        workers = workers_resp.json()["workers"]
        assert len(workers) > 0
        worker_id = workers[0]["id"]

        segment_resp = await client.post(f"{BASE_URL}/api/segment", json={
            "image": img_b64,
            "text": "object on the left",
            "worker_id": worker_id
        })
        assert segment_resp.status_code == 200
        task_id = segment_resp.json()["task_id"]

        for _ in range(20):
            await asyncio.sleep(0.5)
            task_resp = await client.get(f"{BASE_URL}/api/tasks/{task_id}")
            task = task_resp.json()
            if task["status"] == "completed":
                assert task["mask"] is not None
                return

        pytest.fail("Task did not complete in time")
