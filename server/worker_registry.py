import asyncio
import time
import aiohttp
from typing import Dict, Optional

class WorkerRegistry:
    def __init__(self, timeout: int = 60):
        self.workers: Dict[str, dict] = {}
        self.lock = asyncio.Lock()
        self.timeout = timeout

    async def register(self, worker_id: str, name: str, host: str, port: int, capabilities: dict) -> dict:
        async with self.lock:
            self.workers[worker_id] = {
                "id": worker_id,
                "name": name,
                "host": host,
                "port": port,
                "capabilities": capabilities,
                "last_seen": time.time()
            }
            return self.workers[worker_id]

    async def heartbeat(self, worker_id: str) -> bool:
        async with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id]["last_seen"] = time.time()
                return True
            return False

    async def list_workers(self) -> list:
        async with self.lock:
            return list(self.workers.values())

    async def get_worker(self, worker_id: str) -> Optional[dict]:
        async with self.lock:
            return self.workers.get(worker_id)

    async def ping_workers(self):
        async with self.lock:
            workers_to_check = list(self.workers.items())

        for wid, worker in workers_to_check:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{worker['host']}:{worker['port']}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            async with self.lock:
                                if wid in self.workers:
                                    self.workers[wid]["last_seen"] = time.time()
            except:
                pass

    async def cleanup_stale_workers(self):
        async with self.lock:
            current_time = time.time()
            stale = [wid for wid, w in self.workers.items() if current_time - w["last_seen"] > self.timeout]
            for wid in stale:
                del self.workers[wid]
