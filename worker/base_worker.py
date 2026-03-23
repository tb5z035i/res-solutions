import asyncio
import uuid
from abc import ABC, abstractmethod
from fastapi import FastAPI
import aiohttp
import uvicorn
from pydantic import BaseModel

class ProcessRequest(BaseModel):
    image: str
    text: str

class BaseWorker(ABC):
    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.worker_id = str(uuid.uuid4())
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/process")
        async def process(req: ProcessRequest):
            mask = await self.segment(req.image, req.text)
            return {"mask": mask}

        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

    @abstractmethod
    async def segment(self, image: str, text: str) -> str:
        pass

    async def register_with_server(self, server_url: str):
        async with aiohttp.ClientSession() as session:
            payload = {
                "worker_id": self.worker_id,
                "name": self.name,
                "host": "0.0.0.0",
                "port": self.port,
                "capabilities": {}
            }
            await session.post(f"{server_url}/api/workers/register", json=payload)

    def start(self, server_url: str):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.register_with_server(server_url))
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)
