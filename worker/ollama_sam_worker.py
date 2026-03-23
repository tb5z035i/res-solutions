import argparse
import time
from worker.base_worker import BaseWorker
from worker.api_clients.ollama_client import OllamaClient
from worker.api_clients.sam_client import SAMClient
from server.image_utils import decode_image, encode_image


class OllamaSamWorker(BaseWorker):
    def __init__(self, name: str, port: int):
        super().__init__(name, port)
        self.ollama_client = OllamaClient()
        self.sam_client = SAMClient()

    async def segment(self, image: str, text: str) -> dict:
        """Segment image using Ollama Qwen3-VL + SAM3."""
        start_time = time.time()
        timings = {}

        step_start = time.time()
        img_array = decode_image(image)
        timings["decode_image"] = time.time() - step_start

        step_start = time.time()
        point = await self.ollama_client.locate_object(img_array, text)
        timings["ollama_locate"] = time.time() - step_start

        step_start = time.time()
        mask = self.sam_client.segment(img_array, point)
        timings["sam_segment"] = time.time() - step_start

        step_start = time.time()
        mask_encoded = encode_image(mask)
        timings["encode_mask"] = time.time() - step_start

        inference_time = time.time() - start_time

        return {"mask": mask_encoded, "point": point, "inference_time": inference_time, "timings": timings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ollama-sam-worker")
    parser.add_argument("--server-url", default="http://localhost:7000")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    worker = OllamaSamWorker(args.name, args.port)
    worker.start(args.server_url)
