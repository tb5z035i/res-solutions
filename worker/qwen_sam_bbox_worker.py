import argparse
import time
from worker.base_worker import BaseWorker
from worker.api_clients.qwen_client import QwenClient
from worker.api_clients.sam_client import SAMClient
from server.image_utils import decode_image, encode_image


class QwenSamBboxWorker(BaseWorker):
    def __init__(self, name: str, port: int):
        super().__init__(name, port)
        self.qwen_client = QwenClient()
        self.sam_client = SAMClient()

    async def segment(self, image: str, text: str) -> dict:
        """Segment image using Qwen3-VL-Plus bbox + SAM3."""
        start_time = time.time()
        timings = {}

        step_start = time.time()
        img_array = decode_image(image)
        timings["decode_image"] = time.time() - step_start

        step_start = time.time()
        bbox = await self.qwen_client.locate_object(img_array, text, return_bbox=True)
        timings["qwen_locate"] = time.time() - step_start

        step_start = time.time()
        mask = self.sam_client.segment(img_array, bbox=bbox)
        timings["sam_segment"] = time.time() - step_start

        step_start = time.time()
        mask_encoded = encode_image(mask)
        timings["encode_mask"] = time.time() - step_start

        inference_time = time.time() - start_time

        return {"mask": mask_encoded, "bbox": bbox, "inference_time": inference_time, "timings": timings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="qwen-sam-bbox-worker")
    parser.add_argument("--server-url", default="http://localhost:7000")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()

    worker = QwenSamBboxWorker(args.name, args.port)
    worker.start(args.server_url)
