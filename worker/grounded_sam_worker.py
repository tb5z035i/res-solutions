import argparse
import time
from worker.base_worker import BaseWorker
from worker.api_clients.grounded_sam_client import GroundedSAMClient
from server.image_utils import decode_image, encode_image


class GroundedSamWorker(BaseWorker):
    def __init__(self, name: str, port: int):
        super().__init__(name, port)
        self.grounded_sam_client = GroundedSAMClient()

    async def segment(self, image: str, text: str) -> dict:
        """Segment image using Grounded SAM 2 (Grounding DINO + SAM2)."""
        start_time = time.time()
        timings = {}

        step_start = time.time()
        img_array = decode_image(image)
        timings["decode_image"] = time.time() - step_start

        step_start = time.time()
        mask, bboxes = self.grounded_sam_client.segment(img_array, text)
        timings["grounded_sam"] = time.time() - step_start

        step_start = time.time()
        mask_encoded = encode_image(mask)
        timings["encode_mask"] = time.time() - step_start

        inference_time = time.time() - start_time

        # Return first bbox if available
        bbox = bboxes[0] if bboxes else None

        return {"mask": mask_encoded, "bbox": bbox, "inference_time": inference_time, "timings": timings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="grounded-sam-worker")
    parser.add_argument("--server-url", default="http://localhost:7000")
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()

    worker = GroundedSamWorker(args.name, args.port)
    worker.start(args.server_url)
