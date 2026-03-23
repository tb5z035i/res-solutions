import argparse
from worker.base_worker import BaseWorker
from worker.api_clients.qwen_client import QwenClient
from worker.api_clients.sam_client import SAMClient
from server.image_utils import decode_image, encode_image


class QwenSamWorker(BaseWorker):
    def __init__(self, name: str, port: int):
        super().__init__(name, port)
        self.qwen_client = QwenClient()
        self.sam_client = SAMClient()

    async def segment(self, image: str, text: str) -> str:
        """Segment image using Qwen3-VL-Plus + SAM3."""
        img_array = decode_image(image)

        point = await self.qwen_client.locate_object(img_array, text)

        mask = self.sam_client.segment(img_array, point)

        return encode_image(mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="qwen-sam-worker")
    parser.add_argument("--server-url", default="http://localhost:7000")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    worker = QwenSamWorker(args.name, args.port)
    worker.start(args.server_url)
