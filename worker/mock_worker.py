import numpy as np
from scipy.ndimage import gaussian_filter
from worker.base_worker import BaseWorker
from server.image_utils import decode_image, encode_image

class MockWorker(BaseWorker):
    async def segment(self, image: str, text: str) -> str:
        img = decode_image(image)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        text_lower = text.lower()
        if "left" in text_lower:
            mask[:, :w//2] = 255
        elif "right" in text_lower:
            mask[:, w//2:] = 255
        elif "top" in text_lower:
            mask[:h//2, :] = 255
        elif "bottom" in text_lower:
            mask[h//2:, :] = 255
        elif "center" in text_lower:
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            mask_circle = ((x - cx)**2 + (y - cy)**2) <= (min(h, w) // 4)**2
            mask[mask_circle] = 255
        else:
            mask[:, :w//2] = 255

        mask = gaussian_filter(mask.astype(float), sigma=20)
        mask = (mask / mask.max() * 255).astype(np.uint8) if mask.max() > 0 else mask.astype(np.uint8)

        return encode_image(mask)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mock-worker-1")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    worker = MockWorker(args.name, args.port)
    worker.start(args.server_url)
