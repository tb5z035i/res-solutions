import aiohttp
import asyncio
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from worker.api_clients.vlm_utils import parse_coordinates


class OllamaClient:
    def __init__(self):
        self.endpoint = f"{OLLAMA_BASE_URL}/api/chat"
        self.model = OLLAMA_MODEL

    async def locate_object(self, image: np.ndarray, text: str, max_retries: int = 3) -> list[int]:
        """Locate object in image and return center point [x, y]."""
        img_pil = Image.fromarray(image)
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": f"Find the {text} in this image. Return ONLY a JSON array with the bounding box in this exact format:\n```json\n[{{\n  \"bbox_2d\": [x1, y1, x2, y2],\n  \"label\": \"{text}\"\n}}]\n```\nUse 1000x1000 coordinate system where (0,0) is top-left.",
                        "images": [img_b64]
                    }],
                    "stream": False
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(self.endpoint, json=payload) as resp:
                        result = await resp.json()
                        content = result["message"]["content"]
                        return parse_coordinates(content, image.shape)

            except Exception as e:
                print(f"Warning: VLM parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed after {max_retries} attempts: {str(e)}")
                await asyncio.sleep(1)
