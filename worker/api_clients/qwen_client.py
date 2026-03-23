import aiohttp
import asyncio
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from config import DASHSCOPE_API_KEY
from worker.api_clients.vlm_utils import parse_coordinates, parse_bbox


class QwenClient:
    def __init__(self):
        self.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = "qwen3.5-plus"
        self.api_key = DASHSCOPE_API_KEY

    async def locate_object(self, image: np.ndarray, text: str, max_retries: int = 3, return_bbox: bool = False) -> list[int]:
        """Locate object in image and return center point [x, y] or bbox [x1, y1, x2, y2]."""
        img_pil = Image.fromarray(image)
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{img_b64}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": f"Find the {text} in this image. Return ONLY a JSON array with the bounding box in this exact format:\n```json\n[{{\n  \"bbox_2d\": [x1, y1, x2, y2],\n  \"label\": \"{text}\"\n}}]\n```\nUse 1000x1000 coordinate system where (0,0) is top-left."}
                        ]
                    }]
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        if return_bbox:
                            return parse_bbox(content, image.shape)
                        return parse_coordinates(content, image.shape)

            except Exception as e:
                print(f"Warning: VLM parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed after {max_retries} attempts: {str(e)}")
                await asyncio.sleep(1)
