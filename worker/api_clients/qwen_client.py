import aiohttp
import base64
import json
import re
from io import BytesIO
from PIL import Image
import numpy as np
from config import DASHSCOPE_API_KEY


class QwenClient:
    def __init__(self):
        self.endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = "qwen3.5-plus"
        self.api_key = DASHSCOPE_API_KEY

    async def locate_object(self, image: np.ndarray, text: str) -> list[int]:
        """Locate object in image and return center point [x, y]."""
        img_pil = Image.fromarray(image)
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{img_b64}"

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": f"{text}. Return the center point coordinates of the object in format (x, y)."}
                ]
            }]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                result = await resp.json()
                content = result["choices"][0]["message"]["content"]

                # Parse coordinates from response
                match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', content)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    # Convert normalized to pixel if needed
                    if x <= 1.0 and y <= 1.0:
                        h, w = image.shape[:2]
                        x, y = int(x * w), int(y * h)
                    return [int(x), int(y)]

                raise ValueError(f"Could not parse coordinates from: {content}")
