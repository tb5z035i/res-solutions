import aiohttp
import asyncio
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

    async def locate_object(self, image: np.ndarray, text: str, max_retries: int = 3) -> list[int]:
        """Locate object in image and return center point [x, y]."""
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

                        # Try JSON format first
                        json_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group(1))
                            if data:
                                if isinstance(data[0], dict):
                                    if "point_2d" in data[0]:
                                        point = data[0]["point_2d"]
                                        h, w = image.shape[:2]
                                        return [int(point[0] * w / 1000), int(point[1] * h / 1000)]
                                    elif "bbox_2d" in data[0]:
                                        bbox = data[0]["bbox_2d"]
                                        x_center = (bbox[0] + bbox[2]) / 2
                                        y_center = (bbox[1] + bbox[3]) / 2
                                        h, w = image.shape[:2]
                                        return [int(x_center * w / 1000), int(y_center * h / 1000)]
                                elif isinstance(data[0], (int, float)) and len(data) == 4:
                                    # Flat bbox array [x1, y1, x2, y2]
                                    x_center = (data[0] + data[2]) / 2
                                    y_center = (data[1] + data[3]) / 2
                                    h, w = image.shape[:2]
                                    return [int(x_center * w / 1000), int(y_center * h / 1000)]

                        # Try (x, y) format
                        match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', content)
                        if match:
                            x, y = float(match.group(1)), float(match.group(2))
                            if x <= 1.0 and y <= 1.0:
                                h, w = image.shape[:2]
                                x, y = int(x * w), int(y * h)
                            return [int(x), int(y)]

                        raise ValueError(f"Could not parse coordinates from: {content}")

            except Exception as e:
                print(f"Warning: VLM parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed after {max_retries} attempts: {str(e)}")
                await asyncio.sleep(1)
