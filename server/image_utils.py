import base64
import io
import numpy as np
from PIL import Image

def encode_image(img: np.ndarray) -> str:
    """Convert numpy array to base64 PNG string."""
    img_pil = Image.fromarray(img.astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def decode_image(b64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    img_data = base64.b64decode(b64_str)
    img_pil = Image.open(io.BytesIO(img_data))
    return np.array(img_pil)

def blend_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color: tuple = (255, 0, 0)) -> np.ndarray:
    """Alpha blend mask with image using specified color."""
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask_norm = mask.astype(float) / max(mask.max(), 1.0)
    overlay = np.zeros_like(image)
    overlay[:, :] = color

    blended = image.copy()
    for i in range(3):
        blended[:, :, i] = (1 - alpha * mask_norm) * image[:, :, i] + alpha * mask_norm * overlay[:, :, i]

    return blended.astype(np.uint8)
