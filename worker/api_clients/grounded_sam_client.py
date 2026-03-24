import torch
import numpy as np
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys

# Add Grounded-SAM-2 to path
GROUNDED_SAM2_PATH = Path(__file__).parent.parent.parent / "third_party" / "Grounded-SAM-2"
sys.path.insert(0, str(GROUNDED_SAM2_PATH))

from grounding_dino.groundingdino.util.inference import load_model, predict


class GroundedSAMClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build SAM2 predictor
        sam2_checkpoint = GROUNDED_SAM2_PATH / "checkpoints" / "sam2.1_hiera_large.pt"
        sam2_config = str(GROUNDED_SAM2_PATH / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml")
        sam2_model = build_sam2(sam2_config, str(sam2_checkpoint), device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Build Grounding DINO model
        gdino_config = str(GROUNDED_SAM2_PATH / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
        gdino_checkpoint = str(GROUNDED_SAM2_PATH / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")
        self.grounding_model = load_model(
            model_config_path=gdino_config,
            model_checkpoint_path=gdino_checkpoint,
            device=self.device
        )

        # Enable optimizations
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._initialized = True

    def segment(self, image: np.ndarray, text: str, box_threshold: float = 0.35, text_threshold: float = 0.25) -> tuple[np.ndarray, list]:
        """Perform grounded segmentation. Returns (mask, bboxes)."""
        # Ensure text ends with period and is lowercase
        if not text.endswith('.'):
            text = text + '.'
        text = text.lower()

        # Prepare image for Grounding DINO
        image_rgb = image if image.shape[2] == 3 else image[:, :, :3]
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()

        # Detect boxes with Grounding DINO
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        if len(boxes) == 0:
            # Return empty mask if no detection
            return np.zeros(image.shape[:2], dtype=bool), []

        # Convert boxes to xyxy format
        h, w = image.shape[:2]
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Segment with SAM2
        self.sam2_predictor.set_image(image_rgb)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

        # Combine all masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        combined_mask = np.any(masks, axis=0)

        return combined_mask, input_boxes.tolist()
