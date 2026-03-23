import os
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from config import HTTP_PROXY, HTTPS_PROXY


class SAMClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        os.environ["HTTP_PROXY"] = HTTP_PROXY
        os.environ["HTTPS_PROXY"] = HTTPS_PROXY

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_checkpoint = "facebook/sam2-hiera-large"
        model_cfg = "sam2_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self._initialized = True

    def segment(self, image: np.ndarray, point: list[int]) -> np.ndarray:
        """Generate mask from image and point prompt."""
        self.predictor.set_image(image)

        point_coords = np.array([point])
        point_labels = np.array([1])

        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )

        return masks[0]
