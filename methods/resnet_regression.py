"""
ResNet regression detector for liquid-line Y prediction.

Fine-tuned ResNet18/50 that directly regresses the normalized Y coordinate of
the liquid line from an OBB-aligned barrel crop. The model predicts a single
scalar in [0, 1], representing the line's vertical position as a fraction of
the crop height — all geometric alignment is handled upstream by obb_crop().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from data.annotations import BBox, Line, OBB
from methods.base import LiquidLevelDetector
from utils.obb import obb_crop

BEST_PT_18 = Path("weights/resnet18.pt")
BEST_PT_50 = Path("weights/resnet50.pt")
INPUT_SIZE = (224, 224)
# Standard ImageNet normalization constants — used because the backbone
# was pretrained on ImageNet and expects this input distribution.
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]

_BACKBONES = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ResNetRegressionDetector(LiquidLevelDetector):
    """Fine-tuned ResNet18 that directly regresses the liquid line Y coordinate."""

    name = "resnet_regression"

    def __init__(self, weights: Path = BEST_PT_18, backbone: str = "resnet18",
                 device: Optional[str] = None):
        self.device = device or _best_device()
        # Load the backbone with no pretrained weights — we'll load our own below.
        self.model = _BACKBONES[backbone](weights=None)
        # Replace the classification head with a regression head.
        # Dropout(0.3) provides regularization on the small training set.
        self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.model.fc.in_features, 1))
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(_NORM_MEAN, _NORM_STD),
        ])

    def detect(self, frame: np.ndarray, bbox: Optional[BBox], obb: Optional[OBB] = None) -> Optional[Line]:
        if obb is None:
            return None
        result = obb_crop(frame, obb)
        if result is None:
            return None
        crop, M, rx1, ry1, rx2, *_ = result
        cH, cW = crop.shape[:2]

        tensor = self.transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # The model outputs a single normalized Y in [0, 1] — multiply by
            # crop height to get the pixel row in the aligned crop.
            line_y_norm = float(self.model(tensor).squeeze())

        line_y_crop = line_y_norm * cH
        M_inv = cv2.invertAffineTransform(M)

        def to_orig(x_crop, y_crop):
            # Convert from crop-local coordinates back to original frame coordinates.
            # We add rx1/ry1 to translate from crop space to rotated-frame space,
            # then apply the inverse rotation to get back to the original frame.
            # This is same logic as for the other methods.
            p = M_inv @ np.array([x_crop + rx1, y_crop + ry1, 1.0])
            return float(p[0]), float(p[1])

        x1, y1 = to_orig(0, line_y_crop)
        x2, y2 = to_orig(cW, line_y_crop)
        return Line(x1=x1, y1=y1, x2=x2, y2=y2)


class ResNet50RegressionDetector(ResNetRegressionDetector):
    """Fine-tuned ResNet50 variant — same head and training config as ResNet18."""
    name = "resnet50_regression"

    def __init__(self, device: Optional[str] = None):
        super().__init__(weights=BEST_PT_50, backbone="resnet50", device=device)
