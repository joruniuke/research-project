"""
Template for a new liquid-level detection method.

Steps to add a new method:
1. Copy this file to methods/your_method_name.py
2. Fill in the detect() method
3. Register it in the METHODS dict at the top of run.py:
       "your_name": ("methods.your_method_name", "YourDetector"),
4. Run:  python run.py --method your_name
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from data.annotations import BBox, Line, OBB
from methods.base import LiquidLevelDetector


class TemplateDetector(LiquidLevelDetector):
    name = "template"

    def __init__(self) -> None:
        # Initialize any model weights, parameters, etc. here.
        pass

    def reset(self) -> None:
        # Called before each new video.
        # Reset per-video state here (e.g. background model, optical-flow history).
        pass

    def detect(self, frame: np.ndarray, bbox: Optional[BBox], obb: Optional[OBB] = None) -> Optional[Line]:
        """
        Args:
            frame  — full RGB image, shape (H, W, 3), dtype uint8
            bbox   — syringe bounding box in full-frame coordinates (may be None)

        Returns:
            Line in full-frame pixel coordinates, or None if detection fails.

        Tips:
            - To work on the cropped syringe region:
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                crop = frame[y1:y2, x1:x2]
              Remember to translate any detected coordinates back to full-frame coords.

            - cv2 expects BGR; convert with: bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        """
        raise NotImplementedError
