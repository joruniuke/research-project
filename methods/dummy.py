"""Dummy baseline: returns the horizontal midline of the syringe bbox.

Not a real detector — exists only so you can verify the evaluation pipeline
end-to-end before plugging in real methods.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from data.annotations import BBox, Line
from methods.base import LiquidLevelDetector


class DummyDetector(LiquidLevelDetector):
    name = "dummy"

    def detect(self, frame: np.ndarray, bbox: Optional[BBox], obb=None) -> Optional[Line]:
        if bbox is None:
            h, w = frame.shape[:2]
            mid_y = h / 2
            return Line(x1=0.0, y1=mid_y, x2=float(w), y2=mid_y)
        mid_y = (bbox.y1 + bbox.y2) / 2
        return Line(x1=bbox.x1, y1=mid_y, x2=bbox.x2, y2=mid_y)
