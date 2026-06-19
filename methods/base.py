"""Abstract base class for all liquid-level detection methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from data.annotations import BBox, Line, OBB


class LiquidLevelDetector(ABC):
    """
    Implement `detect` to plug in a new method.

    `detect` receives:
        frame  — full RGB frame, shape (H, W, 3)
        bbox   — syringe bounding box in full-frame pixel coords (may be None)
        obb    — oriented bounding box (may be None)

    `detect` should return:
        A Line in full-frame pixel coordinates, or None if detection failed.

    Override `reset` if your method holds per-video state that should be
    cleared between videos — for example, a temporal method that smooths
    predictions with a rolling median filter would reset its frame buffer here.
    """

    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, bbox: Optional[BBox], obb: Optional[OBB] = None) -> Optional[Line]:
        ...

    def reset(self) -> None:
        """Called before each new video. Clear any accumulated state here."""
        pass
