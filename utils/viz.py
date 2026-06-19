"""Visualization helpers: draw GT and predicted lines on frames."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from data.annotations import BBox, Line


def draw_line(
    img: np.ndarray,
    line: Line,
    color: tuple[int, int, int],
    thickness: int = 3,
    extend: bool = True,
) -> np.ndarray:
    """Draw a line on img (RGB), optionally extended to the image edges.

    extend=True is useful for GT lines that only span the syringe width —
    extending them to the full image makes it easier to judge height visually.
    Set extend=False to draw the line exactly as it was annotated or predicted.
    """
    img = img.copy()
    h, w = img.shape[:2]
    if extend:
        # Project the line to x=0 and x=w using the slope defined by the two endpoints.
        x1, x2 = 0.0, float(w)
        y1 = _y_at_x(line, x1)
        y2 = _y_at_x(line, x2)
    else:
        x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
    pt1 = (int(round(x1)), int(round(np.clip(y1, 0, h - 1))))
    pt2 = (int(round(x2)), int(round(np.clip(y2, 0, h - 1))))
    cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    return img


def draw_bbox(
    img: np.ndarray,
    bbox: BBox,
    color: tuple[int, int, int] = (0, 128, 255),
    thickness: int = 2,
) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(
        img,
        (int(round(bbox.x1)), int(round(bbox.y1))),
        (int(round(bbox.x2)), int(round(bbox.y2))),
        color,
        thickness,
    )
    return img


def annotate_frame(
    img: np.ndarray,
    gt_line: Optional[Line],
    pred_line: Optional[Line],
    bbox: Optional[BBox],
    label: str = "",
) -> np.ndarray:
    """Overlay GT (green) and prediction (red) lines for visual inspection.

    The syringe bbox is intentionally not drawn — it clutters the image and
    isn't useful when evaluating line accuracy. Use draw_bbox() separately
    if you need it for debugging.
    """
    if gt_line is not None:
        img = draw_line(img, gt_line, color=(0, 220, 0), extend=False)
    if pred_line is not None:
        img = draw_line(img, pred_line, color=(220, 30, 30), extend=False)
    if label:
        cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img


def _y_at_x(line: Line, x: float) -> float:
    """Interpolate or extrapolate the y coordinate of a line at a given x position."""
    dx = line.x2 - line.x1
    # Guard against near-vertical lines to avoid division by near-zero.
    if abs(dx) < 1e-6:
        return (line.y1 + line.y2) / 2
    return line.y1 + (x - line.x1) / dx * (line.y2 - line.y1)
