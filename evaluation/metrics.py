"""Metrics for comparing predicted liquid-level lines to ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.annotations import BBox, Line, OBB


@dataclass
class FrameMetrics:
    # Per-frame results. Both fields are None when the detector returned None
    # (i.e. no detection was made for this frame).
    d_end: Optional[float]       # D_end in pixels — symmetric endpoint distance
    y_mid_norm: Optional[float]  # Y_norm — vertical midpoint error as fraction of barrel height


@dataclass
class AggregateMetrics:
    # Summary statistics over a collection of frames.
    # Only detected frames contribute to the error statistics — missed frames
    # are counted separately in n_frames vs n_detected.
    mae_d_end: float       # mean D_end (px)
    med_d_end: float       # median D_end (px)
    mae_y_mid_norm: float  # mean Y_norm (fraction of barrel height)
    med_y_mid_norm: float  # median Y_norm
    n_frames: int          # total frames attempted
    n_detected: int        # frames where the detector returned a line


def _barrel_height_px(obb: OBB) -> float:
    """Compute barrel height in pixels from the OBB.

    The barrel height is the distance between the midpoints of the two shorter
    OBB sides — the same definition used in obb_crop() to find the barrel axis.
    Using the OBB-derived height rather than the axis-aligned bbox height is
    more accurate for tilted syringes, where the bbox height overestimates the
    true barrel length due to the rotation.
    """
    pts = np.array(obb.pts, dtype=np.float32)
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    lengths = [float(np.linalg.norm(e[1] - e[0])) for e in edges]
    short_idx = sorted(range(4), key=lambda i: lengths[i])[:2]
    mids = [(edges[i][0] + edges[i][1]) / 2 for i in short_idx]
    return float(np.linalg.norm(mids[1] - mids[0]))


def compute_frame_metrics(
    pred: Optional[Line],
    gt: Line,
    bbox: Optional[BBox],
    obb: Optional[OBB] = None,
) -> FrameMetrics:
    """Compute D_end and Y_norm for a single frame.

    D_end: symmetric mean endpoint distance. We try both endpoint assignments
    (P1-A1, P2-A2) and (P1-A2, P2-A1) and take the minimum. This makes the
    metric invariant to which end of the line was annotated first, since a
    liquid-level line is undirected (it has no inherent left/right orientation).

    Y_norm: absolute vertical error of the predicted line's midpoint, normalized
    by the barrel height. Prefers OBB-derived height over the bbox height because
    the OBB measurement is more stable for tilted syringes. Falls back to bbox
    if OBB is not available.
    """
    if pred is None:
        return FrameMetrics(d_end=None, y_mid_norm=None)

    # D_end
    a1 = np.array([gt.x1, gt.y1])
    a2 = np.array([gt.x2, gt.y2])
    p1 = np.array([pred.x1, pred.y1])
    p2 = np.array([pred.x2, pred.y2])

    match1 = (np.linalg.norm(p1 - a1) + np.linalg.norm(p2 - a2)) / 2
    match2 = (np.linalg.norm(p1 - a2) + np.linalg.norm(p2 - a1)) / 2
    d_end = float(min(match1, match2))

    # Y_norm
    gt_mid_y = (gt.y1 + gt.y2) / 2.0
    pred_mid_y = (pred.y1 + pred.y2) / 2.0
    y_err_px = abs(pred_mid_y - gt_mid_y)
    if obb is not None:
        barrel_h = _barrel_height_px(obb)
        y_mid_norm = float(y_err_px / barrel_h) if barrel_h > 1 else None
    elif bbox is not None:
        barrel_h = abs(bbox.y2 - bbox.y1)
        y_mid_norm = float(y_err_px / barrel_h) if barrel_h > 1 else None
    else:
        y_mid_norm = None

    return FrameMetrics(d_end=d_end, y_mid_norm=y_mid_norm)


def aggregate_metrics(frame_metrics: list[FrameMetrics]) -> AggregateMetrics:
    """Compute mean and median error over all detected frames.

    Frames where the detector returned None are excluded from the error
    statistics but are still counted in n_frames so the caller can compute
    the detection rate as n_detected / n_frames.
    """
    detected = [m for m in frame_metrics if m.d_end is not None]
    d_vals = [m.d_end for m in detected]
    y_vals = [m.y_mid_norm for m in detected if m.y_mid_norm is not None]
    mae_d = float(np.mean(d_vals)) if d_vals else float("nan")
    med_d = float(np.median(d_vals)) if d_vals else float("nan")
    mae_y = float(np.mean(y_vals)) if y_vals else float("nan")
    med_y = float(np.median(y_vals)) if y_vals else float("nan")
    return AggregateMetrics(
        mae_d_end=mae_d,
        med_d_end=med_d,
        mae_y_mid_norm=mae_y,
        med_y_mid_norm=med_y,
        n_frames=len(frame_metrics),
        n_detected=len(detected),
    )
