"""Shared OBB alignment utility used by all detection methods."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from data.annotations import OBB


def obb_crop(frame: np.ndarray, obb: OBB) -> Optional[tuple]:
    """
    Rotate the frame so the barrel axis is vertical and crop to the barrel.

    The barrel axis is defined as the line connecting the midpoints of the two
    shorter OBB sides. The two shorter sides are the barrel's top and bottom
    caps — this is more stable than using the full corner-to-corner diagonal
    because the barrel is always taller than it is wide.

    Returns (crop, M, rx1, ry1, rx2, start_row, end_row) or None if degenerate.
      crop      — cropped, axis-aligned barrel image (RGB)
      M         — 2x3 affine rotation matrix (needed to back-project predictions)
      rx1, ry1  — top-left of the crop in the rotated frame coordinate space
      rx2       — right edge of the crop in rotated frame space
      start_row — row in crop corresponding to the top OBB cap (scan starts here)
      end_row   — row in crop corresponding to the bottom OBB cap (scan ends here)
    """
    pts = np.array(obb.pts, dtype=np.float32)

    # Find the two shortest OBB sides — these are the barrel's top and bottom caps.
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    lengths = [float(np.linalg.norm(e[1] - e[0])) for e in edges]
    short_idx = sorted(range(4), key=lambda i: lengths[i])[:2]
    mids = [(edges[i][0] + edges[i][1]) / 2 for i in short_idx]
    # Sort the two midpoints by y so top_mid is always closer to the top of the image.
    top_mid, bot_mid = (mids[0], mids[1]) if mids[0][1] <= mids[1][1] else (mids[1], mids[0])

    axis = bot_mid - top_mid
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1:
        # Degenerate OBB (collapsed to a point or line) — skip this frame.
        return None

    # Compute the rotation angle needed to make the barrel axis point straight down.
    # arctan2(x, y) gives the angle from the vertical (not the usual horizontal),
    # which is what we want — we're measuring tilt relative to the image's y-axis.
    angle_deg = float(np.degrees(np.arctan2(axis[0], axis[1])))
    H, W = frame.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -angle_deg, 1.0)

    def rot(pt):
        return (M @ np.array([pt[0], pt[1], 1.0]))[:2]

    rotated = cv2.warpAffine(frame, M, (W, H))
    rot_pts = np.array([rot(p) for p in pts])

    # Crop tightly around the rotated OBB corners.
    # Add 1 pixel on the right/bottom so the edge pixels aren't clipped by integer truncation.
    rx1 = max(0, int(rot_pts[:, 0].min()))
    ry1 = max(0, int(rot_pts[:, 1].min()))
    rx2 = min(W, int(rot_pts[:, 0].max()) + 1)
    ry2 = min(H, int(rot_pts[:, 1].max()) + 1)

    crop = rotated[ry1:ry2, rx1:rx2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None

    # Convert the rotated cap midpoints to crop-local row indices.
    # Detection is restricted to rows between start_row and end_row —
    # the region strictly inside the OBB, between the top and bottom caps.
    start_row = max(0, int(rot(top_mid)[1]) - ry1)
    end_row = min(crop.shape[0], int(rot(bot_mid)[1]) - ry1)

    return crop, M, rx1, ry1, rx2, start_row, end_row
