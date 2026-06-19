"""OBB-guided edge scan detector for rubber stopper liquid-line detection."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from data.annotations import BBox, Line, OBB
from methods.base import LiquidLevelDetector
from utils.obb import obb_crop


class OBBEdgeScanDetector(LiquidLevelDetector):
    """
    Detects the rubber stopper liquid line by scanning for dark-to-light
    vertical gradients inside the OBB-aligned barrel crop.

    Pipeline:
      1. Rotate and crop the barrel using the OBB.
      2. Build a dark pixel mask (excludes rotation fill and bright/colored areas).
      3. Compute downward Sobel gradient; apply morphological closing to
         bridge gaps caused by reflections or graduation markings.
      4. Keep only the widest gradient blob (the stopper is almost always the
         biggest feature); find the densest row group and take its bottommost
         row, which corresponds to the liquid interface.
      5. Back-project the detected row to original frame coordinates.
    """

    name = "edge_scan"

    def __init__(
        self,
        edge_threshold: float = 15.0,
        dark_intensity: float = 120.0,
        max_saturation: float = 40.0,
        blur_factor: float = 0.20,
        close_factor: float = 0.06,
        row_fill_threshold: float = 0.10,
    ) -> None:
        self.edge_threshold = edge_threshold         # minimum gradient peak to attempt detection
        self.dark_intensity = dark_intensity         # HSV value cutoff for the dark pixel mask
        self.max_saturation = max_saturation         # HSV saturation cutoff — excludes colored regions
        self.blur_factor = blur_factor               # Gaussian blur kernel as fraction of crop width
        self.close_factor = close_factor             # morphological close kernel as fraction of crop width
        self.row_fill_threshold = row_fill_threshold # fraction of row width that must be active

    def detect(self, frame: np.ndarray, bbox: Optional[BBox], obb: Optional[OBB] = None) -> Optional[Line]:
        if obb is None or bbox is None:
            return None

        # Step 1: OBB alignment
        result = obb_crop(frame, obb)
        if result is None:
            return None
        crop, M, rx1, ry1, rx2, start_row, end_row = result
        start_row = max(0, start_row)
        end_row = min(crop.shape[0], end_row)
        # If OBB is less than 5 px, it is too small to hold any relevant information.
        if end_row - start_row < 5:
            return None

        # Kernel sizes scale with crop width so behavior is resolution-independent.
        blur_ksize = max(3, int(crop.shape[1] * self.blur_factor)) | 1
        close_ksize = max(3, int(crop.shape[1] * self.close_factor)) | 1

        # Step 2: Dark pixel mask
        # We want to detect the rubber stopper, which is dark gray.
        # Pixels with high brightness OR high saturation are excluded —
        # this rejects the bright barrel body and any colored markings or caps.
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        # The rotation fills corners with pure black, which would wrongly pass
        # the dark mask. Flood-fill from each corner to identify these fill pixels,
        # then dilate slightly to exclude gradient artifacts at the fill boundary.
        black = (val < 2).astype(np.uint8)
        fm = np.zeros((crop.shape[0] + 2, crop.shape[1] + 2), np.uint8)
        for r, c in [(0, 0), (0, crop.shape[1]-1), (crop.shape[0]-1, 0), (crop.shape[0]-1, crop.shape[1]-1)]:
            if black[r, c]:
                cv2.floodFill(black.copy(), fm, (c, r), 1)
        fill_region = cv2.dilate(fm[1:-1, 1:-1], np.ones((5, 5), np.uint8))
        dark_mask = ((val < self.dark_intensity) & (sat < self.max_saturation) & (fill_region == 0)).astype(np.float32)

        # Step 3: Gradient + morphological closing
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # Negative Sobel Y keeps only light-to-dark transitions going downward —
        # this is the top edge of the dark rubber stopper as seen from above.
        # Multiplying by the dark mask suppresses gradients outside the stopper region.
        grad = np.maximum(0.0, -cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)) * dark_mask

        # Closing bridges gaps in the stopper edge caused by graduation markings
        # or small reflections, so the stopper appears as one continuous blob.
        grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad = cv2.morphologyEx(grad_u8, cv2.MORPH_CLOSE,
                                np.ones((close_ksize, close_ksize), np.uint8)).astype(np.float32)

        # Step 4: Row selection
        peak = float(grad.max())
        if peak < self.edge_threshold:
            # Gradient is too weak — no confident stopper edge in this frame.
            return None

        # Binarize and find connected components. Score each by a weighted sum of
        # normalized area and normalized width: the stopper produces a single wide
        # component, while graduation markings produce many small narrow ones.
        # The 0.3/0.7 weighting favors width, which is the more discriminative feature.
        binary = (grad > peak * 0.3).astype(np.uint8)
        _, lbl_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        top_labels = sorted(range(1, len(stats)),
                            key=lambda i: 0.3 * stats[i, cv2.CC_STAT_AREA] / max(1, stats[:, cv2.CC_STAT_AREA].max())
                                        + 0.7 * stats[i, cv2.CC_STAT_WIDTH] / max(1, stats[:, cv2.CC_STAT_WIDTH].max()),
                            reverse=True)[:1]
        clean = np.isin(lbl_map, top_labels).astype(np.float32)
        grad = grad * clean

        scan = grad[start_row:end_row, :]
        if scan.max() == 0:
            return None

        # A row is valid if enough of its width has active gradient pixels.
        # Counting totals (not runs) handles the triangular stopper shape, where
        # the edge appears as two separated side fragments rather than one solid line.
        active = scan > (scan.max() * 0.3)
        counts = active.sum(axis=1).astype(np.float32)
        valid_rows = np.where(counts >= scan.shape[1] * self.row_fill_threshold)[0]
        if not valid_rows.size:
            return None

        # Split valid rows into consecutive groups; pick the densest group.
        # Tolerates gaps of up to 3 rows, which can appear where a thin graduation
        # mark interrupts the stopper edge. The bottommost row of the densest group
        # marks the widest part of the stopper — this is the liquid interface.
        gaps = np.where(np.diff(valid_rows) > 3)[0] + 1
        groups = np.split(valid_rows, gaps)
        best_group = max(groups, key=lambda g: float(counts[g].max()))
        best_y_rot = ry1 + start_row + int(best_group[-1])

        # Step 5: Back-project to original frame coords
        M_inv = cv2.invertAffineTransform(M)

        def inv_rot(pt):
            return (M_inv @ np.array([pt[0], pt[1], 1.0]))[:2]

        p1 = inv_rot([float(rx1), float(best_y_rot)])
        p2 = inv_rot([float(rx2), float(best_y_rot)])
        return Line(x1=float(p1[0]), y1=float(p1[1]), x2=float(p2[0]), y2=float(p2[1]))
