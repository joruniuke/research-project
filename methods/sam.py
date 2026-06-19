"""
OBB-guided SAM2 segmentation detector (zero-shot, heuristic blob prompt).

Pipeline:
  1. Rotate and crop the barrel using the OBB.
  2. Find the largest solid dark blob in the scan region
     (blur → HSV threshold → connected components → score by area × fill_ratio).
  3. Prompt SAM2 with the bounding box of that blob.
  4. Restrict the mask to the blob box; find the first row from the top
     reaching 90% of peak mask width (past the triangular stopper tip).
  5. Back-project that row to original frame coordinates.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from data.annotations import BBox, Line, OBB
from methods.base import LiquidLevelDetector
from utils.obb import obb_crop


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OBBSAMDetector(LiquidLevelDetector):
    """Zero-shot SAM2 segmentation prompted by a heuristic dark-blob detector."""

    name = "sam"

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: Optional[str] = None,
        dark_intensity: float = 85.0,
        max_saturation: float = 40.0,
        blur_factor: float = 0.10,
        width_threshold: float = 0.90,
    ) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device or _best_device()
        self.dark_intensity = dark_intensity    # HSV value cutoff for the dark blob mask
        self.max_saturation = max_saturation    # HSV saturation cutoff — excludes colored regions
        self.blur_factor = blur_factor          # Gaussian blur before thresholding, as fraction of crop width
        self.width_threshold = width_threshold  # fraction of peak mask width that defines the liquid line row
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id,
                                                             device=self.device)

    def detect(self, frame: np.ndarray, bbox: Optional[BBox],
               obb: Optional[OBB] = None) -> Optional[Line]:
        if obb is None or bbox is None:
            return None

        # Step 1: OBB alignment — rotate and crop to the barrel region.
        result = obb_crop(frame, obb)
        if result is None:
            return None
        crop, M, rx1, ry1, rx2, start_row, end_row = result
        cH, cW = crop.shape[:2]
        if end_row - start_row < 5:
            return None

        blur_ksize = max(3, int(cW * self.blur_factor)) | 1

        # Step 2: Dark blob detection
        # Blur before thresholding to reduce noise and help the HSV threshold
        # produce cleaner blob boundaries around the rubber stopper.
        blurred = cv2.GaussianBlur(crop, (blur_ksize, blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        # Exclude rotation fill corners (pure black pixels that are not part of the barrel).
        black = (val < 2).astype(np.uint8)
        fm = np.zeros((cH + 2, cW + 2), np.uint8)
        for r, c in [(0, 0), (0, cW-1), (cH-1, 0), (cH-1, cW-1)]:
            if black[r, c]:
                cv2.floodFill(black.copy(), fm, (c, r), 1)
        fill_region = cv2.dilate(fm[1:-1, 1:-1], np.ones((5, 5), np.uint8))

        # Apply the dark mask only within the barrel extent (between the OBB short sides)
        # to avoid picking up the finger, plunger cap, or needle as the stopper blob.
        dark = np.zeros((cH, cW), np.uint8)
        dark[start_row:end_row, :] = (
            (val[start_row:end_row] < self.dark_intensity) &
            (sat[start_row:end_row] < self.max_saturation) &
            (fill_region[start_row:end_row] == 0)
        ).astype(np.uint8)

        n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        if n_lbl < 2:
            # No dark blobs found in the scan region — can't prompt SAM2.
            return None

        # Score each blob by area × fill_ratio, where fill_ratio = area / (W × H).
        # This prefers blobs that are large AND compact (high fill), which describes
        # the rubber stopper well. Thin streaks or scattered noise score low even
        # if they have a large bounding box.
        best_lbl, best_score = -1, 0.0
        for i in range(1, n_lbl):
            a = int(stats[i, cv2.CC_STAT_AREA])
            fill = a / max(stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT], 1)
            if a * fill > best_score:
                best_score = a * fill
                best_lbl = i
        if best_lbl < 0:
            return None

        bx1 = int(stats[best_lbl, cv2.CC_STAT_LEFT])
        by1 = int(stats[best_lbl, cv2.CC_STAT_TOP])
        bx2 = bx1 + int(stats[best_lbl, cv2.CC_STAT_WIDTH])
        by2 = by1 + int(stats[best_lbl, cv2.CC_STAT_HEIGHT])

        # Step 3: SAM2 segmentation
        # We use the blob's axis-aligned bounding box as the spatial prompt.
        # multimask_output=True asks SAM2 to return 3 candidate masks at
        # different confidence levels; we pick the one with the highest predicted IoU.
        box = np.array([[bx1, by1, bx2, by2]], dtype=np.float32)
        with torch.inference_mode():
            self.predictor.set_image(crop)
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        if masks is None or len(masks) == 0:
            return None

        # Restrict the best mask to the blob box region — this prevents SAM2
        # from leaking onto the barrel body or the finger above the stopper.
        m = masks[int(scores.argmax())].astype(bool)
        m_box = np.zeros_like(m)
        m_box[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2]

        # Step 4: Line extraction
        # The rubber stopper is often roughly triangular — it's narrow at the
        # tip and widens downwards. We scan downward and find the first row 
        # where the mask width reaches 90% of its maximum. That is where the
        # stopper finishes widening, which corresponds to the flat liquid interface.
        row_counts = m_box[by1:by2, bx1:bx2].sum(axis=1).astype(float)
        if row_counts.max() == 0:
            return None
        first_wide = np.where(row_counts >= row_counts.max() * self.width_threshold)[0]
        if not first_wide.size:
            return None

        best_y_rot = ry1 + by1 + int(first_wide[0])

        # Step 5: Back-project to original frame coords
        M_inv = cv2.invertAffineTransform(M)

        def inv_rot(pt):
            return (M_inv @ np.array([pt[0], pt[1], 1.0]))[:2]

        p1 = inv_rot([float(rx1), float(best_y_rot)])
        p2 = inv_rot([float(rx2), float(best_y_rot)])
        return Line(x1=float(p1[0]), y1=float(p1[1]),
                    x2=float(p2[0]), y2=float(p2[1]))
