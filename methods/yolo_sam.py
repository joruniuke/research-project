"""
YOLO-prompted SAM2 segmentation detector.

Pipeline:
  1. Rotate and crop the barrel using the OBB.
  2. Run YOLO (fine-tuned on stopper bounding boxes) to detect the stopper region.
  3. Use the YOLO bounding box as a spatial prompt for SAM2 Hiera-Large.
  4. Restrict the SAM2 mask to the YOLO box region.
  5. Find the first row from the top where the mask reaches 90% of its peak
     width — this is where the triangular stopper tip finishes widening,
     corresponding to the flat liquid interface.
  6. Back-project that row to original frame coordinates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from data.annotations import BBox, Line, OBB
from methods.base import LiquidLevelDetector
from utils.obb import obb_crop

YOLO_WEIGHTS = Path("weights/yolo_obb.pt")
WIDTH_THRESHOLD = 0.90   # fraction of peak mask width used to locate the liquid line


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OBBYoloSAMDetector(LiquidLevelDetector):
    """YOLO-prompted SAM2 segmentation for rubber stopper liquid-line detection."""

    name = "yolo_sam"

    def __init__(
        self,
        yolo_weights: Path = YOLO_WEIGHTS,
        sam_model_id: str = "facebook/sam2.1-hiera-large",
        device: Optional[str] = None,
        width_threshold: float = WIDTH_THRESHOLD,
    ) -> None:
        from ultralytics import YOLO
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device or _best_device()
        self.width_threshold = width_threshold
        # YOLO11n-OBB fine-tuned on stopper crops to locate the stopper precisely.
        self.yolo = YOLO(str(yolo_weights))
        # SAM2.1 Hiera-Large is the largest and most accurate SAM2 variant.
        self.sam2 = SAM2ImagePredictor.from_pretrained(sam_model_id,
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

        # Step 2: YOLO stopper detection in the aligned crop.
        # YOLO was trained on OBB-aligned crops, so we run it on the crop
        # rather than the full frame. No confidence threshold is applied —
        # we always take the highest-confidence detection if any exist.
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        yolo_res = self.yolo(bgr, verbose=False)[0]
        if yolo_res.obb is None or len(yolo_res.obb) == 0:
            # No stopper detected — skip this frame.
            return None

        # The YOLO model outputs oriented bounding boxes (OBB). Convert the
        # best detection to an axis-aligned box by taking the extremes of each
        # axis across all four corners. This is a conservative approximation
        # but avoids propagating rotation to SAM2's box prompt interface.
        best_idx = int(yolo_res.obb.conf.argmax())
        box_pts = yolo_res.obb.xyxyxyxy[best_idx].cpu().numpy()
        bx1 = max(0, int(box_pts[:, 0].min()))
        bx2 = min(cW, int(box_pts[:, 0].max()))
        by1 = max(0, int(box_pts[:, 1].min()))
        by2 = min(cH, int(box_pts[:, 1].max()))
        if bx2 <= bx1 or by2 <= by1:
            return None

        # Step 3: SAM2 segmentation with YOLO box prompt.
        # multimask_output=True returns 3 candidate masks; we pick the one
        # with the highest predicted IoU score.
        box_np = np.array([[bx1, by1, bx2, by2]], dtype=np.float32)
        with torch.inference_mode():
            self.sam2.set_image(crop)
            masks, scores, _ = self.sam2.predict(box=box_np,
                                                  multimask_output=True)
        if masks is None or len(masks) == 0:
            return None

        # Step 4: Restrict the best mask to the YOLO box region.
        # This prevents the mask from leaking onto the barrel body outside the stopper.
        m = masks[int(scores.argmax())].astype(bool)
        m_box = np.zeros_like(m)
        m_box[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2]

        # Step 5: Line extraction.
        # Scan downward within the stopper box and find the first row where
        # the mask width reaches 90% of its peak. The stopper narrows to a tip
        # at the top, so this threshold picks the row where it becomes fully
        # wide — the flat bottom edge that marks the liquid interface.
        row_counts = m_box[by1:by2, bx1:bx2].sum(axis=1).astype(float)
        if row_counts.max() == 0:
            return None

        first_wide = np.where(row_counts >= row_counts.max() * self.width_threshold)[0]
        if not first_wide.size:
            return None

        best_y_rot = ry1 + by1 + int(first_wide[0])

        # Step 6: Back-project to original frame coords.
        M_inv = cv2.invertAffineTransform(M)

        def inv_rot(pt):
            return (M_inv @ np.array([pt[0], pt[1], 1.0]))[:2]

        p1 = inv_rot([float(rx1), float(best_y_rot)])
        p2 = inv_rot([float(rx2), float(best_y_rot)])
        return Line(x1=float(p1[0]), y1=float(p1[1]),
                    x2=float(p2[0]), y2=float(p2[1]))
