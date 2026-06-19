"""
Generate paper-quality pipeline figure for Section 4.2 (Edge Detection).

Produces 4 separate panels saved as both PDF and PNG:
  (a) OBB-aligned barrel crop
  (b) Dark pixel mask
  (c) Gradient map after morphological closing
  (d) Aligned crop with the detected liquid line drawn

Each panel is saved individually so they can be placed independently in LaTeX.
Run this script from the repo root: python figures/generate_pipeline_fig.py
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data.dataset import SyringeDataset
from methods.edge_scan import OBBEdgeScanDetector
from utils.obb import obb_crop

# Video 132807, frame 73 was chosen because the stopper is clearly visible,
# the syringe is nearly vertical, and the graduation marks create an
# instructive gap-bridging example for the closing step.
VIDEO_ID = "132807"
TARGET_F = 73

ds = SyringeDataset(video_dir="test-videos", labels_dir="test-labels",
                    video_prefix="20260121_")
det = OBBEdgeScanDetector()

# Find the target frame (or the first annotated frame at or after TARGET_F).
sample = None
for s in ds.iter_video(VIDEO_ID):
    if s.annotation.obb is None or s.annotation.bbox is None:
        continue
    if s.frame_id >= TARGET_F:
        sample = s
        break

assert sample is not None, "Frame not found"

obb = sample.annotation.obb
bbox = sample.annotation.bbox
gt = sample.annotation.liquid_line

# Reproduce the detector's internal steps so we can visualize each intermediate.
result = obb_crop(sample.frame, obb)
assert result is not None
crop, M, rx1, ry1, rx2, start_row, end_row = result
cH, cW = crop.shape[:2]

# Kernel sizes mirror the detector exactly so the figure is truly representative.
blur_ksize = max(3, int(cW * 0.20)) | 1
close_ksize = max(3, int(cW * 0.06)) | 1

# Step 1: Dark pixel mask — same logic as OBBEdgeScanDetector.detect()
hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1].astype(np.float32)
val = hsv[:, :, 2].astype(np.float32)
black = (val < 2).astype(np.uint8)
fm = np.zeros((cH + 2, cW + 2), np.uint8)
for r, c in [(0, 0), (0, cW-1), (cH-1, 0), (cH-1, cW-1)]:
    if black[r, c]:
        cv2.floodFill(black.copy(), fm, (c, r), 1)
fill_region = cv2.dilate(fm[1:-1, 1:-1], np.ones((5, 5), np.uint8))
dark_mask = ((val < det.dark_intensity) &
             (sat < det.max_saturation) &
             (fill_region == 0)).astype(np.uint8)

# Step 2: Gradient computation and morphological closing
gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
# Negative Sobel Y isolates downward light-to-dark transitions (top edge of stopper).
grad = np.maximum(0.0, -sobel_y) * dark_mask.astype(np.float32)
grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
closed = cv2.morphologyEx(grad_u8, cv2.MORPH_CLOSE,
                          np.ones((close_ksize, close_ksize), np.uint8)).astype(np.float32)

# Step 3: Row scan to find the liquid-line row
scan = closed[start_row:end_row, :]
row_peak = float(scan.max())
best_local = 0
if row_peak > 0:
    active = scan > row_peak * 0.3
    counts = active.sum(axis=1).astype(np.float32)
    valid_rows = np.where(counts >= scan.shape[1] * 0.10)[0]
    if valid_rows.size:
        # Group consecutive valid rows and pick the densest group.
        groups, g_start = [], 0
        for i in range(1, len(valid_rows)):
            if valid_rows[i] - valid_rows[i-1] > 3:
                groups.append(valid_rows[g_start:i])
                g_start = i
        groups.append(valid_rows[g_start:])
        best_group = max(groups, key=lambda g: float(counts[g].max()))
        best_local = int(best_group[-1])

detected_row = start_row + best_local

# Panel (d): crop with the detected line drawn on it.
result_img = cv2.line(crop.copy(), (0, detected_row), (cW, detected_row), (220, 0, 0), 2)

# Crop the bottom 18% of each panel to remove the finger/hand that held the syringe.
CROP_FRAC = 0.82
ch_keep = int(cH * CROP_FRAC)

panels = [
    ("edge_scan_pipeline_crop",     crop[:ch_keep],     False),
    ("edge_scan_pipeline_mask",     dark_mask[:ch_keep], True),
    ("edge_scan_pipeline_gradient", closed[:ch_keep],    True),
    ("edge_scan_pipeline_result",   result_img[:ch_keep], False),
]

OUT_DIR = Path("figures/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, img, is_gray in panels:
    fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0 * ch_keep / cW))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if is_gray:
        # vmax=(1 if max<=1 else 255) handles both float [0,1] and uint8 masks.
        ax.imshow(img, cmap="gray", vmin=0, vmax=(1 if img.max() <= 1 else 255))
    else:
        ax.imshow(img)
    ax.axis("off")
    out = OUT_DIR / name
    plt.savefig(str(out) + ".pdf", bbox_inches="tight", dpi=200)
    plt.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}.pdf / .png")

print(f"Frame: {sample.frame_id}")
