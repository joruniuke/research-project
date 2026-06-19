"""
Generate paper-quality pipeline figures for Section 4.3 (Blob-Prompted SAM2).

Produces 4 separate panels saved as both PDF and PNG:
  (a) OBB-aligned barrel crop (clean input)
  (b) Crop with the dark-blob bounding box prompt drawn
  (c) Crop with the SAM2 mask overlaid semi-transparently
  (d) Crop with the detected liquid line drawn

Each panel is saved individually so they can be placed independently in LaTeX.
Run this script from the repo root: python figures/generate_sam_fig.py
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data.dataset import SyringeDataset
from utils.obb import obb_crop

# Same video and frame as the edge scan figure so both figures show identical input,
# making the method comparison easier for the reader.
VIDEO_ID = "132807"
TARGET_F = 73
# Crop the bottom 25% to remove the finger/hand from the visualization.
CROP_FRAC = 0.75

# SAM2 parameters — must match OBBSAMDetector defaults exactly.
BLUR_FACTOR = 0.10
DARK_INTENSITY = 85.0
MAX_SATURATION = 40.0
WIDTH_THRESHOLD = 0.90

device = "mps" if torch.backends.mps.is_available() else "cpu"
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)

ds = SyringeDataset(video_dir="test-videos", labels_dir="test-labels",
                    video_prefix="20260121_")

# Find the target frame (or the first annotated frame at or after TARGET_F).
sample = None
for s in ds.iter_video(VIDEO_ID):
    if s.annotation.obb is None or s.annotation.bbox is None:
        continue
    if s.frame_id >= TARGET_F:
        sample = s
        break
assert sample is not None

obb = sample.annotation.obb

result = obb_crop(sample.frame, obb)
assert result is not None
crop, M, rx1, ry1, rx2, start_row, end_row = result
cH, cW = crop.shape[:2]

blur_ksize = max(3, int(cW * BLUR_FACTOR)) | 1

# Blob detection — mirrors OBBSAMDetector step by step.
blurred = cv2.GaussianBlur(crop, (blur_ksize, blur_ksize), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1].astype(np.float32)
val = hsv[:, :, 2].astype(np.float32)

# Exclude rotation fill corners.
black = (val < 2).astype(np.uint8)
fm = np.zeros((cH + 2, cW + 2), np.uint8)
for r, c in [(0, 0), (0, cW-1), (cH-1, 0), (cH-1, cW-1)]:
    if black[r, c]:
        cv2.floodFill(black.copy(), fm, (c, r), 1)
fill_region = cv2.dilate(fm[1:-1, 1:-1], np.ones((5, 5), np.uint8))

# Threshold only within the scan region between the OBB caps.
dark = np.zeros((cH, cW), np.uint8)
dark[start_row:end_row, :] = (
    (val[start_row:end_row] < DARK_INTENSITY) &
    (sat[start_row:end_row] < MAX_SATURATION) &
    (fill_region[start_row:end_row] == 0)
).astype(np.uint8)

n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
assert n_lbl >= 2, "No dark blob found"

# Score blobs by area × fill_ratio and select the best one.
best_lbl, best_score = -1, 0.0
for i in range(1, n_lbl):
    a = int(stats[i, cv2.CC_STAT_AREA])
    fill = a / max(stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT], 1)
    if a * fill > best_score:
        best_score = a * fill
        best_lbl = i

bx1 = int(stats[best_lbl, cv2.CC_STAT_LEFT])
by1 = int(stats[best_lbl, cv2.CC_STAT_TOP])
bx2 = bx1 + int(stats[best_lbl, cv2.CC_STAT_WIDTH])
by2 = by1 + int(stats[best_lbl, cv2.CC_STAT_HEIGHT])

# SAM2 prediction using the blob box as a spatial prompt.
box = np.array([[bx1, by1, bx2, by2]], dtype=np.float32)
with torch.inference_mode():
    predictor.set_image(crop)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)

# Select the mask with the highest predicted IoU and restrict to the blob box.
best_mask = masks[int(scores.argmax())].astype(bool)
m_box = np.zeros_like(best_mask)
m_box[by1:by2, bx1:bx2] = best_mask[by1:by2, bx1:bx2]

# Line extraction — first row from top where mask width reaches 90% of peak.
row_counts = m_box[by1:by2, bx1:bx2].sum(axis=1).astype(float)
first_wide = np.where(row_counts >= row_counts.max() * WIDTH_THRESHOLD)[0]
detected_row = by1 + int(first_wide[0]) if first_wide.size else (by1 + by2) // 2

# Apply the bottom crop before building the panels.
ch_keep = int(cH * CROP_FRAC)

# Panel (a): clean OBB-aligned crop — the raw input to the detector.
img_a = crop[:ch_keep]

# Panel (b): crop with the blob bounding box prompt drawn in blue.
img_b = crop.copy()
cv2.rectangle(img_b, (bx1, by1), (bx2, by2), (30, 144, 255), 2)
img_b = img_b[:ch_keep]

# Panel (c): SAM2 mask overlaid with a semi-transparent orange tint.
# The 0.4/0.6 blend makes the mask visible without hiding the underlying image.
img_c = crop.copy()
overlay = img_c.copy()
overlay[m_box] = (overlay[m_box] * 0.4 + np.array([255, 140, 0]) * 0.6).astype(np.uint8)
img_c = overlay[:ch_keep]

# Panel (d): detected liquid line drawn in red.
img_d = crop.copy()
cv2.line(img_d, (0, detected_row), (cW, detected_row), (220, 0, 0), 2)
img_d = img_d[:ch_keep]

panels = [
    ("sam_pipeline_crop",   img_a),
    ("sam_pipeline_prompt", img_b),
    ("sam_pipeline_mask",   img_c),
    ("sam_pipeline_result", img_d),
]

OUT_DIR = Path("figures/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_H = 4.5
for name, img in panels:
    h, w = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(FIG_H * w / h, FIG_H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img)
    ax.axis("off")
    out = OUT_DIR / name
    plt.savefig(str(out) + ".pdf", bbox_inches="tight", dpi=200)
    plt.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}.pdf / .png")

print(f"Frame: {sample.frame_id}")
