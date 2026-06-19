#!/usr/bin/env python3
"""
Extract the three-panel OBB preprocessing figure for the paper.

  Panel A: raw video frame (full frame as captured)
  Panel B: same frame with the OBB drawn on it (shows annotation)
  Panel C: OBB-aligned barrel crop (output of obb_crop())

This figure illustrates the preprocessing step shared by all four methods.
Panels are saved as separate PNG files so they can be placed individually in LaTeX.

Usage:
    python figures/extract_obb_panels.py --video 132807 --frame 60 --out output/figures
    python figures/extract_obb_panels.py --split train --video 113458 --frame 10
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from data.dataset import SyringeDataset
from utils.obb import obb_crop


def draw_obb(img: np.ndarray, obb, color=(0, 220, 0), thickness=3) -> np.ndarray:
    """Draw the oriented bounding box as a green quadrilateral on the image."""
    out = img.copy()
    pts = np.array(obb.pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="132807",
                        help="6-digit video ID to extract from")
    parser.add_argument("--frame", type=int, default=60,
                        help="Annotation index within the video (0-based, counts only annotated frames)")
    parser.add_argument("--out", default="output/figures",
                        help="Directory to save the panel PNG files")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Dataset split the video belongs to")
    args = parser.parse_args()

    # Resolve paths relative to the repo root, not this script's location.
    base = Path(__file__).parent.parent
    if args.split == "test":
        dataset = SyringeDataset(
            video_dir=str(base / "test-videos"),
            labels_dir=str(base / "test-labels"),
            video_prefix="20260121_",
        )
    else:
        dataset = SyringeDataset(
            video_dir=str(base / "train-videos"),
            labels_dir=str(base / "train-labels"),
            video_prefix="20260213_",
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through annotated frames until we reach the requested index.
    target = args.frame
    sample = None
    for i, s in enumerate(dataset.iter_video(args.video)):
        if i == target:
            sample = s
            break

    if sample is None:
        print(f"Frame index {target} not found in video {args.video} "
              f"(video may have fewer than {target + 1} annotated frames).")
        return

    frame = sample.frame   # RGB numpy array
    obb = sample.annotation.obb

    if obb is None:
        print("No OBB annotation on this frame — try a different frame index.")
        return

    # Panel A: raw frame, converted to BGR for OpenCV imwrite.
    panel_a = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    path_a = out_dir / f"{args.video}_f{target:04d}_A_raw.png"
    cv2.imwrite(str(path_a), panel_a)
    print(f"Panel A → {path_a}")

    # Panel B: raw frame with the OBB quadrilateral drawn in green.
    panel_b = draw_obb(frame, obb, color=(0, 220, 0), thickness=3)
    panel_b = cv2.cvtColor(panel_b, cv2.COLOR_RGB2BGR)
    path_b = out_dir / f"{args.video}_f{target:04d}_B_obb.png"
    cv2.imwrite(str(path_b), panel_b)
    print(f"Panel B → {path_b}")

    # Panel C: OBB-aligned barrel crop — the input seen by all detection methods.
    result = obb_crop(frame, obb)
    if result is None:
        print("obb_crop returned None for this frame — try a different frame index.")
        return
    crop, *_ = result
    panel_c = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    path_c = out_dir / f"{args.video}_f{target:04d}_C_crop.png"
    cv2.imwrite(str(path_c), panel_c)
    print(f"Panel C → {path_c}")

    print("Done.")


if __name__ == "__main__":
    main()
