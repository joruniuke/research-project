#!/usr/bin/env python3
"""
Evaluate a liquid-level detection method on annotated syringe videos.

Usage:
    python run.py --method dummy
    python run.py --method dummy --video 113458
    python run.py --method dummy --save-frames output/frames/
    python run.py --method dummy --save-video output/result.mp4

Add new methods to the METHODS dict below, then implement them in methods/.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

from data.dataset import SyringeDataset
from evaluation.metrics import aggregate_metrics, compute_frame_metrics
from utils.viz import annotate_frame

# Registry of available detection methods.
# Each entry maps a CLI name to (module_path, class_name).
# Methods are imported lazily so that e.g. SAM2 and YOLO are only loaded
# when actually requested — not on every invocation of run.py.
METHODS: dict[str, tuple[str, str]] = {
    "dummy":      ("methods.dummy",             "DummyDetector"),
    "edge_scan":  ("methods.edge_scan",          "OBBEdgeScanDetector"),
    "sam":        ("methods.sam",               "OBBSAMDetector"),
    "yolo_sam":   ("methods.yolo_sam",          "OBBYoloSAMDetector"),
    "resnet":     ("methods.resnet_regression", "ResNetRegressionDetector"),
    "resnet50":   ("methods.resnet_regression", "ResNet50RegressionDetector"),
}

DATA_DIR = Path(__file__).parent


def load_method(name: str):
    """Import and instantiate a detector by its registry name."""
    if name not in METHODS:
        print(f"[error] Unknown method '{name}'. Available: {sorted(METHODS)}")
        sys.exit(1)
    module_path, class_name = METHODS[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def main() -> None:
    parser = argparse.ArgumentParser(description="Liquid-level detection evaluator")
    parser.add_argument("--method", required=True, help="Method name (see METHODS in run.py)")
    parser.add_argument("--video", default=None, help="Only evaluate this video ID, e.g. 113458")
    parser.add_argument("--save-frames", default=None, metavar="DIR",
                        help="Save annotated JPEG frames to this directory")
    parser.add_argument("--save-video", default=None, metavar="PATH",
                        help="Save annotated video(s) here (use {video_id} in path for multiple)")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Dataset split to evaluate on (default: train)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="Root directory containing train-videos/ and train-labels/")
    args = parser.parse_args()

    # Train and test splits were recorded in separate sessions on different days,
    # which is why they use different filename prefixes for their video files.
    # This cross-session setup is intentional — it tests real-world domain shift.
    if args.split == "test":
        video_dir = str(Path(args.data_dir) / "test-videos")
        labels_dir = str(Path(args.data_dir) / "test-labels")
        video_prefix = "20260121_"
    else:
        video_dir = str(Path(args.data_dir) / "train-videos")
        labels_dir = str(Path(args.data_dir) / "train-labels")
        video_prefix = "20260213_"

    dataset = SyringeDataset(
        video_dir=video_dir,
        labels_dir=labels_dir,
        video_prefix=video_prefix,
    )

    detector = load_method(args.method)
    video_ids = [args.video] if args.video else dataset.video_ids

    save_frame_dir = Path(args.save_frames) if args.save_frames else None
    if save_frame_dir:
        save_frame_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for vid in video_ids:
        print(f"\n=== Video {vid} ===")
        # reset() clears any per-video state such as ones using temporal video analysis
        # Our methods are stateless, but we call it unconditionally so stateful methods
        # work correctly across videos.
        detector.reset()
        vid_metrics = []
        video_writer = None

        for sample in dataset.iter_video(vid):
            pred = detector.detect(sample.frame, sample.annotation.bbox, sample.annotation.obb)
            gt = sample.annotation.liquid_line
            fm = compute_frame_metrics(pred, gt, sample.annotation.bbox, sample.annotation.obb)
            vid_metrics.append(fm)

            if save_frame_dir or args.save_video:
                y_err_str = f"  D_end={fm.d_end:.1f}px" if fm.d_end is not None else "  no det"
                vis = annotate_frame(
                    sample.frame,
                    gt,
                    pred,
                    sample.annotation.bbox,
                    label=f"{vid} f{sample.frame_id}{y_err_str}",
                )

                if save_frame_dir:
                    out_path = save_frame_dir / f"{vid}_f{sample.frame_id:04d}.jpg"
                    # Crop to the syringe bbox so saved frames are zoomed in
                    # and easier to inspect visually (the full frame is mostly background).
                    if sample.annotation.bbox is not None:
                        bx1 = max(0, int(sample.annotation.bbox.x1))
                        by1 = max(0, int(sample.annotation.bbox.y1))
                        bx2 = min(vis.shape[1], int(sample.annotation.bbox.x2))
                        by2 = min(vis.shape[0], int(sample.annotation.bbox.y2))
                        vis_save = vis[by1:by2, bx1:bx2]
                    else:
                        vis_save = vis
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_save, cv2.COLOR_RGB2BGR))

                if args.save_video:
                    # {video_id} in the path allows writing one file per video,
                    # e.g. --save-video output/{video_id}.mp4
                    vid_path = args.save_video.format(video_id=vid)
                    if video_writer is None:
                        h, w = vis.shape[:2]
                        Path(vid_path).parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(vid_path, fourcc, 30.0, (w, h))
                    video_writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        if video_writer is not None:
            video_writer.release()
            print(f"  Saved video → {args.save_video.format(video_id=vid)}")

        agg = aggregate_metrics(vid_metrics)
        _print_metrics(agg)
        all_metrics.extend(vid_metrics)

    if len(video_ids) > 1:
        print(f"\n=== Overall ({len(video_ids)} videos) ===")
        _print_metrics(aggregate_metrics(all_metrics))


def _print_metrics(agg) -> None:
    print(f"  Detected:  {agg.n_detected}/{agg.n_frames} frames  ({100*agg.n_detected/max(agg.n_frames,1):.0f}%)")
    print(f"  D_end:     mean={agg.mae_d_end:.1f} px   median={agg.med_d_end:.1f} px")
    print(f"  Y_norm:    mean={agg.mae_y_mid_norm:.4f}   median={agg.med_y_mid_norm:.4f}  (fraction of barrel height)")


if __name__ == "__main__":
    main()
