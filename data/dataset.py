"""Dataset: pairs each label zip with its video and yields annotated frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from data.annotations import FrameAnnotation, parse_annotations


@dataclass
class Sample:
    # One fully loaded, annotated video frame ready for detection.
    video_id: str       # e.g. "113458" — matches the label zip stem
    frame_id: int       # 0-based frame index within the video file
    frame: np.ndarray   # RGB, shape (H, W, 3), dtype uint8
    annotation: FrameAnnotation


class SyringeDataset:
    """Iterates over annotated frames from all labeled syringe videos.

    Pairs label zips with video files by matching filenames. The video file is
    expected at <video_dir>/<prefix><zip_stem>.mp4, where the prefix differs
    between the training and test recording sessions (see run.py for details).

    Frames are loaded on demand from disk — the full dataset is never held
    in memory at once. This matters because each video is several hundred MB.
    """

    def __init__(self, video_dir: str, labels_dir: str,
                 video_prefix: str = "20260213_") -> None:
        self.video_dir = Path(video_dir)
        self.labels_dir = Path(labels_dir)
        self.video_prefix = video_prefix
        self._pairs = self._find_pairs()

    def _find_pairs(self) -> list[tuple[Path, Path]]:
        # Match each label zip to its corresponding video file. If a zip has no
        # matching video (e.g. the data directory is not present), it's silently
        # skipped so the rest of the dataset still loads cleanly.
        pairs = []
        for zip_path in sorted(self.labels_dir.glob("*.zip")):
            video_path = self.video_dir / f"{self.video_prefix}{zip_path.stem}.mp4"
            if video_path.exists():
                pairs.append((video_path, zip_path))
        return pairs

    @property
    def video_ids(self) -> list[str]:
        # Returns the 6-digit video IDs (zip stems), used for display and CLI filtering.
        return [zp.stem for _, zp in self._pairs]

    def __iter__(self) -> Iterator[Sample]:
        for video_path, zip_path in self._pairs:
            yield from self._iter_video(video_path, zip_path)

    def iter_video(self, video_id: str) -> Iterator[Sample]:
        """Iterate over annotated frames for a single video."""
        for video_path, zip_path in self._pairs:
            if zip_path.stem == video_id:
                yield from self._iter_video(video_path, zip_path)
                return
        raise ValueError(f"No labeled video with id {video_id!r}. Available: {self.video_ids}")

    def _iter_video(self, video_path: Path, zip_path: Path) -> Iterator[Sample]:
        annotations = parse_annotations(str(zip_path))
        cap = cv2.VideoCapture(str(video_path))
        try:
            for fid, ann in sorted(annotations.items()):
                # Skip frames that lack either the liquid line or OBB —
                # without both we can't run or evaluate any of the four methods.
                if ann.liquid_line is None or ann.obb is None:
                    continue
                # Seek directly to the annotated frame rather than reading
                # sequentially, since not every frame in the video is annotated.
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame_bgr = cap.read()
                if not ret:
                    continue
                yield Sample(
                    video_id=zip_path.stem,
                    frame_id=fid,
                    frame=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    annotation=ann,
                )
        finally:
            # Always release the video capture, even if iteration is interrupted
            # or raises an exception midway through.
            cap.release()
