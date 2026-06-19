"""Parse CVAT XML annotations from label zip files."""
from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Line:
    # Represents a liquid-level line as two endpoints in pixel coordinates.
    # The line is undirected — (x1,y1) and (x2,y2) can be either end.
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class BBox:
    # Axis-aligned bounding box in pixel coordinates (top-left to bottom-right).
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class OBB:
    """Oriented bounding box: 4 corner points in pixel coordinates.

    Points come directly from CVAT's polygon annotation and are stored in
    CVAT order (not necessarily clockwise or axis-aligned). All methods that
    use the OBB sort the sides by length to find the barrel axis, so the
    point ordering here doesn't matter.
    """
    pts: list[tuple[float, float]]  # exactly 4 (x, y) corner points


@dataclass
class FrameAnnotation:
    """Holds all annotation data for a single frame.

    obb is None for older videos that were annotated with axis-aligned bboxes only.
    """
    frame_id: int
    bbox: Optional[BBox]
    liquid_line: Optional[Line]
    obb: Optional[OBB] = field(default=None)


def _bbox_from_pts(pts: list[tuple[float, float]]) -> BBox:
    """Compute axis-aligned bounding box from a list of (x, y) points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def parse_annotations(zip_path: str) -> dict[int, FrameAnnotation]:
    """Parse CVAT XML from a label zip. Returns {frame_id: FrameAnnotation}.

    The annotation format evolved during the project. Early videos use a <box>
    element for the syringe (axis-aligned bbox). Later videos were re-annotated
    with a <polygon> (4-point OBB) to capture barrel rotation. This function
    handles both formats transparently.

    When a polygon is present, the OBB is stored in FrameAnnotation.obb and an
    axis-aligned BBox is derived from corner extremes so that any code path that
    only uses bbox still works correctly.
    """
    with zipfile.ZipFile(zip_path) as zf:
        # Each zip contains exactly one XML file exported from CVAT.
        xml_name = next(n for n in zf.namelist() if n.endswith(".xml"))
        with zf.open(xml_name) as f:
            root = ET.parse(f).getroot()

    tracks = root.findall("track")

    # CVAT uses per-object tracks. Each track has one annotation element per frame.
    # outside="1" means the object was marked as not visible in that frame —
    # we skip those because there's nothing to detect there.
    bbox_by_frame: dict[int, BBox] = {}
    obb_by_frame: dict[int, OBB] = {}
    for track in tracks:
        if track.attrib["label"] != "Syringe":
            continue
        for elem in track:
            if elem.attrib.get("outside") != "0":
                continue
            fid = int(elem.attrib["frame"])
            if elem.tag == "box":
                # Old-format axis-aligned annotation.
                bbox_by_frame[fid] = BBox(
                    x1=float(elem.attrib["xtl"]),
                    y1=float(elem.attrib["ytl"]),
                    x2=float(elem.attrib["xbr"]),
                    y2=float(elem.attrib["ybr"]),
                )
            elif elem.tag == "polygon":
                # New-format OBB — 4 corners stored as "x,y;x,y;x,y;x,y".
                raw_pts = elem.attrib.get("points", "")
                pts = []
                for p in raw_pts.split(";"):
                    if p.strip():
                        pts.append(tuple(map(float, p.split(","))))
                if len(pts) == 4:
                    obb = OBB(pts=pts)  # type: ignore[arg-type]
                    obb_by_frame[fid] = obb
                    # Derive a BBox from the OBB corners for backwards compatibility.
                    bbox_by_frame[fid] = _bbox_from_pts(pts)

    # The liquid line label name changed slightly across annotation batches —
    # accept all known variants so older and newer zips are handled uniformly.
    LIQUID_LABELS = {"Liquid Level", "Liquid Line", "Liquid line"}
    line_by_frame: dict[int, Line] = {}
    for track in tracks:
        if track.attrib["label"] not in LIQUID_LABELS:
            continue
        for elem in track:
            # outside="1" frames have no visible liquid line — skip them.
            if elem.attrib.get("outside") == "1":
                continue
            fid = int(elem.attrib["frame"])
            pts = elem.attrib["points"].split(";")
            x1, y1 = map(float, pts[0].split(","))
            x2, y2 = map(float, pts[1].split(","))
            line_by_frame[fid] = Line(x1, y1, x2, y2)

    # Build the union of all frames that have either a bbox or a line annotation.
    # Some frames may have one but not the other (e.g. syringe temporarily
    # occluded), so we keep them and let the caller decide what to do.
    all_frames = sorted(set(bbox_by_frame) | set(line_by_frame))
    return {
        fid: FrameAnnotation(
            frame_id=fid,
            bbox=bbox_by_frame.get(fid),
            liquid_line=line_by_frame.get(fid),
            obb=obb_by_frame.get(fid),
        )
        for fid in all_frames
    }
