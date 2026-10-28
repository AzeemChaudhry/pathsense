"""Run a detector on overlapping tiles of a large frame, for small and crowded people.

A detector shrinks every frame to its input size (YOLO 640). A 4K portrait frame is then
shrunk 6 times, and a person 175 px tall becomes 30 px: most are missed, and a cluster
leaving a train turns into a few boxes or none. Cutting the frame into overlapping tiles of
`tile` pixels and detecting in each keeps people large enough to find.

Merging needs care. A person cut by a seam gives a partial box in that tile (their top
half), which a usual overlap check keeps next to the whole box, and merging boxes that lie
inside one another would also merge two people in a crowd. So:

1. the whole frame is detected once, for people too large to fit in a tile;
2. each tile's boxes that touch a seam inside the frame are dropped: with a 20 % overlap, a
   person small enough to be cut appears whole in the neighbouring tile;
3. boxes found twice (whole frame and tile, or two tiles) are reduced to one by NMS.

On the GTX 1650, tiles of 960 px on a 4K portrait frame find 100 people on the market
square (2 at the default input size) and 39 at the subway doors (14), at about 160 ms a
frame instead of 12 ms (NOTES.md). Meant for recorded video; for live cameras a larger
`detector.imgsz` is the cheaper step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from pathsense.detect.base import Detector

OVERLAP = 0.2  # share of a tile that overlaps the next
SEAM = 2  # px; a box this close to a seam inside the frame was cut by it
DUPLICATE_IOU = 0.5


def tiles(width: int, height: int, tile: int) -> list[tuple[int, int, int, int]]:
    """Tile rectangles (x1, y1, x2, y2) covering the frame, overlapping by OVERLAP; the last
    tile in each row and column is aligned to the frame's edge."""

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        step = int(tile * (1 - OVERLAP))
        positions = list(range(0, size - tile, step))
        return [*positions, size - tile]

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]


class SlicedDetector:
    def __init__(self, detector: Detector, tile: int) -> None:
        self.detector, self.tile = detector, tile
        self.name = f"{detector.name}+slice{tile}"

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        whole = self.detector(frame)
        if width <= self.tile and height <= self.tile:
            return whole  # one tile would be the whole frame
        found = [whole]
        for x1, y1, x2, y2 in tiles(width, height, self.tile):
            boxes = self.detector(frame[y1:y2, x1:x2])
            if len(boxes) == 0:
                continue
            boxes.xyxy = boxes.xyxy + [x1, y1, x1, y1]
            cut = np.zeros(len(boxes), bool)  # touching a seam that is inside the frame
            if x1 > 0:
                cut |= boxes.xyxy[:, 0] <= x1 + SEAM
            if y1 > 0:
                cut |= boxes.xyxy[:, 1] <= y1 + SEAM
            if x2 < width:
                cut |= boxes.xyxy[:, 2] >= x2 - SEAM
            if y2 < height:
                cut |= boxes.xyxy[:, 3] >= y2 - SEAM
            found.append(boxes[~cut])
        merged = sv.Detections.merge(found)
        if len(merged) == 0:
            return merged
        return merged.with_nms(threshold=DUPLICATE_IOU, class_agnostic=True)
