"""Count people crossing a line.

Each person is reduced to one point, the bottom centre of their box (roughly their feet,
which is where they are on the floor). The line goes from `start` to `end`. For every
point we compute which side of the line it is on:

    side = (end - start) x (point - start)      (2D cross product)

The sign is the side. When a person's side flips between two observations, they crossed.

Direction: picture standing on `start` looking at `end`. In image coordinates (y points
down), a person passing from your left to your right is an `enter`; right to left is an
`exit`. For a line drawn left to right across the frame, walking down the image (towards
the camera) is `enter`. Swap the two points to swap the meaning.

Two rules keep the count honest:

- **Margin.** Points closer than `margin` pixels to the line count as "on the line" and do
  not change the side. Someone standing on the line, whose box jitters by a few pixels,
  would otherwise produce enter, exit, enter, exit...
- **Segment, not infinite line.** A flip only counts if the point where it is seen lies
  within the ends of the line, so the line can cover one doorway, not the whole frame.
"""

from __future__ import annotations

import math

import numpy as np
import supervision as sv

from pathsense.analytics.events import Event

Point = tuple[float, float]


def anchor_points(tracked: sv.Detections) -> np.ndarray:
    """Bottom centre of each box, shape (n, 2)."""
    xyxy = tracked.xyxy
    return np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]))


def side_of_line(point: Point, start: Point, end: Point) -> float:
    """Signed distance in pixels: > 0 right of start->end (in image coordinates), < 0 left."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    cross = dx * (point[1] - start[1]) - dy * (point[0] - start[0])
    return cross / math.hypot(dx, dy)


def within_segment(point: Point, start: Point, end: Point) -> bool:
    """Whether the point's projection onto the line falls between start and end."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
    return 0.0 <= t <= 1.0


class LineCounter:
    """Feed it each frame's tracked people; it returns the crossings as events.

    counter = LineCounter((0, 540), (1920, 540))
    for result in pipeline.run(source):
        events = counter.update(result.tracked, result.frame.timestamp, result.frame.index)
    counter.entered, counter.exited
    """

    def __init__(self, start: Point, end: Point, margin: float = 5.0) -> None:
        if start == end:
            raise ValueError("A counting line needs two different points")
        self.start, self.end, self.margin = start, end, margin
        self.entered = 0
        self.exited = 0
        self._side: dict[int, int] = {}  # track id -> last side seen off the line: +1 or -1

    def update(self, tracked: sv.Detections, t: float, frame: int) -> list[Event]:
        events = []
        if tracked.tracker_id is None:
            return events
        for track_id, point in zip(tracked.tracker_id, anchor_points(tracked), strict=True):
            track_id, point = int(track_id), (float(point[0]), float(point[1]))
            distance = side_of_line(point, self.start, self.end)
            if abs(distance) <= self.margin:
                continue  # on the line: wait until they are clearly on one side
            side = 1 if distance > 0 else -1
            previous = self._side.get(track_id)
            self._side[track_id] = side
            if previous is None or previous == side:
                continue  # first sighting, or still on the same side
            if not within_segment(point, self.start, self.end):
                continue  # crossed the line's extension, outside the counted stretch
            if side > 0:
                self.entered += 1
                events.append(Event(t, frame, track_id, "enter"))
            else:
                self.exited += 1
                events.append(Event(t, frame, track_id, "exit"))
        return events
