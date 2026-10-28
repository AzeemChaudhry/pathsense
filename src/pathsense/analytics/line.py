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

Three rules keep the count honest:

- **Margin.** Points closer than `margin` pixels to the line count as "on the line" and do
  not change the side. Someone standing on the line, whose box jitters by a few pixels,
  would otherwise produce enter, exit, enter, exit...
- **Segment, not infinite line.** A flip only counts if the point where it is seen lies
  within the ends of the line, so the line can cover one doorway, not the whole frame.
- **Confirmation.** A flip is only a crossing once the person has stayed on the new side
  for `confirm` seconds; if they come back sooner it never happened. A close-up box that
  widens when an arm stretches out, or a box jittering on the line, flips back within a
  fraction of a second, while a person who walks across stays. The event keeps the time
  of the flip, not of the confirmation. Someone who vanishes on the new side (walked out
  of view) is counted once they have been gone for `confirm` seconds, or by `finish()`
  when the video ends. On a close-up webcam, 53 false crossings from waving became 14 at
  1 s and 4 at 1.5 s, while the sample clip's counts did not change from 0 to 3 s
  (NOTES.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass
class _Pending:
    side: int  # the side they flipped to
    t: float  # when they flipped: the event's time
    frame: int
    last_seen: float


class LineCounter:
    """Feed it each frame's tracked people; it returns the crossings as events.

    counter = LineCounter((0, 540), (1920, 540), confirm=1.0)
    for result in pipeline.run(source):
        events = counter.update(result.tracked, result.frame.timestamp, result.frame.index)
    events = counter.finish()  # crossings still waiting for confirmation
    counter.entered, counter.exited
    """

    def __init__(self, start: Point, end: Point, margin: float = 5.0, confirm: float = 0.0) -> None:
        if start == end:
            raise ValueError("A counting line needs two different points")
        self.start, self.end, self.margin, self.confirm = start, end, margin, confirm
        self.entered = 0
        self.exited = 0
        self._side: dict[int, int] = {}  # track id -> confirmed side: +1 or -1
        self._pending: dict[int, _Pending] = {}  # track id -> flip waiting for confirmation

    def update(self, tracked: sv.Detections, t: float, frame: int) -> list[Event]:
        events: list[Event] = []
        seen: set[int] = set()
        if tracked.tracker_id is not None:
            for track_id, point in zip(tracked.tracker_id, anchor_points(tracked), strict=True):
                track_id, point = int(track_id), (float(point[0]), float(point[1]))
                seen.add(track_id)
                if pending := self._pending.get(track_id):
                    pending.last_seen = t
                distance = side_of_line(point, self.start, self.end)
                if abs(distance) <= self.margin:
                    continue  # on the line: wait until they are clearly on one side
                side = 1 if distance > 0 else -1
                confirmed = self._side.get(track_id)
                if confirmed is None:
                    self._side[track_id] = side  # first sighting is not a crossing
                elif side == confirmed:
                    self._pending.pop(track_id, None)  # came back: it never happened
                elif track_id not in self._pending:
                    if within_segment(point, self.start, self.end):
                        self._pending[track_id] = _Pending(side, t, frame, t)
                    else:
                        self._side[track_id] = side  # passed beside the line, not over it
        # Confirm flips held long enough, or whose person vanished on the new side.
        for track_id, pending in list(self._pending.items()):
            held = t - pending.t >= self.confirm and track_id in seen
            gone = track_id not in seen and t - pending.last_seen >= self.confirm
            if held or gone:
                events.append(self._count(track_id, pending))
        return events

    def finish(self) -> list[Event]:
        """At the end of the video: crossings still waiting were never undone, so count them."""
        return [self._count(track_id, p) for track_id, p in list(self._pending.items())]

    def _count(self, track_id: int, pending: _Pending) -> Event:
        del self._pending[track_id]
        self._side[track_id] = pending.side
        if pending.side > 0:
            self.entered += 1
            return Event(pending.t, pending.frame, track_id, "enter")
        self.exited += 1
        return Event(pending.t, pending.frame, track_id, "exit")
