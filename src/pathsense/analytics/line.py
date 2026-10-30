"""Count people crossing a line.

Each person is reduced to one point, their feet (the bottom centre of their box, or an
estimate from the head when their legs are hidden: pathsense.analytics.feet). The line
goes through two or more points, from `start` (the first) to `end` (the last); more than
two follow a curved edge, such as a train's side bent by a wide lens. For every person we
find the nearest segment and compute which side of it they are on:

    side = (b - a) x (point - a)      (2D cross product, segment from a to b)

The sign is the side. When a person's side flips between two observations, they crossed.

Direction: picture standing on `start` looking at `end`. In image coordinates (y points
down), a person passing from your left to your right is an `enter`; right to left is an
`exit`. For a line drawn left to right across the frame, walking down the image (towards
the camera) is `enter`. Swap the two points to swap the meaning.

Three rules keep the count honest:

- **Margin.** Points closer than `margin` pixels to the line count as "on the line" and do
  not change the side. Someone standing on the line, whose box jitters by a few pixels,
  would otherwise produce enter, exit, enter, exit... With `margin_height` above 0 the
  margin is also at least that share of the person's height, so it grows with people
  close to the camera (for example 0.15: 100 px for someone 680 px tall).
- **Segment, not infinite line.** A flip only counts if the point where it is seen lies
  within the ends of the line, so the line can cover one doorway, not the whole frame.
- **Confirmation.** A flip is only a crossing once the person has stayed on the new side
  for `confirm` seconds; if they come back sooner it never happened. Frames where the feet
  are on the line (within the margin) neither cancel nor confirm it. A close-up box that
  widens when an arm stretches out, or a box jittering on the line, flips back within a
  fraction of a second, while a person who walks across stays. The event keeps the time
  of the flip, not of the confirmation. On a close-up webcam, 53 false crossings from
  waving became 14 at 1 s and 4 at 1.5 s, while the sample clip's counts did not change
  from 0 to 3 s (NOTES.md).
- **Vanishing.** A person not seen for `lost` seconds is forgotten; the pipeline sets it
  just past the tracker's `lost_track_buffer`, so the ID cannot come back. A flip still
  waiting then counts: they walked out of view on the new side. This also counts a track
  the tracker drops right after a flip, trading a rare false crossing for not missing
  people who leave the picture just past the line. A shorter gap, someone hidden behind
  another person, changes nothing. `finish()` counts flips still waiting when the video
  ends, even if the feet are on the line.

One more rule is off in this class (`bounce=0`) and on in the pipeline
(`analytics.line_bounce`, 12 s by default):

- **Bounce.** In a dense crowd the tracker sometimes slides an ID from one person onto the
  person behind them. The ID then seems to turn round: person A's crossing, a crossing
  back (the ID moving onto B), and B's crossing the same way as A. With `bounce` above 0,
  a crossing that reverses the ID's last one is held for `bounce` seconds; if the ID
  crosses back the original way within that time, the held one is dropped and the other
  two count. Someone who really turns round at the line crosses back once and stays, so
  their crossing is only late, not lost. Someone who dithers, crossing, crossing back and
  crossing again within `bounce`, looks exactly like a swap: their middle crossing is
  dropped (IN, OUT, IN counts as IN, IN). A held crossing waits at most `bounce` plus
  `confirm` seconds, so the totals lag by at most that; a crossing back still unconfirmed
  by then (feet on the line) no longer drops it, and is held in its turn. On the tracker
  study's crowds the rule removed 4 to 15% of the wrong crossings left after tuning
  (docs/TRACKER_STUDY.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import supervision as sv

from pathsense.analytics.events import Event
from pathsense.analytics.feet import Feet, anchor_points

Point = tuple[float, float]
__all__ = ["LineCounter", "anchor_points", "polyline_side", "side_of_line", "within_segment"]


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


def polyline_side(point: Point, points: list[Point]) -> tuple[float, bool]:
    """Signed distance to the nearest segment (sign as in side_of_line), and whether the
    point is within the polyline's ends rather than beyond its first or last point."""
    best: tuple[float, float, int, float] | None = None  # (distance, signed, segment, t)
    last = len(points) - 2
    for i, (a, b) in enumerate(pairwise(points)):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)
        c = min(1.0, max(0.0, t))
        distance = math.hypot(point[0] - (a[0] + c * dx), point[1] - (a[1] + c * dy))
        if best is None or distance < best[0]:
            best = (distance, side_of_line(point, a, b), i, t)
    distance, signed, segment, t = best  # type: ignore[misc]
    within = not ((segment == 0 and t < 0) or (segment == last and t > 1))
    if len(points) == 2:  # one segment: the perpendicular distance, as before
        return signed, within
    return math.copysign(distance, signed), within


# Time comparisons allow for float rounding: frame / fps can give 0.49999999999999994.
EPS = 1e-6


@dataclass
class _Pending:
    side: int  # the side they flipped to
    t: float  # when they flipped: the event's time
    frame: int


class LineCounter:
    """Feed it each frame's tracked people; it returns the crossings as events.

    counter = LineCounter((0, 540), (1920, 540), confirm=0.5)  # or 3+ points
    for result in pipeline.run(source):
        events = counter.update(result.tracked, result.frame.timestamp, result.frame.index)
    events = counter.finish()  # crossings still waiting for confirmation
    counter.entered, counter.exited
    """

    def __init__(
        self,
        *points: Point,
        margin: float = 5.0,
        margin_height: float = 0.0,
        confirm: float = 0.0,
        lost: float = 1.0,
        hidden_legs: float = 0.0,
        bounce: float = 0.0,
    ) -> None:
        if len(points) < 2 or any(a == b for a, b in pairwise(points)):
            raise ValueError(
                "A counting line needs two or more points, each different from the next"
            )
        self.points = [(float(x), float(y)) for x, y in points]
        self.start, self.end = self.points[0], self.points[-1]
        self.margin, self.margin_height = margin, margin_height
        self.confirm, self.lost, self.bounce = confirm, lost, bounce
        self.feet = Feet(hidden_legs)
        self.entered = 0
        self.exited = 0
        self._side: dict[int, int] = {}  # track id -> confirmed side: +1 or -1
        self._pending: dict[int, _Pending] = {}  # track id -> flip waiting for confirmation
        self._last_seen: dict[int, float] = {}
        self._crossed: dict[int, int] = {}  # track id -> side of their last crossing
        self._held: dict[int, _Pending] = {}  # track id -> reversing crossing held (bounce)
        self.bounced = 0  # crossings dropped by the bounce rule

    def update(self, tracked: sv.Detections, t: float, frame: int) -> list[Event]:
        events: list[Event] = []
        seen: set[int] = set()
        clear: dict[int, int] = {}  # track id -> side, for feet clearly off the line now
        if tracked.tracker_id is not None:
            feet, heights = self.feet(tracked, t)
            for track_id, point, height in zip(tracked.tracker_id, feet, heights, strict=True):
                track_id, point = int(track_id), (float(point[0]), float(point[1]))
                seen.add(track_id)
                self._last_seen[track_id] = t
                distance, within = polyline_side(point, self.points)
                if abs(distance) <= max(self.margin, self.margin_height * float(height)):
                    continue  # on the line: wait until they are clearly on one side
                side = clear[track_id] = 1 if distance > 0 else -1
                confirmed = self._side.get(track_id)
                if confirmed is None:
                    self._side[track_id] = side  # first sighting is not a crossing
                elif side == confirmed:
                    self._pending.pop(track_id, None)  # came back: it never happened
                elif track_id not in self._pending:
                    if within:
                        self._pending[track_id] = _Pending(side, t, frame)
                    else:
                        self._side[track_id] = side  # passed beside the line, not over it
        for track_id, pending in list(self._pending.items()):
            if clear.get(track_id) == pending.side and t - pending.t >= self.confirm - EPS:
                events += self._count(track_id, pending)
        for track_id, held in list(self._held.items()):
            # Past `bounce`, wait only for a crossing back that started in time and is
            # still being confirmed, and never past `bounce + confirm`.
            pending = self._pending.get(track_id)
            waiting = pending is not None and pending.t - held.t <= self.bounce + EPS
            overdue = t - held.t > self.bounce + self.confirm + EPS
            if (overdue or not waiting) and t - held.t > self.bounce + EPS:
                events.append(self._event(track_id, self._held.pop(track_id)))
        for track_id, last in list(self._last_seen.items()):
            if track_id not in seen and t - last >= self.lost - EPS:
                events += self._forget(track_id)
        return events

    def finish(self) -> list[Event]:
        """At the end of the video: crossings still waiting were never undone, so count them."""
        events = [e for i, p in list(self._pending.items()) for e in self._count(i, p)]
        return events + [self._event(i, self._held.pop(i)) for i in list(self._held)]

    def _forget(self, track_id: int) -> list[Event]:
        """The track is gone; a flip still waiting means they walked off on the new side."""
        del self._last_seen[track_id]
        self.feet.forget(track_id)
        pending = self._pending.get(track_id)
        events = self._count(track_id, pending) if pending else []
        if track_id in self._held:
            events.append(self._event(track_id, self._held.pop(track_id)))
        self._side.pop(track_id, None)  # never off the line: no side yet
        self._crossed.pop(track_id, None)
        return events

    def _count(self, track_id: int, pending: _Pending) -> list[Event]:
        """A confirmed flip: count it, or hold or drop it under the bounce rule."""
        del self._pending[track_id]
        self._side[track_id] = pending.side
        held = self._held.pop(track_id, None)
        if held is not None and pending.t - held.t <= self.bounce + EPS:
            self._crossed[track_id] = pending.side
            same = self._same_person(track_id)
            self._counted(track_id)
            if same:
                return [self._event(track_id, held), self._event(track_id, pending)]
            self.bounced += 1  # straight back the original way: the held one was a swap
            return [self._event(track_id, pending)]
        events = [self._event(track_id, held)] if held else []
        reverses = self._crossed.get(track_id, pending.side) != pending.side
        self._crossed[track_id] = pending.side
        if self.bounce > 0 and reverses:
            if self._other_person(track_id):
                self._counted(track_id)
                return events
            self._held[track_id] = pending  # the look of the last counted crossing stays
            return events
        self._counted(track_id)
        return events + [self._event(track_id, pending)]

    def _same_person(self, track_id: int) -> bool:
        """A held bounce was really one person turning back; see AppearanceLine."""
        return False

    def _other_person(self, track_id: int) -> bool:
        """A reversal is plainly the ID moving to someone else; see AppearanceLine."""
        return False

    def _counted(self, track_id: int) -> None:
        """A crossing was settled for this ID."""

    def _event(self, track_id: int, pending: _Pending) -> Event:
        if pending.side > 0:
            self.entered += 1
            return Event(pending.t, pending.frame, track_id, "enter")
        self.exited += 1
        return Event(pending.t, pending.frame, track_id, "exit")
