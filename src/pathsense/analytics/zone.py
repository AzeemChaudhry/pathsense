"""Who is inside an area: zone enter, exit and dwell events, and occupancy.

A zone is a polygon in pixels. Like the line counter, each person is reduced to their
feet (pathsense.analytics.feet), and they are inside when that point is.

A line answers "did someone cross here"; a zone answers "who is in this area, and for how
long". So someone first seen inside (they walked in from the edge of the picture) enters
the zone, and dwell time is measured from that moment.

Rules:

- **Confirmation.** Stepping in or out only counts once the person has stayed there for
  `confirm` seconds; the event keeps the time they stepped. Feet hovering on the edge
  produce nothing.
- **Already inside.** People inside when the video starts (first seen within its first
  `start` seconds) count as entered when `count_present` is true. When false they are
  inside (occupancy, dwell) but get no `zone_enter` and are not in `entered`; their exit is
  still recorded, so `exited` can be larger than `entered`.
- **Vanishing.** Someone not seen for `lost` seconds is gone; the pipeline sets it just
  past the tracker's `lost_track_buffer`. If they were inside, that is a `zone_exit` dated
  when they were last seen (or when they stepped out, if that was waiting for
  confirmation). A glimpse inside shorter than `confirm` is dropped.
- **Dwell.** Every exit also gives a `dwell` event with the seconds spent inside. At the
  end of the video, `finish()` confirms changes still waiting and gives people still
  inside a `dwell` event for their time so far, but no exit: they had not left.
"""

from __future__ import annotations

from dataclasses import dataclass

import supervision as sv

from pathsense.analytics.events import Event
from pathsense.analytics.feet import Feet
from pathsense.analytics.line import EPS, Point


def inside_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray casting: a ray from the point to the right crosses the edges an odd number of
    times when the point is inside."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


@dataclass
class _Change:
    inside: bool  # the state they changed to
    t: float  # when: the event's time
    frame: int


class ZoneCounter:
    """Feed it each frame's tracked people; it returns zone_enter, zone_exit and dwell events.

    zone = ZoneCounter("floor", [(0, 675), (1920, 675), (1920, 1080), (0, 1080)], confirm=0.5)
    for result in pipeline.run(source):
        events = zone.update(result.tracked, result.frame.timestamp, result.frame.index)
    events = zone.finish()
    zone.entered, zone.exited, zone.occupancy
    """

    def __init__(
        self,
        name: str,
        polygon: list[Point],
        confirm: float = 0.0,
        lost: float = 1.0,
        count_present: bool = True,
        start: float = 1.0,
        hidden_legs: float = 0.0,
    ) -> None:
        if len(polygon) < 3:
            raise ValueError(f"Zone {name!r} needs at least 3 points")
        self.name, self.polygon = name, [(float(x), float(y)) for x, y in polygon]
        self.confirm, self.lost = confirm, lost
        self.count_present, self.start = count_present, start
        self.feet = Feet(hidden_legs)
        self.entered = 0
        self.exited = 0
        self._inside: dict[int, float] = {}  # confirmed inside -> since when (s)
        self._pending: dict[int, _Change] = {}  # change waiting for confirmation
        self._last_seen: dict[int, tuple[float, int]] = {}  # track id -> (t, frame)
        self._t0: float | None = None  # time of the first frame
        self._finished = False

    @property
    def occupancy(self) -> int:
        """People inside right now."""
        return len(self._inside)

    def update(self, tracked: sv.Detections, t: float, frame: int) -> list[Event]:
        if self._t0 is None:
            self._t0 = t
        events: list[Event] = []
        seen: set[int] = set()
        if tracked.tracker_id is not None:
            feet, _ = self.feet(tracked, t)
            for track_id, point in zip(tracked.tracker_id, feet, strict=True):
                track_id = int(track_id)
                first_sighting = track_id not in self._last_seen
                seen.add(track_id)
                self._last_seen[track_id] = (t, frame)
                inside = inside_polygon((float(point[0]), float(point[1])), self.polygon)
                if inside and first_sighting and self._present_at_start(t):
                    self._inside[track_id] = t  # already there: inside, no event
                elif inside == (track_id in self._inside):
                    self._pending.pop(track_id, None)  # back where they were: never happened
                elif track_id not in self._pending:
                    self._pending[track_id] = _Change(inside, t, frame)
        for track_id, change in list(self._pending.items()):
            if track_id in seen and t - change.t >= self.confirm - EPS:
                events += self._apply(track_id, change)
        for track_id, (last_t, last_frame) in list(self._last_seen.items()):
            if track_id not in seen and t - last_t >= self.lost - EPS:
                events += self._forget(track_id, last_t, last_frame)
        return events

    def finish(self) -> list[Event]:
        """At the end of the video: confirm changes still waiting, and give people still
        inside a dwell event for their time so far (no exit: they had not left).

        A change still waiting counts without its hold: the video ended, not the person's
        stay, so they cannot be shown to have stepped back. Calling it again returns nothing.
        """
        if self._finished:
            return []
        self._finished = True
        events = []
        for track_id, change in list(self._pending.items()):
            events += self._apply(track_id, change)
        for track_id, since in self._inside.items():
            last_t, last_frame = self._last_seen[track_id]
            events.append(self._dwell(track_id, last_t, last_frame, since))
        return events

    def _present_at_start(self, t: float) -> bool:
        """Already inside when the video started, and not to be counted as entering."""
        return not self.count_present and self._t0 is not None and t - self._t0 < self.start

    def _apply(self, track_id: int, change: _Change) -> list[Event]:
        del self._pending[track_id]
        if change.inside:
            self._inside[track_id] = change.t
            self.entered += 1
            return [Event(change.t, change.frame, track_id, "zone_enter", self.name)]
        return self._exit(track_id, change.t, change.frame)

    def _exit(self, track_id: int, t: float, frame: int) -> list[Event]:
        since = self._inside.pop(track_id)
        self.exited += 1
        return [
            Event(t, frame, track_id, "zone_exit", self.name),
            self._dwell(track_id, t, frame, since),
        ]

    def _dwell(self, track_id: int, t: float, frame: int, since: float) -> Event:
        return Event(t, frame, track_id, "dwell", self.name, duration=round(t - since, 3))

    def _forget(self, track_id: int, last_t: float, last_frame: int) -> list[Event]:
        """The track is gone. If they were inside, they left: when they stepped out if that
        was waiting, else when last seen."""
        del self._last_seen[track_id]
        self.feet.forget(track_id)
        change = self._pending.pop(track_id, None)
        if track_id not in self._inside:
            return []  # a glimpse inside that never got confirmed is dropped
        t, frame = (change.t, change.frame) if change else (last_t, last_frame)
        return self._exit(track_id, t, frame)
