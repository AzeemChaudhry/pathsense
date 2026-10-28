"""Experimental, study only: split a track when its box jumps the way a walker cannot.

In a dense crowd the tracker sometimes moves an ID from one person to the person next to
them. A line then sees a U-turn: the first person's ID crossed one way, and the second
person, now holding that ID, is on the other side (NOTES.md: 18 of 29 wrong-way crossings
on MOT20-03 were swaps like this).

`IdentityGuard` sits between the tracker and the stitcher. It follows each tracker ID and
gives it a new ID when, from one frame to the next:

- the feet land more than `jump` box heights away from where the person was heading, or
- the box height changes by more than `size` (from 0.67 to 1.5 times with 0.5), measured
  against the median of the last `HISTORY` seconds so one cut box does not count.

The old ID then looks like a person who vanished, and the new one like a person who just
appeared, so the stitcher can give each box back the ID of the person who was there. Only
position, motion and box size are used: no appearance, nothing kept after the video.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from pathsense.analytics.feet import anchor_points

HISTORY = 0.5  # seconds of positions and heights used for speed and usual height
FIRST_ID = 1_000_000  # split IDs start here, far above the tracker's own


@dataclass
class _Track:
    alias: int  # the ID this tracker ID is given now
    history: deque = field(default_factory=deque)  # (t, feet, height) over HISTORY seconds

    def expected(self, t: float) -> np.ndarray:
        (t0, p0, _), (t1, p1, _) = self.history[0], self.history[-1]
        velocity = (p1 - p0) / (t1 - t0) if t1 > t0 else np.zeros(2)
        return p1 + velocity * (t - t1)

    def usual_height(self) -> float:
        return float(np.median([h for _, _, h in self.history]))


class IdentityGuard:
    def __init__(self, jump: float = 0.5, size: float = 0.5) -> None:
        self.jump, self.size = jump, size
        self._tracks: dict[int, _Track] = {}
        self._next = FIRST_ID
        self.splits = 0  # how many times a track was split

    def __call__(self, tracked: sv.Detections, t: float) -> sv.Detections:
        """The same detections, with tracker IDs that jumped replaced by new ones."""
        if tracked.tracker_id is None or len(tracked) == 0:
            self._prune(t)
            return tracked
        feet = anchor_points(tracked)
        heights = tracked.xyxy[:, 3] - tracked.xyxy[:, 1]
        out = []
        for tracker_id, point, height in zip(tracked.tracker_id, feet, heights, strict=True):
            tracker_id, height = int(tracker_id), float(height)
            track = self._tracks.get(tracker_id)
            if track is None:
                track = self._tracks[tracker_id] = _Track(tracker_id)
            elif self._jumped(track, point, height, t):
                track.alias, track.history = self._next, deque()
                self._next += 1
                self.splits += 1
            track.history.append((t, point.copy(), height))
            while t - track.history[0][0] > HISTORY:
                track.history.popleft()
            out.append(track.alias)
        self._prune(t)
        result = tracked[np.arange(len(tracked))]  # a copy
        result.tracker_id = np.array(out, dtype=int)
        return result

    def _jumped(self, track: _Track, feet: np.ndarray, height: float, t: float) -> bool:
        usual = track.usual_height()
        ratio = height / usual
        if not 1 / (1 + self.size) <= ratio <= 1 + self.size:
            return True
        off = float(np.hypot(*(feet - track.expected(t))))
        return off > self.jump * max(height, usual)

    def _prune(self, t: float) -> None:
        """Forget tracker IDs not seen for a while; the tracker has dropped them too."""
        gone = [i for i, s in self._tracks.items() if t - s.history[-1][0] > 10.0]
        for tracker_id in gone:
            del self._tracks[tracker_id]
