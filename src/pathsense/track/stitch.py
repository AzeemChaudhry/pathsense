"""Give a person the tracker lost and found again their old ID back.

The tracker forgets a person after `lost_track_buffer` (about 1 s). Someone hidden longer,
behind a pillar or in a crowd, or missed by the detector for a while because they are far
away, comes back with a new ID. Lines and zones then see a new person: a second zone enter,
a crossing counted twice, a dwell time cut in two (NOTES.md: the grocery shopper's first
track, the subway platform).

`Stitcher` sits between the tracker and the analytics. When a new tracker ID appears, it
looks at people who vanished in the last `seconds`. One of them is the same person when:

- the new box's feet are within `distance` box heights of where that person would be now,
  walking on at their last speed, and
- the box height is within `size` of theirs (from 0.67 to 1.5 times with 0.5).

With `top` above 0, a box outside that height range can still be linked by its top. When a
cart hides someone's legs the detector moves from a whole body box to an upper body box:
the bottom jumps and the height halves, but the head and the width stay. The new box's top
centre must then be within `top` heights of where theirs would be now, using the taller of
the two boxes as the height, and its width within `size` of theirs. Off by default: see
docs/DETECTOR_STUDY.md.

The closest match wins; each vanished person can be matched once. Only position, motion
and box size are used here: no faces, nothing kept after the video. With appearance on,
`pathsense.appearance.AppearanceStitcher` also refuses links that look wrong.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from pathsense.analytics.feet import anchor_points

HISTORY = 0.5  # seconds of positions used for a person's speed


@dataclass
class _Person:
    feet: np.ndarray  # last feet (x, y)
    height: float  # last box height
    width: float  # last box width
    t: float  # last seen
    history: deque = field(default_factory=deque)  # (t, feet, height) over HISTORY seconds

    def velocity(self) -> np.ndarray:
        if len(self.history) < 2:
            return np.zeros(2)
        (t0, p0, _), (t1, p1, _) = self.history[0], self.history[-1]
        return (p1 - p0) / (t1 - t0) if t1 > t0 else np.zeros(2)

    def top_velocity(self) -> np.ndarray:
        """Of the top of the box, which a change of box leaves alone."""
        if len(self.history) < 2:
            return np.zeros(2)
        (t0, p0, h0), (t1, p1, h1) = self.history[0], self.history[-1]
        if t1 <= t0:
            return np.zeros(2)
        return (p1 - p0 - np.array([0.0, h1 - h0])) / (t1 - t0)


class Stitcher:
    def __init__(
        self, seconds: float = 2.0, distance: float = 1.0, size: float = 0.5, top: float = 0.0
    ) -> None:
        self.seconds, self.distance, self.size, self.top = seconds, distance, size, top
        self._person: dict[int, int] = {}  # tracker id -> person id
        self._people: dict[int, _Person] = {}  # person id -> last state
        self.stitched = 0  # how many times a lost person got their ID back

    def __call__(self, tracked: sv.Detections, t: float) -> sv.Detections:
        """The same detections with tracker_id replaced by person IDs."""
        if tracked.tracker_id is None or len(tracked) == 0:
            self._prune(t)
            return tracked
        feet = anchor_points(tracked)
        heights = tracked.xyxy[:, 3] - tracked.xyxy[:, 1]
        widths = tracked.xyxy[:, 2] - tracked.xyxy[:, 0]
        tracker_ids = [int(i) for i in tracked.tracker_id]
        present = {self._person[i] for i in tracker_ids if i in self._person}
        new = [k for k, i in enumerate(tracker_ids) if i not in self._person]
        for k, person in self._match(new, feet, heights, widths, present, t).items():
            self._person[tracker_ids[k]] = person
            present.add(person)
            self.stitched += 1
        out, used = [0] * len(tracker_ids), set()
        # People under their own tracker ID go first, so when an old tracker ID comes back
        # next to the one stitched to it, the stitched one gives the person ID up. It falls
        # back to its own tracker ID, which no one else can hold: it was stitched at birth.
        owners_first = sorted(
            range(len(tracker_ids)),
            key=lambda k: self._person.get(tracker_ids[k], tracker_ids[k]) != tracker_ids[k],
        )
        for k in owners_first:
            tracker_id = tracker_ids[k]
            person = self._person.setdefault(tracker_id, tracker_id)
            if person in used:
                person = self._person[tracker_id] = tracker_id
            used.add(person)
            out[k] = person
            self._remember(person, feet[k], float(heights[k]), float(widths[k]), t)
        self._prune(t)
        result = tracked[np.arange(len(tracked))]  # a copy
        result.tracker_id = np.array(out, dtype=int)
        return result

    def _match(self, new, feet, heights, widths, present, t) -> dict[int, int]:
        """Detection index -> person ID, for new tracker IDs that continue a vanished person."""
        pairs = []
        for k in new:
            for person, state in self._people.items():
                gap = t - state.t
                if person in present or not 0 < gap <= self.seconds:
                    continue
                ratio = heights[k] / state.height
                if 1 / (1 + self.size) <= ratio <= 1 + self.size:
                    expected = state.feet + state.velocity() * gap
                    off = float(np.hypot(*(feet[k] - expected))) / state.height
                    same = off <= self.distance
                elif self.top > 0 and self._as_wide(widths[k], state):
                    # the box changed: only the top of it can still say who this is
                    off = self._top_off(feet[k], float(heights[k]), state, gap)
                    same = off <= self.top
                else:
                    continue
                if same and self._looks_right(k, person):
                    pairs.append((off, k, person))
        matches: dict[int, int] = {}
        for _, k, person in sorted(pairs):
            if k not in matches and person not in matches.values():
                matches[k] = person
        return matches

    def _as_wide(self, width: float, state: _Person) -> bool:
        """Hidden legs change a box's height, not its width (pathsense.analytics.feet)."""
        return 1 / (1 + self.size) <= width / state.width <= 1 + self.size

    @staticmethod
    def _top_off(feet: np.ndarray, height: float, state: _Person, gap: float) -> float:
        """How far a box's top centre is from where this person's would be, in heights."""
        top = feet - np.array([0.0, height])
        last = state.feet - np.array([0.0, state.height])
        expected = last + state.top_velocity() * gap
        return float(np.hypot(*(top - expected))) / max(height, state.height)

    def _looks_right(self, k: int, person: int) -> bool:
        """Another reason to refuse the link; see AppearanceStitcher."""
        return True

    def _remember(
        self, person: int, feet: np.ndarray, height: float, width: float, t: float
    ) -> None:
        state = self._people.get(person)
        if state is None:
            state = self._people[person] = _Person(feet.copy(), height, width, t)
        state.feet, state.height, state.width, state.t = feet.copy(), height, width, t
        state.history.append((t, feet.copy(), height))
        while state.history and t - state.history[0][0] > HISTORY:
            state.history.popleft()

    def _prune(self, t: float) -> None:
        gone = [p for p, s in self._people.items() if t - s.t > self.seconds]
        for person in gone:
            del self._people[person]
        if gone:
            dead = set(gone)
            self._person = {k: v for k, v in self._person.items() if v not in dead}
