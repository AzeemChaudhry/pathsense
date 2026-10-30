"""The bounce rule decided by looks: is the ID still the person who last crossed?

Compares how the ID looks now with how it looked at its last counted crossing.

- `same` above 0: a held reversal that scores this or more is one person dithering, so
  every crossing counts.
- `different` above 0: a reversal that scores below this is an ID that moved to someone
  else, so it is dropped at once, without waiting.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from pathsense.analytics import Event, LineCounter


class AppearanceLine(LineCounter):
    def __init__(
        self, *points, same: float = 0.0, different: float = 0.0, memory: float = 0.8, **settings
    ) -> None:
        super().__init__(*points, **settings)
        self.same, self.different, self.memory = same, different, memory
        self._looks: dict[int, np.ndarray] = {}  # track id -> look now
        self._at_crossing: dict[int, np.ndarray] = {}  # track id -> look at its last crossing
        self.kept = 0  # reversals counted as a person dithering
        self.dropped = 0  # reversals dropped as an ID moving

    def update(self, tracked: sv.Detections, t: float, frame: int) -> list[Event]:
        vectors = tracked.data.get("embedding") if len(tracked) else None
        if vectors is not None and tracked.tracker_id is not None:
            for track_id, vector in zip(tracked.tracker_id, vectors, strict=True):
                old = self._looks.get(int(track_id))
                mixed = vector if old is None else self.memory * old + (1 - self.memory) * vector
                self._looks[int(track_id)] = mixed / (np.linalg.norm(mixed) + 1e-12)
        return super().update(tracked, t, frame)

    def _alike(self, track_id: int) -> float | None:
        then, now = self._at_crossing.get(track_id), self._looks.get(track_id)
        return None if then is None or now is None else float(then @ now)

    def _same_person(self, track_id: int) -> bool:
        alike = self._alike(track_id)
        keep = self.same > 0 and alike is not None and alike >= self.same
        self.kept += keep
        return keep

    def _other_person(self, track_id: int) -> bool:
        alike = self._alike(track_id)
        drop = self.different > 0 and alike is not None and alike < self.different
        self.dropped += drop
        return drop

    def _counted(self, track_id: int) -> None:
        look = self._looks.get(track_id)
        if look is not None:
            self._at_crossing[track_id] = look

    def _forget(self, track_id: int) -> list[Event]:
        events = super()._forget(track_id)  # may count a last crossing, which needs the looks
        self._looks.pop(track_id, None)
        self._at_crossing.pop(track_id, None)
        return events
