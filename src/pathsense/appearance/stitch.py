"""The Stitcher, refusing a link when the two people do not look alike."""

from __future__ import annotations

import numpy as np
import supervision as sv

from pathsense.track.stitch import Stitcher


class AppearanceStitcher(Stitcher):
    def __init__(self, gate: float, memory: float = 0.9, **settings) -> None:
        super().__init__(**settings)
        self.gate, self.memory = gate, memory
        self._looks: dict[int, np.ndarray] = {}  # person id -> remembered look
        self._vectors: np.ndarray | None = None  # this frame's

    def __call__(self, tracked: sv.Detections, t: float) -> sv.Detections:
        self._vectors = tracked.data.get("embedding") if len(tracked) else None
        result = super().__call__(tracked, t)
        if self._vectors is not None and result.tracker_id is not None:
            for person, vector in zip(result.tracker_id, self._vectors, strict=True):
                old = self._looks.get(int(person))
                mixed = vector if old is None else self.memory * old + (1 - self.memory) * vector
                self._looks[int(person)] = mixed / (np.linalg.norm(mixed) + 1e-12)
        return result

    def _looks_right(self, k: int, person: int) -> bool:
        old = self._looks.get(person)
        if self._vectors is None or old is None:
            return True
        return float(old @ self._vectors[k]) >= self.gate

    def _prune(self, t: float) -> None:
        super()._prune(t)
        if len(self._looks) > len(self._people):
            self._looks = {p: v for p, v in self._looks.items() if p in self._people}
