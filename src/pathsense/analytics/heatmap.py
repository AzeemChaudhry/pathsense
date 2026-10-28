"""Where people spend time: a heatmap of person seconds on the floor.

The picture is split into square cells of `cell` pixels. Every frame, each tracked
person's feet (pathsense.analytics.feet) add the time since the previous frame to their
cell, so someone standing still for 10 s weighs ten times more than someone crossing in
1 s. `render` blurs the grid, colours it (dark red to yellow) and lays it over a frame;
cells nobody stood in stay transparent.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import supervision as sv

from pathsense.analytics.feet import Feet

MAX_STEP = 1.0  # seconds; a longer gap between frames (a stall) adds no more than this
FORGET = 10.0  # seconds; people unseen this long leave the feet memory


class Heatmap:
    def __init__(self, width: int, height: int, cell: int = 8, hidden_legs: float = 0.25) -> None:
        self.width, self.height, self.cell = width, height, cell
        self.seconds = np.zeros((math.ceil(height / cell), math.ceil(width / cell)), np.float32)
        self.feet = Feet(hidden_legs)
        self._last_t: float | None = None
        self._last_seen: dict[int, float] = {}

    def add(self, tracked: sv.Detections, t: float) -> None:
        step = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), MAX_STEP)
        self._last_t = t
        if tracked.tracker_id is None or len(tracked) == 0:
            return
        points, _ = self.feet(tracked, t)
        for track_id, (x, y) in zip(tracked.tracker_id, points, strict=True):
            self._last_seen[int(track_id)] = t
            if 0 <= x < self.width and 0 <= y < self.height:
                self.seconds[int(y // self.cell), int(x // self.cell)] += step
        for track_id, last in list(self._last_seen.items()):
            if t - last > FORGET:
                del self._last_seen[track_id]
                self.feet.forget(track_id)

    def render(self, background: np.ndarray, opacity: float = 0.6) -> np.ndarray:
        """The heatmap over `background` (same size as the video); a new image."""
        grid = cv2.GaussianBlur(self.seconds, (0, 0), sigmaX=2.0)
        image = background.copy()
        if grid.max() <= 0:
            return image
        level = grid / grid.max()
        level[level < 0.05] = 0  # the blur's faint tails stay transparent
        colours = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        size = (self.width, self.height)
        colours = cv2.resize(colours, size, interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(np.clip(level * 3, 0, 1), size)[..., None] * opacity
        return (image * (1 - alpha) + colours * alpha).astype(np.uint8)
