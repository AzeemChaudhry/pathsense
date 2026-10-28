"""Where each person stands: their feet, even when something hides their legs.

The bottom centre of a person's box is where they stand only while their legs are
visible. A shopping cart, a bench or another person in front hides the legs; the detector
then boxes the visible upper body, the bottom of the box moves up to the cart handle or
the waist, and a line or zone sees the person jump. On the grocery-store clip one shopper
walking past a line once was counted IN, OUT, IN, OUT, IN this way (NOTES.md).

The head is rarely hidden from a raised camera, so the top of the box stays reliable.
`Feet` remembers each person's full height from frames where they look whole (their
tallest box in the last `window` seconds). When a box is more than `hidden_legs` shorter
than that, the legs are taken as hidden and the feet are placed at top + remembered
height. During a long occlusion the last full height is kept, so a cart that hides the
legs for many seconds is still handled. `hidden_legs=0` turns the estimate off.

Someone who sits or crouches also gets a shorter box; their feet are then placed lower
than they are. A top down camera barely changes box heights, so the estimate rarely acts
there.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import supervision as sv


def anchor_points(tracked: sv.Detections) -> np.ndarray:
    """Bottom centre of each box, shape (n, 2)."""
    xyxy = tracked.xyxy
    return np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]))


class Feet:
    def __init__(self, hidden_legs: float = 0.25, window: float = 2.0) -> None:
        if not 0.0 <= hidden_legs < 1.0:
            raise ValueError("hidden_legs must be from 0 up to (not including) 1")
        self.hidden_legs, self.window = hidden_legs, window
        self._whole: dict[int, deque[tuple[float, float]]] = {}  # (t, height) of whole frames
        self._height: dict[int, float] = {}  # remembered full height

    def __call__(self, tracked: sv.Detections, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Feet points (n, 2) and full body heights (n,) for each tracked person."""
        points = anchor_points(tracked)
        heights = tracked.xyxy[:, 3] - tracked.xyxy[:, 1]
        if tracked.tracker_id is None or self.hidden_legs == 0:
            return points, heights
        full = heights.copy()
        for i, track_id in enumerate(int(tid) for tid in tracked.tracker_id):
            height = float(heights[i])
            remembered = self._height.get(track_id)
            if remembered is not None and height < (1 - self.hidden_legs) * remembered:
                points[i, 1] = tracked.xyxy[i, 1] + remembered  # legs hidden: from the head
                full[i] = remembered
                continue
            whole = self._whole.setdefault(track_id, deque())
            whole.append((t, height))
            while whole and t - whole[0][0] > self.window:
                whole.popleft()
            self._height[track_id] = max(h for _, h in whole)
        return points, full

    def forget(self, track_id: int) -> None:
        self._whole.pop(track_id, None)
        self._height.pop(track_id, None)
