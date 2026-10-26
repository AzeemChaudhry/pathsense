"""Draw tracked people on a frame: boxes, "#id" labels, trails and a stats line."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


class Annotator:
    def __init__(self, trail_length: int = 30) -> None:
        # Colour by track ID, not class (everyone is "person"), so an ID switch shows up as
        # a colour change.
        by_track = sv.ColorLookup.TRACK
        self.boxes = sv.BoxAnnotator(color_lookup=by_track)
        # smart_position keeps the label on screen when a close-up box touches the top edge.
        self.labels = sv.LabelAnnotator(
            color_lookup=by_track, text_position=sv.Position.TOP_LEFT, smart_position=True
        )
        self.trails = sv.TraceAnnotator(color_lookup=by_track, trace_length=trail_length)

    def draw(self, image: np.ndarray, tracked: sv.Detections, stats: str = "") -> np.ndarray:
        """Return an annotated copy of `image`; `tracked` must have tracker_id set."""
        labels = [f"#{tid}" for tid in tracked.tracker_id]
        out = self.trails.annotate(image.copy(), tracked)
        out = self.boxes.annotate(out, tracked)
        out = self.labels.annotate(out, tracked, labels=labels)
        if stats:
            draw_stats(out, stats)
        return out


def draw_stats(image: np.ndarray, text: str) -> None:
    """Bottom left, so it never covers the ID labels at the top. Thick black then thin green,
    so it stays readable on light and dark backgrounds."""
    origin = (10, image.shape[0] - 12)
    for color, thickness in (((0, 0, 0), 4), ((0, 255, 0), 2)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
