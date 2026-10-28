"""Draw tracked people on a frame: boxes, "#id" labels, trails, a stats line, and the
counting line with its IN and OUT totals."""

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


LINE_COLOUR = (0, 200, 255)  # orange (BGR)
FLASH_COLOUR = (60, 200, 60)  # green, just after someone crossed
IN_COLOUR, OUT_COLOUR = (60, 200, 60), (60, 60, 230)
ZONE_COLOUR = (230, 160, 40)  # blue


def draw_line(
    image: np.ndarray,
    line: tuple[tuple[float, float], ...],
    entered: int,
    exited: int,
    flash: bool = False,
) -> None:
    """The counting line, two or more points (green and thicker for a moment after a
    crossing), and a top left box with the totals. Sizes grow with 4K frames."""
    k = _scale(image)
    colour, thickness = (FLASH_COLOUR, 6) if flash else (LINE_COLOUR, 3)
    points = np.array([[int(x), int(y)] for x, y in line], np.int32)
    cv2.polylines(image, [points], False, colour, int(thickness * k))
    cv2.rectangle(image, (10, 10), (int(330 * k), int(70 * k)), (0, 0, 0), -1)
    for text, x, text_colour in (
        (f"IN {entered}", 22, IN_COLOUR),
        (f"OUT {exited}", 165, OUT_COLOUR),
    ):
        origin = (int(x * k), int(55 * k))
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.3 * k, text_colour, int(3 * k))


def draw_zone(
    image: np.ndarray,
    name: str,
    polygon: tuple[tuple[float, float], ...],
    occupancy: int,
    entered: int,
    exited: int,
    flash: bool = False,
) -> None:
    """A zone: lightly filled, outlined (green for a moment after an enter or exit), with
    its name, how many are inside, and its enter and exit totals at its first point."""
    k = _scale(image)
    points = np.array([[int(x), int(y)] for x, y in polygon], np.int32)
    fill = image.copy()
    cv2.fillPoly(fill, [points], ZONE_COLOUR)
    cv2.addWeighted(fill, 0.15, image, 0.85, 0, dst=image)
    cv2.polylines(image, [points], True, FLASH_COLOUR if flash else ZONE_COLOUR, int(3 * k))
    text = f"{name}: {occupancy} inside  in {entered}  out {exited}"
    x, y = int(points[:, 0].min()), max(int(points[:, 1].min()) - int(10 * k), int(30 * k))
    for colour, thickness in (((0, 0, 0), int(5 * k)), (ZONE_COLOUR, int(2 * k))):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * k, colour, thickness)


def _scale(image: np.ndarray) -> float:
    """Drawing sizes are for 1920 px wide frames; a 4K frame gets them twice as large."""
    return max(1.0, max(image.shape[:2]) / 1920)


def draw_stats(image: np.ndarray, text: str) -> None:
    """Bottom left, so it never covers the ID labels at the top. Thick black then thin green,
    so it stays readable on light and dark backgrounds."""
    origin = (10, image.shape[0] - 12)
    for color, thickness in (((0, 0, 0), 4), ((0, 255, 0), 2)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
