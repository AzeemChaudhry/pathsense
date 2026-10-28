"""The one interface every detector implements.

Keeping this contract small is the key design decision of the project: the tracker,
analytics and app only ever see `supervision.Detections`, so YOLO26 and RF-DETR are
interchangeable through config.

The contract:
- input: one BGR frame (H, W, 3) uint8, as OpenCV reads it
- output: `sv.Detections` with `xyxy` in pixels, `confidence`, and `class_id` all equal to
  PERSON_CLASS_ID; nothing but people
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import supervision as sv

PERSON_CLASS_ID = 0  # what every detector reports for "person", whatever its model uses


class Detector(Protocol):
    """Detects people in a single BGR frame."""

    name: str

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Return person detections (xyxy, confidence, class_id) for one frame."""
        ...


def select_people(detections: sv.Detections, model_person_id: int) -> sv.Detections:
    """Keep only the model's person class and relabel it as PERSON_CLASS_ID.

    Models number classes differently: YOLO (COCO 80 classes) uses 0 for person, RF-DETR
    (COCO 91 category ids) uses 1. After this, every detector looks the same downstream.
    """
    people = detections[detections.class_id == model_person_id]
    people.class_id[:] = PERSON_CLASS_ID
    return people


def drop_partial_duplicates(
    detections: sv.Detections, width_overlap: float = 0.8, top: float = 0.1
) -> sv.Detections:
    """Drop a box that is another box's partial view of the same person.

    When something hides a person's legs (a shopping cart), a detector can return both their
    upper body and their whole body: two boxes with the same width and the same top, one
    inside the other. NMS keeps both (their IoU is about 0.5), and the tracker then follows
    a second, phantom person. A box is dropped when a taller box covers at least
    `width_overlap` of its width and both tops are within `top` of the taller box's height.
    Two people in a crowd rarely share both their width and their top, so they are kept.
    """
    if len(detections) < 2:
        return detections
    x1, y1, x2, y2 = detections.xyxy.T
    heights, widths = y2 - y1, np.maximum(x2 - x1, 1e-6)
    keep = np.ones(len(detections), bool)
    for i in range(len(detections)):
        for j in range(len(detections)):
            if i == j or heights[j] <= heights[i] or not keep[j]:
                continue
            shared = max(0.0, min(x2[i], x2[j]) - max(x1[i], x1[j])) / widths[i]
            if shared >= width_overlap and abs(y1[i] - y1[j]) <= top * heights[j]:
                keep[i] = False
                break
    return detections[keep]
