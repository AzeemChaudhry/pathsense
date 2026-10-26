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

if TYPE_CHECKING:
    import numpy as np
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
