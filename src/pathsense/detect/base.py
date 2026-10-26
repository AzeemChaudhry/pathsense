"""The one interface every detector implements.

Keeping this contract small is the key design decision of the project: the tracker,
analytics and app only ever see `supervision.Detections`, so YOLO26 and RF-DETR are
interchangeable through config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import supervision as sv


class Detector(Protocol):
    """Detects people in a single BGR frame."""

    name: str

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Return person detections (xyxy, confidence, class_id) for one frame."""
        ...


# Phase 1 TODO: add YoloDetector (ultralytics) and RfDetrDetector (rfdetr) adapters here,
# each loading weights from the Hugging Face Hub and keeping only the person class.
