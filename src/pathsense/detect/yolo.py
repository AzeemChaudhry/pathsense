"""Ultralytics YOLO adapter (YOLO26n by default). AGPL 3.0, see docs/DATASETS.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from pathsense.detect.base import select_people

COCO_PERSON_ID = 0  # Ultralytics COCO models: 0 is "person"
MAX_DET = 1000  # the library's 300 is near the box count of a loose NMS on a crowd


class YoloDetector:
    def __init__(
        self,
        model: Any,
        name: str = "yolo",
        imgsz: int = 640,
        conf: float = 0.1,
        device: str | None = None,
        person_id: int = COCO_PERSON_ID,
        nms: bool | None = None,
        iou: float | None = None,
    ) -> None:
        """Wrap an already loaded `ultralytics.YOLO` model. Use `load()` to build one."""
        self.model = model
        self.name = name
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.person_id = person_id
        # Only what is set goes to the library, so its own defaults apply otherwise.
        self.options = {k: v for k, v in (("nms", nms), ("iou", iou)) if v is not None}

    @classmethod
    def load(
        cls,
        weights: str,
        imgsz: int | None = None,
        conf: float = 0.1,
        device: str | None = None,
        nms: bool | None = None,
        iou: float | None = None,
    ) -> YoloDetector:
        from ultralytics import YOLO  # heavy import, only when a YOLO detector is built

        detector = cls(
            YOLO(weights),
            name=Path(weights).stem,
            imgsz=imgsz or 640,
            conf=conf,
            device=device,
            nms=nms,
            iou=iou,
        )
        detector.warmup()
        return detector

    def warmup(self) -> None:
        # The first GPU call pays for CUDA start up; do it now so it does not skew timing.
        self(np.zeros((self.imgsz, self.imgsz, 3), np.uint8))

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        result = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[self.person_id],  # let YOLO drop other classes before NMS
            device=self.device,
            max_det=MAX_DET,
            verbose=False,
            **self.options,
        )[0]
        return select_people(sv.Detections.from_ultralytics(result), self.person_id)
