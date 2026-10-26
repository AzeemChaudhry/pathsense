"""Roboflow RF-DETR adapter. Apache 2.0, the commercial track (docs/DATASETS.md).

Install with `uv sync --extra rfdetr`. Measured on a GTX 1650 (NOTES.md, Phase 1): keep
FP32. FP16 is about 4x slower on that card (no tensor cores), and compiling saves about
1 ms per frame for a long start up, so neither is enabled.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import supervision as sv

from pathsense.detect.base import select_people

COCO_PERSON_ID = 1  # RF-DETR reports COCO category ids, where 1 is "person"
SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


class RfDetrDetector:
    def __init__(
        self,
        model: Any,
        name: str = "rfdetr",
        conf: float = 0.1,
        person_id: int = COCO_PERSON_ID,
    ) -> None:
        """Wrap an already loaded `rfdetr.RFDETR*` model. Use `load()` to build one."""
        self.model = model
        self.name = name
        self.conf = conf
        self.person_id = person_id

    @classmethod
    def load(
        cls,
        size: str = "nano",
        weights: str | None = None,
        imgsz: int | None = None,
        conf: float = 0.1,
        device: str | None = None,
    ) -> RfDetrDetector:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "`torch.jit.script` is deprecated")
                import rfdetr  # heavy import, only when an RF-DETR detector is built
        except ImportError:
            raise ImportError("RF-DETR is an optional extra: uv sync --extra rfdetr") from None
        # RF-DETR logs start up warnings that do not apply here (backbone weights it does not
        # need, and advice to use FP16, which is slower on GPUs without tensor cores). Set
        # after the import, which configures the logger and would reset the level.
        logging.getLogger("rf-detr").setLevel(logging.ERROR)

        kwargs: dict[str, Any] = {}
        if weights:
            kwargs["pretrain_weights"] = weights
        if imgsz:
            kwargs["resolution"] = imgsz  # must be a multiple of 32
        if device:
            kwargs["device"] = device
        model = getattr(rfdetr, SIZES[size])(**kwargs)
        detector = cls(model, name=f"rfdetr-{size}", conf=conf)
        detector(np.zeros((480, 640, 3), np.uint8))  # warm up, as for YOLO
        return detector

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        # OpenCV frames are BGR; RF-DETR expects RGB. [..., ::-1] flips the channel order.
        rgb = np.ascontiguousarray(frame[..., ::-1])
        detections = self.model.predict(rgb, threshold=self.conf, include_source_image=False)
        return select_people(detections, self.person_id)
