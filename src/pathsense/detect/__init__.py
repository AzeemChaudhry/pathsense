"""Detectors. Every adapter returns `supervision.Detections` filtered to people.

`build_detector(config)` picks the adapter by name, so the rest of the code never imports
a model library directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pathsense.config import ConfigError
from pathsense.detect.base import PERSON_CLASS_ID, Detector, select_people

if TYPE_CHECKING:
    from pathsense.config import DetectorConfig


def _yolo(default_weights: str) -> Callable[[DetectorConfig], Detector]:
    def build(cfg: DetectorConfig) -> Detector:
        from pathsense.detect.yolo import YoloDetector

        weights = cfg.weights or default_weights
        return YoloDetector.load(
            weights, imgsz=cfg.imgsz, conf=cfg.conf, device=cfg.device, nms=cfg.nms, iou=cfg.iou
        )

    return build


def _rfdetr(size: str) -> Callable[[DetectorConfig], Detector]:
    def build(cfg: DetectorConfig) -> Detector:
        from pathsense.detect.rfdetr import RfDetrDetector

        return RfDetrDetector.load(
            size, weights=cfg.weights, imgsz=cfg.imgsz, conf=cfg.conf, device=cfg.device
        )

    return build


# Config name -> builder. Add a model here and it is available in every config.
DETECTORS: dict[str, Callable[[DetectorConfig], Detector]] = {
    "yolo26n": _yolo("yolo26n.pt"),
    "yolo26s": _yolo("yolo26s.pt"),
    "yolo26m": _yolo("yolo26m.pt"),
    "yolo26l": _yolo("yolo26l.pt"),
    "yolo26x": _yolo("yolo26x.pt"),
    "yolo11s": _yolo("yolo11s.pt"),
    "yolo11m": _yolo("yolo11m.pt"),
    "rfdetr-nano": _rfdetr("nano"),
    "rfdetr-small": _rfdetr("small"),
}


def build_detector(cfg: DetectorConfig) -> Detector:
    try:
        builder = DETECTORS[cfg.name]
    except KeyError:
        raise ConfigError(
            f"Unknown detector {cfg.name!r}; choose one of {sorted(DETECTORS)}"
        ) from None
    detector = builder(cfg)
    if cfg.slice:
        from pathsense.detect.slice import SlicedDetector

        detector = SlicedDetector(detector, cfg.slice)
    return detector


__all__ = ["DETECTORS", "PERSON_CLASS_ID", "Detector", "build_detector", "select_people"]
