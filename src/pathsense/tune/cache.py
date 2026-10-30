"""Run the detector over a video once and keep every frame's boxes.

Tuning replays the same video through many tracker setups. The detector is by far the
slowest stage (38 to 57 ms a frame on the GTX 1650 at `imgsz` 1920, against 4 to 12 ms for
tracking), so it runs once and the boxes are saved:

    saved = cached_detections("data/samples/subway.mp4", config.detector)
    for frame, boxes in saved:          # frame.image is None: only the boxes were kept
        pipeline.track(boxes, frame)

The file remembers the detector settings it was made with; a change to any of them (model,
input size, confidence, tiling, dropping partial boxes) makes a new one. The device is not
part of it: a GPU and a CPU give the same boxes up to rounding.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

from pathsense.config import DetectorConfig
from pathsense.detect import Detector, build_detector
from pathsense.detect.base import drop_partial_duplicates
from pathsense.io import Frame, VideoSource

CACHE = Path("data/cache")
WARMUP = 5  # first frames left out of the detector time: model loading, CUDA start up


def detector_key(config: DetectorConfig) -> str:
    """The detector settings that change the boxes, as one comparable string."""
    settings = dataclasses.asdict(config)
    settings.pop("device")
    return json.dumps(settings, sort_keys=True)


@dataclass(frozen=True)
class SavedDetections:
    """Every frame's boxes from one video, in a few flat arrays (frame k's boxes are rows
    `starts[k]` to `starts[k + 1]` of `xyxy` and `confidence`)."""

    xyxy: np.ndarray  # (boxes, 4) float32, all frames one after another
    confidence: np.ndarray  # (boxes,) float32
    counts: np.ndarray  # (frames,) boxes in each frame
    index: np.ndarray  # (frames,) frame index, from 0
    timestamp: np.ndarray  # (frames,) seconds
    fps: float
    detect_ms: float = float("nan")  # median detector time per frame
    size: tuple[int, int] = (0, 0)  # frame width and height; (0, 0) if not known
    detector: str = ""  # detector_key() of the settings used; "" if not known

    @property
    def starts(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.counts)])

    def __len__(self) -> int:
        return len(self.counts)

    def __iter__(self) -> Iterator[tuple[Frame, sv.Detections]]:
        starts = self.starts
        for k, (index, t) in enumerate(zip(self.index, self.timestamp, strict=True)):
            rows = slice(starts[k], starts[k + 1])
            boxes = sv.Detections(
                xyxy=self.xyxy[rows],
                confidence=self.confidence[rows],
                class_id=np.zeros(starts[k + 1] - starts[k], int),
            )
            yield Frame(None, int(index), float(t)), boxes  # type: ignore[arg-type]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            xyxy=self.xyxy,
            conf=self.confidence,
            n=self.counts,
            frame=self.index,
            t=self.timestamp,
            fps=self.fps,
            detect_ms=self.detect_ms,
            size=np.array(self.size),
            detector=self.detector,
        )

    @classmethod
    def load(cls, path: Path | str) -> SavedDetections:
        with np.load(path) as d:
            return cls(
                xyxy=d["xyxy"],
                confidence=d["conf"],
                counts=d["n"],
                index=d["frame"],
                timestamp=d["t"],
                fps=float(d["fps"]),
                detect_ms=float(d["detect_ms"]) if "detect_ms" in d else float("nan"),
                size=tuple(int(v) for v in d["size"]) if "size" in d else (0, 0),
                detector=str(d["detector"]) if "detector" in d else "",
            )


def detect_video(
    video: Path | str, detector: Detector, drop_partial: bool = False, key: str = ""
) -> SavedDetections:
    """Run `detector` over every frame of `video`, as `Pipeline.process` would."""
    boxes, confidences, indexes, times, ms = [], [], [], [], []
    size = (0, 0)
    with VideoSource(str(video)) as source:
        fps = source.fps
        for frame in source:
            start = time.perf_counter()
            found = detector(frame.image)
            ms.append((time.perf_counter() - start) * 1000)
            if drop_partial:
                found = drop_partial_duplicates(found)
            boxes.append(found.xyxy.astype(np.float32))
            confidences.append(found.confidence.astype(np.float32))
            indexes.append(frame.index)
            times.append(frame.timestamp)
            size = (frame.image.shape[1], frame.image.shape[0])
    return SavedDetections(
        xyxy=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        confidence=np.concatenate(confidences) if confidences else np.zeros(0, np.float32),
        counts=np.array([len(b) for b in boxes], int),
        index=np.array(indexes, int),
        timestamp=np.array(times, float),
        fps=fps,
        detect_ms=float(np.median(ms[WARMUP:] or ms)) if ms else float("nan"),
        size=size,
        detector=key,
    )


def cached_detections(
    video: Path | str,
    config: DetectorConfig,
    folder: Path | str = CACHE,
    refresh: bool = False,
    build: Callable[[DetectorConfig], Detector] = build_detector,
) -> SavedDetections:
    """The saved detections for `video` with these detector settings, made if missing,
    made with other settings, or `refresh` is set. Kept in `folder/<video name>.npz`."""
    path = Path(folder) / f"{Path(video).stem}.npz"
    key = detector_key(config)
    if path.exists() and not refresh:
        saved = SavedDetections.load(path)
        if saved.detector == key:
            return saved
    saved = detect_video(video, build(config), config.drop_partial, key)
    saved.save(path)
    return saved
