"""Run the detector over a video once and keep every frame's boxes.

Tuning replays the same video through many tracker setups. The detector is by far the
slowest stage (38 to 57 ms a frame on the GTX 1650 at `imgsz` 1920, against 4 to 12 ms for
tracking), so it runs once and the boxes are saved:

    saved = cached_detections("data/samples/subway.mp4", config.detector)
    for frame, boxes in saved:          # frame.image is None: only the boxes were kept
        pipeline.track(boxes, frame)

The file remembers what it was made from: the video (its full path, size and time
modified) and the detector settings (model, input size, confidence, tiling, dropping
partial boxes, and the weights file when it is a local one). A change to any of them makes
a new one. Two limits: pretrained weights the model library downloads are known by name
only, and moving the repo or touching the video also makes a new one (a full detector
pass). The device is not part of it: a GPU and a CPU give the same boxes up to rounding.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

from pathsense.config import ConfigError, DetectorConfig
from pathsense.detect import Detector, build_detector
from pathsense.detect.base import drop_partial_duplicates
from pathsense.io import Frame, VideoSource

CACHE = Path("data/cache")
WARMUP = 5  # first frames left out of the detector time: model loading, CUDA start up


def _stamp(path: Path | str) -> str:
    """A file's full path, size and time modified: changes when the file is replaced."""
    path = Path(path).resolve()
    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def detector_key(config: DetectorConfig, video: Path | str | None = None) -> str:
    """What the boxes were made from, as one comparable string: the detector settings
    that change them, the weights file if it is a local one, and the video."""
    settings: dict = dataclasses.asdict(config)
    settings.pop("device")
    if config.weights and Path(config.weights).is_file():
        settings["weights_file"] = _stamp(config.weights)  # retrained in place: new boxes
    if video is not None:
        if not Path(video).is_file():
            raise ConfigError(f"video not found: {video}")
        settings["video"] = _stamp(video)
    return json.dumps(settings, sort_keys=True)


def cache_path(video: Path | str, folder: Path | str = CACHE) -> Path:
    """Where a video's saved detections go: named after the video, and its full path so two
    videos with the same name in different folders do not share a file."""
    resolved = str(Path(video).resolve()).encode()
    return Path(folder) / f"{Path(video).stem}-{hashlib.sha1(resolved).hexdigest()[:8]}.npz"


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
        """Write to a temporary file, then rename: an interrupted run leaves no half file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        try:
            with os.fdopen(handle, "wb") as out:
                self._write(out)
            os.chmod(temporary, 0o644)  # mkstemp makes it private to its owner
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _write(self, out) -> None:
        np.savez(
            out,
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
    made from another video or other settings, or `refresh` is set (cache_path())."""
    path = cache_path(video, folder)
    key = detector_key(config, video)
    if path.exists() and not refresh:
        saved = SavedDetections.load(path)
        if saved.detector == key:
            return saved
    saved = detect_video(video, build(config), config.drop_partial, key)
    saved.save(path)
    return saved
