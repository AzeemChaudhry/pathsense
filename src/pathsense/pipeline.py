"""Glues the stages together, one frame at a time.

    frame -> Detector -> Tracker -> FrameResult (tracked people + timings)

Phase 2 adds analytics after the tracker. Drawing and display stay outside, in the CLI, so
the same pipeline runs headless in tests, evaluation and the app.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import supervision as sv

from pathsense.config import Config, ConfigError
from pathsense.detect import DETECTORS, Detector, build_detector
from pathsense.io import Frame, VideoSource
from pathsense.track import TRACKERS, Tracker, build_tracker


@dataclass(frozen=True)
class FrameResult:
    frame: Frame
    tracked: sv.Detections  # people with confirmed track IDs (tracker_id)
    detect_ms: float
    track_ms: float


class Pipeline:
    def __init__(self, detector: Detector, tracker: Tracker) -> None:
        self.detector = detector
        self.tracker = tracker

    @staticmethod
    def check(config: Config) -> None:
        """Raise ConfigError for an unknown detector or tracker before anything is opened or
        downloaded, so a typo is reported straight away, not after the camera starts."""
        if config.detector.name not in DETECTORS:
            raise ConfigError(
                f"Unknown detector {config.detector.name!r}; choose one of {sorted(DETECTORS)}"
            )
        if config.tracker.name not in TRACKERS:
            raise ConfigError(
                f"Unknown tracker {config.tracker.name!r}; choose one of {sorted(TRACKERS)}"
            )

    @classmethod
    def from_config(cls, config: Config, frame_rate: float = 30.0) -> Pipeline:
        """Build the detector and tracker the config names. `frame_rate` is the source's."""
        detector = build_detector(config.detector)
        tracker = build_tracker(config.tracker.name, frame_rate, config.tracker.params)
        return cls(detector, tracker)

    def process(self, frame: Frame) -> FrameResult:
        start = time.perf_counter()
        detections = self.detector(frame.image)
        detected = time.perf_counter()
        tracked = self.tracker.update(detections, frame=frame.image, timestamp=frame.timestamp)
        done = time.perf_counter()
        return FrameResult(frame, tracked, (detected - start) * 1000, (done - detected) * 1000)

    def run(self, source: VideoSource) -> Iterator[FrameResult]:
        for frame in source:
            yield self.process(frame)
