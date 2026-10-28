"""Glues the stages together, one frame at a time.

    frame -> Detector -> Tracker -> analytics -> FrameResult (people, events, timings)

Analytics turn tracks into events: a `LineCounter` when the config has `analytics.line`.
Drawing and display stay outside, in the CLI, so the same pipeline runs headless in tests,
evaluation and the app.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import supervision as sv

from pathsense.analytics import Event, LineCounter
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
    events: list[Event] = field(default_factory=list)  # what happened on this frame


class Pipeline:
    def __init__(
        self, detector: Detector, tracker: Tracker, line_counter: LineCounter | None = None
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.line_counter = line_counter  # running totals: line_counter.entered / .exited

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
        analytics = config.analytics
        line_counter = (
            LineCounter(
                *analytics.line, margin=analytics.line_margin, confirm=analytics.line_confirm
            )
            if analytics.line
            else None
        )
        return cls(detector, tracker, line_counter)

    def process(self, frame: Frame) -> FrameResult:
        start = time.perf_counter()
        detections = self.detector(frame.image)
        detected = time.perf_counter()
        tracked = self.tracker.update(detections, frame=frame.image, timestamp=frame.timestamp)
        done = time.perf_counter()
        events = (
            self.line_counter.update(tracked, frame.timestamp, frame.index)
            if self.line_counter
            else []
        )
        detect_ms, track_ms = (detected - start) * 1000, (done - detected) * 1000
        return FrameResult(frame, tracked, detect_ms, track_ms, events)

    def run(self, source: VideoSource) -> Iterator[FrameResult]:
        for frame in source:
            yield self.process(frame)
