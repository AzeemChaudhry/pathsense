"""Glues the stages together, one frame at a time.

    frame -> Detector -> Tracker -> Stitcher -> analytics -> FrameResult (people, events)

The Stitcher gives people the tracker lost for a moment their old ID back, so analytics
see one person, not a new one each time they are hidden.

Analytics turn tracks into events: a `LineCounter` when the config has `analytics.line`,
and a `ZoneCounter` for each of `analytics.zones`. Call `finish()` at the end of the video
for the events still waiting (crossings to confirm, dwell of people still inside).
Drawing and display stay outside, in the CLI, so the same pipeline runs headless in tests,
evaluation and the app.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import supervision as sv

from pathsense.analytics import Event, LineCounter, ZoneCounter
from pathsense.config import Config, ConfigError
from pathsense.detect import DETECTORS, Detector, build_detector
from pathsense.io import Frame, VideoSource
from pathsense.track import TRACKERS, Tracker, build_tracker
from pathsense.track.stitch import Stitcher


@dataclass(frozen=True)
class FrameResult:
    frame: Frame
    tracked: sv.Detections  # people with confirmed track IDs (tracker_id)
    detect_ms: float
    track_ms: float
    events: list[Event] = field(default_factory=list)  # what happened on this frame


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        line_counter: LineCounter | None = None,
        zones: list[ZoneCounter] | None = None,
        stitcher: Stitcher | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.stitcher = stitcher
        self.line_counter = line_counter  # running totals: line_counter.entered / .exited
        self.zones = zones or []  # each with .entered, .exited, .occupancy

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
        stitch = config.tracker.stitch_seconds
        stitcher = (
            Stitcher(stitch, config.tracker.stitch_distance, config.tracker.stitch_size)
            if stitch > 0
            else None
        )
        # Forget a person one frame after their ID can no longer return, from the tracker or
        # the stitcher.
        lost = max(tracker.lost_seconds, stitch) + 1 / frame_rate
        line_counter = (
            LineCounter(
                *analytics.line,
                margin=analytics.line_margin,
                margin_height=analytics.line_margin_height,
                confirm=analytics.line_confirm,
                lost=lost,
                hidden_legs=analytics.hidden_legs,
            )
            if analytics.line
            else None
        )
        zones = [
            ZoneCounter(
                zone.name,
                list(zone.polygon),
                confirm=analytics.zone_confirm,
                lost=lost,
                count_present=analytics.zone_count_present,
                hidden_legs=analytics.hidden_legs,
            )
            for zone in analytics.zones
        ]
        return cls(detector, tracker, line_counter, zones, stitcher)

    def process(self, frame: Frame) -> FrameResult:
        start = time.perf_counter()
        detections = self.detector(frame.image)
        detected = time.perf_counter()
        tracked = self.tracker.update(detections, frame=frame.image, timestamp=frame.timestamp)
        if self.stitcher:
            tracked = self.stitcher(tracked, frame.timestamp)
        done = time.perf_counter()
        events = []
        for counter in self.counters:
            events += counter.update(tracked, frame.timestamp, frame.index)
        detect_ms, track_ms = (detected - start) * 1000, (done - detected) * 1000
        return FrameResult(frame, tracked, detect_ms, track_ms, events)

    def run(self, source: VideoSource) -> Iterator[FrameResult]:
        for frame in source:
            yield self.process(frame)

    @property
    def counters(self) -> list[LineCounter | ZoneCounter]:
        return ([self.line_counter] if self.line_counter else []) + self.zones

    def finish(self) -> list[Event]:
        """Events still waiting at the end of the video, from every counter."""
        return [event for counter in self.counters for event in counter.finish()]
