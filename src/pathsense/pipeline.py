"""Glues the stages together, one frame at a time.

    frame -> Detector -> Tracker -> Stitcher -> analytics -> FrameResult (people, events)

The Stitcher gives people the tracker lost for a moment their old ID back, so analytics
see one person, not a new one each time they are hidden. With `appearance.enabled` an
Embedder runs between the detector and the tracker (pathsense.appearance; off by default).

Analytics turn tracks into events: a `LineCounter` when the config has `analytics.line`,
and a `ZoneCounter` for each of `analytics.zones`. Call `finish()` at the end of the video
for the events still waiting (crossings to confirm, dwell of people still inside).
Drawing and display stay outside, in the CLI, so the same pipeline runs headless in tests,
evaluation and the app.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import supervision as sv

from pathsense.analytics import Event, LineCounter, ZoneCounter
from pathsense.config import AnalyticsConfig, AppearanceConfig, Config, ConfigError
from pathsense.detect import DETECTORS, Detector, build_detector
from pathsense.detect.base import drop_partial_duplicates
from pathsense.io import Frame, VideoSource
from pathsense.track import TRACKERS, Tracker, build_tracker
from pathsense.track.stitch import Stitcher

if TYPE_CHECKING:
    from pathsense.appearance.embed import Embedder


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
        drop_partial: bool = False,
        embedder: Embedder | None = None,
    ) -> None:
        self.drop_partial = drop_partial
        self.embedder = embedder
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
        Pipeline.check_appearance(config)

    @staticmethod
    def check_appearance(config: Config) -> None:
        look = config.appearance
        if not look.enabled:
            return
        if config.tracker.name != "bytetrack":  # the only one tested to keep vectors on tracks
            raise ConfigError(
                f"appearance needs tracker.name bytetrack, not {config.tracker.name!r}"
            )
        if (look.line_same or look.line_different) and config.analytics.line_bounce <= 0:
            raise ConfigError("appearance.line_same and line_different need analytics.line_bounce")

    @classmethod
    def from_config(
        cls, config: Config, frame_rate: float = 30.0, detector: Detector | None = None
    ) -> Pipeline:
        """Build the detector and tracker the config names. `frame_rate` is the source's.
        Pass `detector` to skip building one, e.g. when replaying saved detections; with
        appearance on, those boxes then carry their vectors (`data["embedding"]`)."""
        cls.check_appearance(config)
        look = config.appearance
        embedder = None
        if look.enabled and detector is None:
            from pathsense.appearance.embed import Embedder

            embedder = Embedder(look.weights, look.device, look.batch)
        detector = build_detector(config.detector) if detector is None else detector
        tracker = build_tracker(
            config.tracker.name, frame_rate, config.tracker.params, tracker_factory(look)
        )
        analytics = config.analytics
        stitch = config.tracker.stitch_seconds
        settings = (stitch, config.tracker.stitch_distance, config.tracker.stitch_size)
        stitcher = None
        if stitch > 0 and look.enabled and look.stitch_gate > 0:
            from pathsense.appearance.stitch import AppearanceStitcher

            seconds, distance, size = settings
            stitcher = AppearanceStitcher(
                look.stitch_gate, look.memory, seconds=seconds, distance=distance, size=size
            )
        elif stitch > 0:
            stitcher = Stitcher(*settings)
        # Forget a person one frame after their ID can no longer return, from the tracker or
        # the stitcher.
        lost = max(tracker.lost_seconds, stitch) + 1 / frame_rate
        line_counter, zones = build_counters(analytics, lost, look)
        return cls(
            detector,
            tracker,
            line_counter,
            zones,
            stitcher,
            config.detector.drop_partial,
            embedder,
        )

    def process(self, frame: Frame) -> FrameResult:
        start = time.perf_counter()
        detections = self.detector(frame.image)
        if self.drop_partial:
            detections = drop_partial_duplicates(detections)
        detect_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if self.embedder:
            detections.data["embedding"] = self.embedder(frame.image, detections.xyxy)
        embed_ms = (time.perf_counter() - start) * 1000
        result = self.track(detections, frame)
        return dataclasses.replace(result, detect_ms=detect_ms, track_ms=result.track_ms + embed_ms)

    def track(self, detections: sv.Detections, frame: Frame) -> FrameResult:
        """Everything after the detector. Evaluation replays saved detections through it, so
        a replay runs the same code as a live video (`frame.image` may then be None).

        `detections` must already be through `drop_partial_duplicates` when the scene sets
        `detector.drop_partial`; `process` does that, and saved detections keep the setting
        they were saved with.
        Without an image, BoT-SORT cannot compensate camera motion."""
        start = time.perf_counter()
        tracked = self.tracker.update(detections, frame=frame.image, timestamp=frame.timestamp)
        if self.stitcher:
            tracked = self.stitcher(tracked, frame.timestamp)
        track_ms = (time.perf_counter() - start) * 1000
        events = []
        for counter in self.counters:
            events += counter.update(tracked, frame.timestamp, frame.index)
        tracked.data.pop("embedding", None)  # 512 numbers a box; the stages above are done
        return FrameResult(frame, tracked, 0.0, track_ms, events)

    def run(self, source: VideoSource) -> Iterator[FrameResult]:
        for frame in source:
            yield self.process(frame)

    @property
    def counters(self) -> list[LineCounter | ZoneCounter]:
        return ([self.line_counter] if self.line_counter else []) + self.zones

    def finish(self) -> list[Event]:
        """Events still waiting at the end of the video, from every counter."""
        return [event for counter in self.counters for event in counter.finish()]


def tracker_factory(look: AppearanceConfig) -> Callable[..., Any] | None:
    """What replaces the tracker class when looks join its matching; None: nothing does."""
    if not look.enabled or look.match == "off":
        return None
    from pathsense.appearance.tracker import AppearanceByteTrack

    return partial(
        AppearanceByteTrack,
        mode=look.match,
        closeness=look.closeness,
        proximity=look.proximity,
        blend=look.blend,
        gate=look.gate,
        memory=look.memory,
        adaptive_weight=look.adaptive_weight,
        adaptive_cap=look.adaptive_cap,
        dynamic=look.dynamic,
        skip_overlap=look.skip_overlap,
    )


def build_counters(
    analytics: AnalyticsConfig, lost: float, look: AppearanceConfig | None = None
) -> tuple[LineCounter | None, list[ZoneCounter]]:
    """The line counter and zone counters `analytics` asks for. `lost` is how long a person
    can be unseen before they are gone. Evaluation builds them the same way to count ground
    truth tracks."""
    line = LineCounter
    if look is not None and look.enabled and (look.line_same or look.line_different):
        from pathsense.appearance.line import AppearanceLine

        line = partial(
            AppearanceLine,
            same=look.line_same,
            different=look.line_different,
            memory=look.line_memory,
        )
    line_counter = (
        line(
            *analytics.line,
            margin=analytics.line_margin,
            margin_height=analytics.line_margin_height,
            confirm=analytics.line_confirm,
            lost=lost,
            hidden_legs=analytics.hidden_legs,
            bounce=analytics.line_bounce,
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
    return line_counter, zones
