"""Replay saved detections through a pipeline: everything after the detector, frame by frame.

    pipeline = Pipeline.from_config(config, saved.fps, detector=no_detector)
    result = replay(pipeline, saved)
    result.events, result.tracks[frame_index], result.track_ms

It calls `Pipeline.track`, the code a live video runs, so a replay gives the live result
(tests/test_pipeline.py checks this).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import supervision as sv

from pathsense.analytics import Event
from pathsense.pipeline import Pipeline
from pathsense.tune.cache import SavedDetections


def no_detector(image) -> sv.Detections:
    """Stands in for the detector when replaying: Pipeline.track never calls it."""
    raise RuntimeError("a replay passes saved detections to Pipeline.track")


@dataclass(frozen=True)
class Replay:
    events: list[Event]  # the event log, in the order it was produced
    tracks: dict[int, sv.Detections]  # frame index -> tracked people
    track_ms: float  # CPU time per frame from the tracker to the counters, one core

    @property
    def ids(self) -> int:
        """How many different track IDs were given out."""
        return len({int(i) for people in self.tracks.values() for i in people.tracker_id})


def replay(pipeline: Pipeline, saved: SavedDetections) -> Replay:
    tracks: dict[int, sv.Detections] = {}
    events: list[Event] = []
    seconds = 0.0
    for frame, boxes in saved:
        start = time.thread_time()  # this thread's CPU time: fair when replays run in parallel
        result = pipeline.track(boxes, frame)
        seconds += time.thread_time() - start
        tracks[frame.index] = result.tracked
        events += result.events
    events += pipeline.finish()
    return Replay(events, tracks, seconds * 1000 / max(len(saved), 1))


__all__ = ["no_detector", "Replay", "replay"]
