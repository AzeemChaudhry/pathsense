"""Tracker wrapper around the Roboflow `trackers` library.

    tracker = Tracker("bytetrack")
    tracked = tracker.update(detections, frame=frame, timestamp=seconds)

All of them match boxes by motion and overlap only; none remembers what a person looks
like, so someone who leaves the frame for longer than `lost_track_buffer` comes back with a
new ID (NOTES.md, Phase 0). `pathsense.appearance` adds looks to ByteTrack, off by default.

| name      | idea                                                                    |
| --------- | ----------------------------------------------------------------------- |
| sort      | one matching pass on Kalman predicted boxes: the classic baseline       |
| bytetrack | two matching passes: confident boxes first, then low confidence ones    |
| botsort   | ByteTrack style matching plus camera motion compensation (uses frames)  |
| ocsort    | corrects the motion model after occlusions using the observed path      |
| cbiou     | BoT-SORT style passes on boxes enlarged by a buffer, for jerky motion   |

`params: {iou: giou}` swaps how box overlap is measured for the trackers that take it
(`iou`, `giou`, `diou`, `ciou`, `biou`; see IOU_MEASURES). docs/TRACKER_STUDY.md compares
all of them on MOT17 and MOT20.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import supervision as sv
from trackers import (
    BIoU,
    BoTSORTTracker,
    ByteTrackTracker,
    CBIoUTracker,
    CIoU,
    DIoU,
    GIoU,
    IoU,
    OCSORTTracker,
    SORTTracker,
)

from pathsense.config import ConfigError

TRACKERS = {
    "sort": SORTTracker,
    "bytetrack": ByteTrackTracker,
    "botsort": BoTSORTTracker,
    "ocsort": OCSORTTracker,
    "cbiou": CBIoUTracker,
}
IOU_MEASURES = {"iou": IoU, "giou": GIoU, "diou": DIoU, "ciou": CIoU, "biou": BIoU}
# Only BoT-SORT looks at the image (to estimate camera motion); the others warn if given one.
USES_FRAME = {"botsort"}


def accepted_params(name: str) -> set[str]:
    """Named settings of the tracker class (not *args or **kwargs, and not frame_rate, which
    the wrapper sets from the video)."""
    parameters = inspect.signature(TRACKERS[name].__init__).parameters.values()
    named = {p.name for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    return named - {"self", "frame_rate"}


def split_params(name: str, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The params this tracker takes, and those only another tracker takes.

    One config serves all trackers, so a setting such as `track_activation_threshold`
    (ByteTrack and BoT-SORT only) must not break `tracker.name=ocsort`; it is skipped with
    a warning. A name no tracker knows is a typo and stays an error.
    """
    if "frame_rate" in params:
        raise ConfigError("tracker.params cannot set frame_rate; it comes from the video")
    known = set().union(*(accepted_params(n) for n in TRACKERS))
    typos = sorted(set(params) - known)
    if typos:
        raise ConfigError(f"Unknown tracker params {typos}; known: {sorted(known)}")
    ours = accepted_params(name)
    return {k: v for k, v in params.items() if k in ours}, sorted(set(params) - ours)


class Tracker:
    def __init__(
        self,
        name: str = "bytetrack",
        frame_rate: float = 30.0,
        factory: Callable[..., Any] | None = None,
        **params: Any,
    ) -> None:
        """`frame_rate` scales the lost track budget; `params` go to the tracker class, or
        to `factory` when it replaces that class (a subclass with the same settings)."""
        if name not in TRACKERS:
            raise ConfigError(f"Unknown tracker {name!r}; choose one of {sorted(TRACKERS)}")
        self.name = name
        params, skipped = split_params(name, params)
        if skipped:
            warnings.warn(f"{name} does not use {skipped}; skipped", stacklevel=2)
        if "iou" in params:
            params["iou"] = iou_measure(params["iou"])
        self._tracker = (factory or TRACKERS[name])(frame_rate=frame_rate, **params)

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Match this frame's detections to existing tracks and return confirmed tracks.

        `timestamp` (seconds) makes the lost track budget real time, so a camera that
        delivers fewer frames than it claims still forgets people after the same time.
        Detections whose track is not confirmed yet (tracker_id -1) are dropped: a new
        person must be seen in a few frames in a row before getting an ID.
        """
        frame = frame if self.name in USES_FRAME else None
        tracked = self._tracker.update(detections, frame=frame, timestamp=timestamp)
        return tracked[tracked.tracker_id != -1]

    @property
    def lost_seconds(self) -> float:
        """How long a hidden person keeps their ID: `lost_track_buffer` frames at 30 FPS."""
        return float(getattr(self._tracker, "maximum_time_without_update", 1.0))

    def reset(self) -> None:
        """Forget every track, e.g. before processing a new video."""
        self._tracker.reset()


def iou_measure(name: Any):
    """`params.iou` from a config ("giou") -> the measure object the tracker takes."""
    if name not in IOU_MEASURES:
        raise ConfigError(f"Unknown tracker iou {name!r}; choose one of {sorted(IOU_MEASURES)}")
    return IOU_MEASURES[name]()


def build_tracker(
    name: str,
    frame_rate: float = 30.0,
    params: dict[str, Any] | None = None,
    factory: Callable[..., Any] | None = None,
):
    return Tracker(name, frame_rate=frame_rate, factory=factory, **(params or {}))


__all__ = ["IOU_MEASURES", "TRACKERS", "Tracker", "build_tracker"]
