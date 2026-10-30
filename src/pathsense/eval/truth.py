"""The labelled people's own events: what a perfect tracker would have produced.

The labels go through the same line and zone counters as our tracks, so a true crossing
means exactly what one of ours means. Two of our rules are turned off for them: hidden legs
(a label box is the whole person, legs included, even when they are hidden) and the bounce
rule (a label ID never swaps, so a crossing back is real).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
import supervision as sv

from pathsense.analytics import Event
from pathsense.config import AnalyticsConfig
from pathsense.pipeline import build_counters


def label_events(
    analytics: AnalyticsConfig,
    labels: dict[int, sv.Detections],
    frames: Iterable[tuple[int, float]],
    fps: float,
    rules: bool = False,
) -> list[Event]:
    """Events of the labelled people over (frame index, time) pairs.

    With `rules`, the scene's own hidden legs and bounce settings stay on: the events our
    counting rules give with perfect tracking, to tell their errors from the tracker's.
    """
    if not rules:
        analytics = dataclasses.replace(analytics, hidden_legs=0.0, line_bounce=0.0)
    line, zones = build_counters(analytics, lost=1.0 + 1 / fps)
    counters = ([line] if line else []) + zones
    nobody = sv.Detections.empty()
    nobody.tracker_id = np.array([], int)
    events: list[Event] = []
    for index, t in frames:
        people = labels.get(index, nobody)
        events += [e for c in counters for e in c.update(people, t, index)]
    return events + [e for c in counters for e in c.finish()]


__all__ = ["label_events"]
