"""Replay one scene's saved detections through one tracker setup and score the result.

The replay, the labels' own events and the crossing scoring are the package's
(pathsense.tune, pathsense.eval), the same code `pathsense tune` runs; this module adds
what only the study needs: named setups, the experimental guard, and saving every run.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from functools import cache
from pathlib import Path

import supervision as sv
from guard import IdentityGuard

from pathsense.analytics import Event
from pathsense.config import Config, TrackerConfig, load_config
from pathsense.eval import (
    Identity,
    ground_truth,
    label_events,
    match_crossings,
    tracking_metrics,
    write_mot,
)
from pathsense.pipeline import Pipeline
from pathsense.tune import NO_DETECTOR, SavedDetections
from pathsense.tune import replay as replay_detections
from pathsense.tune.cache import CACHE

DATA = Path("data/mot")
SCENES = Path("configs/scenes/mot")


@dataclasses.dataclass(frozen=True)
class Setup:
    """One way of tracking: a tracker with its settings, and our layers on top."""

    name: str
    tracker: str
    params: tuple = ()  # (key, value) pairs, hashable so setups can be dict keys
    stitch_seconds: float = 0.0
    bounce: float = 0.0
    guard: tuple | None = None  # (jump, size) or None

    def config(self, scene: str) -> Config:
        config = load_config(scene=SCENES / f"{scene}.yaml")
        tracker = TrackerConfig(self.tracker, dict(self.params), self.stitch_seconds)
        analytics = dataclasses.replace(config.analytics, line_bounce=self.bounce)
        return dataclasses.replace(config, tracker=tracker, analytics=analytics)

    def as_dict(self) -> dict:
        return {**dataclasses.asdict(self), "params": dict(self.params)}


@cache
def detections(scene: str) -> SavedDetections:
    """The scene's saved detections, made by `pathsense tune` or the study's cache step."""
    return SavedDetections.load(CACHE / f"{scene}.npz")


@cache
def labels(scene: str) -> dict[int, sv.Detections]:
    return ground_truth(DATA / "gt" / f"{scene}.txt")


@cache
def true_events(scene: str) -> tuple[Event, ...]:
    """The labelled people's line and zone events, through the same counters as ours."""
    saved = detections(scene)
    analytics = load_config(scene=SCENES / f"{scene}.yaml").analytics
    frames = zip(saved.index.tolist(), saved.timestamp.tolist(), strict=True)
    return tuple(label_events(analytics, labels(scene), frames, saved.fps))


def replay(setup: Setup, scene: str, out: Path | None = None, full: bool = True) -> dict:
    """Run `setup` on `scene` and return its metrics. With `out`, also save the tracks
    (MOT format), the events and the metrics there. `full=False` skips HOTA and the
    other tracking metrics, for quick sweeps."""
    saved = detections(scene)
    pipeline = Pipeline.from_config(setup.config(scene), saved.fps, detector=NO_DETECTOR)
    if setup.guard:
        guard, update = IdentityGuard(*setup.guard), pipeline.tracker.update
        pipeline.tracker.update = lambda dets, frame=None, timestamp=None: guard(
            update(dets, frame=frame, timestamp=timestamp), timestamp
        )
    result = replay_detections(pipeline, saved)
    identity = Identity()
    for index, people in result.tracks.items():
        identity.add(index, people, labels(scene).get(index))

    truth = true_events(scene)
    line = [e for e in result.events if e.type in ("enter", "exit")]
    score = match_crossings(
        line, [e for e in truth if e.type in ("enter", "exit")], identity, saved.fps
    )
    metrics: dict = {"setup": setup.name, "scene": scene, **score.as_dict()}
    metrics["macro_f1"] = (score.f1("enter") + score.f1("exit")) / 2
    for direction in ("enter", "exit"):
        metrics[f"{direction}_ours"] = sum(e.type == direction for e in line)
        metrics[f"{direction}_true"] = sum(e.type == direction for e in truth)
    zone_error = zone_true = 0
    for kind in ("zone_enter", "zone_exit"):
        for zone in {e.zone for e in truth if e.type == kind}:
            ours = sum(e.type == kind and e.zone == zone for e in result.events)
            true = sum(e.type == kind and e.zone == zone for e in truth)
            zone_error, zone_true = zone_error + abs(ours - true), zone_true + true
    metrics["zone_error"] = zone_error / max(zone_true, 1)
    metrics["track_ms"] = result.track_ms
    metrics["detect_ms"] = saved.detect_ms
    metrics["ids"] = result.ids
    metrics["bounced"] = pipeline.line_counter.bounced if pipeline.line_counter else 0

    tracks = sorted(result.tracks.items())
    if full or out:
        with tempfile.TemporaryDirectory() as tmp:
            folder = out or Path(tmp)
            folder.mkdir(parents=True, exist_ok=True)
            write_mot(folder / "tracks.txt", tracks)
            metrics |= tracking_metrics(DATA / "gt" / f"{scene}.txt", folder / "tracks.txt")
    if out:
        with open(out / "events.jsonl", "w", encoding="utf-8") as f:
            for event in sorted(result.events, key=lambda e: (e.t, e.frame)):
                f.write(json.dumps(event.to_dict()) + "\n")
        marks = [
            {"frame": e.frame, "id": e.track_id, "type": e.type, "mark": mark}
            for mark, group in (
                ("right", score.right),
                ("wrong", score.wrong),
                ("missed", score.missed),
            )
            for e in group
        ]
        (out / "crossings.json").write_text(json.dumps(marks))
        (out / "metrics.json").write_text(
            json.dumps(metrics | {"setup": setup.as_dict()}, indent=1)
        )
    return metrics
