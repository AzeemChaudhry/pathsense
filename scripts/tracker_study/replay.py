"""Replay one scene's saved detections through one tracker setup and score the result."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import time
from functools import cache
from pathlib import Path

import numpy as np
import supervision as sv
from guard import IdentityGuard

from pathsense.analytics import Event
from pathsense.config import Config, TrackerConfig, load_config
from pathsense.eval import Identity, ground_truth, match_crossings, tracking_metrics, write_mot
from pathsense.io import Frame
from pathsense.pipeline import Pipeline, build_counters

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
def detections(scene: str) -> dict:
    with np.load(DATA / "dets" / f"{scene}.npz") as d:
        return {key: d[key] for key in d.files}


@cache
def labels(scene: str) -> dict[int, sv.Detections]:
    return ground_truth(DATA / "gt" / f"{scene}.txt")


@cache
def true_events(scene: str) -> tuple[Event, ...]:
    """The labelled people's line and zone events, through the same counters as ours."""
    config = load_config(scene=SCENES / f"{scene}.yaml")
    analytics = dataclasses.replace(config.analytics, hidden_legs=0.0, line_bounce=0.0)
    fps = float(detections(scene)["fps"])
    line, zones = build_counters(analytics, lost=1.0 + 1 / fps)
    counters = ([line] if line else []) + zones
    empty = sv.Detections.empty()
    empty.tracker_id = np.array([], int)
    events: list[Event] = []
    for frame, t in zip(detections(scene)["frame"], detections(scene)["t"], strict=True):
        people = labels(scene).get(int(frame), empty)
        events += [e for c in counters for e in c.update(people, float(t), int(frame))]
    return tuple(events + [e for c in counters for e in c.finish()])


def replay(setup: Setup, scene: str, out: Path | None = None, full: bool = True) -> dict:
    """Run `setup` on `scene` and return its metrics. With `out`, also save the tracks
    (MOT format), the events and the metrics there. `full=False` skips HOTA and the
    other tracking metrics, for quick sweeps."""
    d = detections(scene)
    fps = float(d["fps"])
    config = setup.config(scene)
    pipeline = Pipeline.from_config(config, frame_rate=fps, detector=lambda image: None)
    if setup.guard:
        guard, update = IdentityGuard(*setup.guard), pipeline.tracker.update
        pipeline.tracker.update = lambda dets, frame=None, timestamp=None: guard(
            update(dets, frame=frame, timestamp=timestamp), timestamp
        )
    identity = Identity()
    starts = np.concatenate([[0], np.cumsum(d["n"])])
    tracks, events, track_ms = [], [], 0.0
    for k, (frame, t) in enumerate(zip(d["frame"], d["t"], strict=True)):
        s, e = starts[k], starts[k + 1]
        boxes = sv.Detections(
            xyxy=d["xyxy"][s:e], confidence=d["conf"][s:e], class_id=np.zeros(e - s, int)
        )
        start = time.thread_time()
        result = pipeline.track(boxes, Frame(None, int(frame), float(t)))
        track_ms += (time.thread_time() - start) * 1000
        events += result.events
        identity.add(int(frame), result.tracked, labels(scene).get(int(frame)))
        tracks.append((int(frame), result.tracked))
    events += pipeline.finish()

    truth = true_events(scene)
    line = [e for e in events if e.type in ("enter", "exit")]
    score = match_crossings(line, [e for e in truth if e.type in ("enter", "exit")], identity, fps)
    metrics: dict = {"setup": setup.name, "scene": scene, **score.as_dict()}
    metrics["macro_f1"] = (score.f1("enter") + score.f1("exit")) / 2
    for direction in ("enter", "exit"):
        metrics[f"{direction}_ours"] = sum(e.type == direction for e in line)
        metrics[f"{direction}_true"] = sum(e.type == direction for e in truth)
    zone_error = zone_true = 0
    for kind in ("zone_enter", "zone_exit"):
        for zone in {e.zone for e in truth if e.type == kind}:
            ours = sum(e.type == kind and e.zone == zone for e in events)
            true = sum(e.type == kind and e.zone == zone for e in truth)
            zone_error, zone_true = zone_error + abs(ours - true), zone_true + true
    metrics["zone_error"] = zone_error / max(zone_true, 1)
    metrics["track_ms"] = track_ms / len(d["frame"])
    metrics["detect_ms"] = float(d.get("detect_ms", np.nan))
    metrics["ids"] = len({int(i) for _, t in tracks for i in t.tracker_id})
    metrics["bounced"] = pipeline.line_counter.bounced if pipeline.line_counter else 0

    if full or out:
        with tempfile.TemporaryDirectory() as tmp:
            folder = out or Path(tmp)
            folder.mkdir(parents=True, exist_ok=True)
            write_mot(folder / "tracks.txt", tracks)
            metrics |= tracking_metrics(DATA / "gt" / f"{scene}.txt", folder / "tracks.txt")
    if out:
        with open(out / "events.jsonl", "w", encoding="utf-8") as f:
            for event in sorted(events, key=lambda e: (e.t, e.frame)):
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
