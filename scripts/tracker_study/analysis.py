"""Two checks on a finished study, from the saved runs (no replays).

misses(): for each missed crossing of one setup, was the person detected around the
crossing, and what did our tracks do on them? The two are crossed in one table
(pathsense.eval.misses, also used by `pathsense tune`).

rescore(): every saved run scored again with a stricter and a looser crossing matcher, to
see whether the ranking of the setups depends on how crossings are matched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import supervision as sv
from replay import detections, labels, true_events

from pathsense.analytics import Event
from pathsense.eval import Identity, explain_misses, match_crossings, misses_table
from pathsense.eval.misses import DETECT_LOOK, MATCH_IOU, TRACK_LOOK

# Matcher settings: (box overlap to tie a track to a label, look after the crossing in s,
# time window in s). The first is the one the study uses.
MATCHERS = ((0.3, 0.5, 2.0), (0.5, 0.5, 2.0), (0.3, 1.0, 2.0), (0.3, 0.5, 3.0))


def _tracks(folder: Path) -> dict[int, sv.Detections]:
    """A saved tracks.txt as frame index (from 0) -> tracked boxes."""
    rows = np.loadtxt(folder / "tracks.txt", delimiter=",", ndmin=2)
    frames = {}
    for frame in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == frame]
        xyxy = np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])
        frames[frame - 1] = sv.Detections(xyxy=xyxy, tracker_id=r[:, 1].astype(int))
    return frames


def _marks(folder: Path, mark: str) -> list[Event]:
    """Our crossings marked `mark` (right, wrong), or the true ones missed, as events."""
    marks = json.loads((folder / "crossings.json").read_text())
    return [Event(0.0, m["frame"], m["id"], m["type"]) for m in marks if m["mark"] == mark]


def misses(out: Path, setup: str, scenes: list[str]) -> str:
    """The missed crossings of `setup`, detected against tracked, as a Markdown table."""
    table = None
    for scene in scenes:
        folder = out / "runs" / setup / scene
        saved = detections(scene)
        found = {frame.index: boxes for frame, boxes in saved}
        counted = explain_misses(
            _marks(folder, "missed"),
            _marks(folder, "wrong"),
            labels(scene),
            found,
            _tracks(folder),
            saved.fps,
        )
        table = counted if table is None else table + counted
    assert table is not None
    return (
        f"Missed crossings of {setup}, {sum(table.values())} on {len(scenes)} scenes. Detected:"
        f" a box at IoU {MATCH_IOU} or more on the person, within {DETECT_LOOK} s of the"
        f" crossing. Tracked: the ID on the person in the {TRACK_LOOK} s before and after.\n\n"
        + misses_table(table)
    )


def _rescore(job: tuple[str, str, Path]) -> tuple[str, str, list[tuple[int, int, int]]]:
    setup, scene, folder = job
    fps = detections(scene).fps
    tracks = _tracks(folder)
    with open(folder / "events.jsonl", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    ours = [
        Event(e["t"], e["frame"], e["track_id"], e["type"])
        for e in events
        if e["type"] in ("enter", "exit")
    ]
    truth = [e for e in true_events(scene) if e.type in ("enter", "exit")]
    out = []
    for min_iou, look, window in MATCHERS:
        identity = Identity(min_iou)
        for frame, tracked in tracks.items():
            identity.add(frame, tracked, labels(scene).get(frame))
        s = match_crossings(ours, truth, identity, fps, window=window, look=look)
        out.append((sum(s.tp.values()), sum(s.fp.values()), sum(s.fn.values())))
    return setup, scene, out


def rescore(out: Path, setups: list[str], scenes: list[str], check: list[str], pool) -> str:
    """Crossing F1 of each setup under each matcher, as a Markdown table."""
    jobs = [(s, scene, out / "runs" / s / scene) for s in setups for scene in scenes]
    results = list(pool.map(_rescore, jobs))

    def f1(setup: str, group: list[str], k: int) -> float:
        tp, fp, fn = (
            sum(r[2][k][i] for r in results if r[0] == setup and r[1] in group) for i in range(3)
        )
        return 2 * tp / max(2 * tp + fp + fn, 1)

    head = " | ".join(f"IoU {m[0]}, look {m[1]} s, {m[2]} s" for m in MATCHERS)
    lines = [
        "Crossing F1 under each matcher: all scenes / check scenes. The first column is the"
        " matcher the study uses.",
        "",
        f"| setup | {head} |",
        "| --- |" + " --- |" * len(MATCHERS),
    ]
    for s in setups:
        cells = [f"{f1(s, scenes, k):.3f} / {f1(s, check, k):.3f}" for k in range(len(MATCHERS))]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
