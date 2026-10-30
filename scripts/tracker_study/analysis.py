"""Two checks on a finished study, from the saved runs (no replays).

misses(): for each missed crossing of one setup, was the person detected around the
crossing, and what did our tracks do on them? The two are crossed in one table.

rescore(): every saved run scored again with a stricter and a looser crossing matcher, to
see whether the ranking of the setups depends on how crossings are matched.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import supervision as sv
from replay import detections, labels, true_events

from pathsense.analytics import Event
from pathsense.eval import Identity, match_crossings

MATCH_IOU = 0.5  # a detection or track box is on the person at this overlap with the label
DETECT_LOOK = 0.5  # seconds either side of the crossing, for "was the person detected"
TRACK_LOOK = 1.0  # seconds before and after the crossing, for "which ID was on them"

DETECTED = ("in at least half the frames", "in some frames", "never")
TRACKED = (
    "no track on them",
    "one ID before and after, no matching crossing",
    "track starts or ends at the line",
    "ID changes at the line",
    "swap: our ID crossed, scored on someone else",
)

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


def _box(people: sv.Detections | None, person: int) -> np.ndarray | None:
    if people is None or person not in people.tracker_id:
        return None
    return people.xyxy[list(people.tracker_id).index(person)]


def _best(box: np.ndarray, boxes: np.ndarray) -> int | None:
    """Index of the box overlapping `box` most, if by at least MATCH_IOU."""
    if not len(boxes):
        return None
    overlap = sv.box_iou_batch(box[None], boxes)[0]
    return int(overlap.argmax()) if overlap.max() >= MATCH_IOU else None


def _detected(scene: str, person: int, frame: int) -> str:
    d, fps = detections(scene), float(detections(scene)["fps"])
    starts = np.concatenate([[0], np.cumsum(d["n"])])
    index = {int(f): k for k, f in enumerate(d["frame"])}
    look = round(DETECT_LOOK * fps)
    seen = shown = 0
    for f in range(frame - look, frame + look + 1):
        box, k = _box(labels(scene).get(f), person), index.get(f)
        if box is None or k is None:
            continue
        shown += 1
        seen += _best(box, d["xyxy"][starts[k] : starts[k + 1]]) is not None
    if seen and seen >= shown / 2:
        return DETECTED[0]
    return DETECTED[1] if seen else DETECTED[2]


def _tracked(scene, tracks, person, frame, kind, wrong) -> str:
    look = round(TRACK_LOOK * float(detections(scene)["fps"]))

    def on(frames: range) -> int | None:
        ids = []
        for f in frames:
            box, ours = _box(labels(scene).get(f), person), tracks.get(f)
            if box is not None and ours is not None:
                i = _best(box, ours.xyxy)
                if i is not None:
                    ids.append(int(ours.tracker_id[i]))
        return Counter(ids).most_common(1)[0][0] if ids else None

    before, after = on(range(frame - look, frame)), on(range(frame, frame + look + 1))
    fps = float(detections(scene)["fps"])
    if before is None and after is None:
        return TRACKED[0]
    if any(
        e["type"] == kind and abs(e["frame"] - frame) <= 2 * fps and e["id"] in (before, after)
        for e in wrong
    ):
        return TRACKED[4]
    if before is None or after is None:
        return TRACKED[2]
    return TRACKED[3] if before != after else TRACKED[1]


def misses(out: Path, setup: str, scenes: list[str]) -> str:
    """The missed crossings of `setup`, detected against tracked, as a Markdown table."""
    table: Counter = Counter()
    for scene in scenes:
        folder = out / "runs" / setup / scene
        marks = json.loads((folder / "crossings.json").read_text())
        tracks = _tracks(folder)
        wrong = [m for m in marks if m["mark"] == "wrong"]
        for m in marks:
            if m["mark"] == "missed":
                detected = _detected(scene, m["id"], m["frame"])
                tracked = _tracked(scene, tracks, m["id"], m["frame"], m["type"], wrong)
                table[tracked, detected] += 1
    lines = [
        f"Missed crossings of {setup}, {sum(table.values())} on {len(scenes)} scenes. Detected:"
        f" a box at IoU {MATCH_IOU} or more on the person, within {DETECT_LOOK} s of the"
        f" crossing. Tracked: the ID on the person in the {TRACK_LOOK} s before and after.",
        "",
        "| what our tracks did | " + " | ".join(f"detected {d}" for d in DETECTED) + " | all |",
        "| --- |" + " --- |" * (len(DETECTED) + 1),
    ]
    for t in (*TRACKED, None):
        row = [table[t, d] if t else sum(table[x, d] for x in TRACKED) for d in DETECTED]
        lines.append(f"| {t or 'all'} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    return "\n".join(lines) + "\n"


def _rescore(job: tuple[str, str, Path]) -> tuple[str, str, list[tuple[int, int, int]]]:
    setup, scene, folder = job
    fps = float(detections(scene)["fps"])
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
