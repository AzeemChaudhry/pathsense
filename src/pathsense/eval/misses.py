"""Why a true crossing was missed: was the person detected, and what did our tracks do?

For each missed crossing, two questions about the labelled person who crossed:

- *Detected?* Did a detection overlap them (IoU >= 0.5) in at least half the frames within
  0.5 s of the crossing, in some, or in none? None is a detector limit.
- *Tracked?* Which of our IDs covered them in the second before and the second after:
  - no track: no ID followed them, detected or not;
  - one ID, no match: one ID stayed on them, but its crossing, if any, was too far in
    time or scored on someone else;
  - starts or ends: their track started or ended at the line;
  - ID changes: the ID before and after differ, so the track split;
  - swap: an ID on them crossed the same way but was scored on someone else (that
    crossing is also counted as wrong).

The tracker study's misses section is built from this (docs/TRACKER_STUDY.md).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np
import supervision as sv

from pathsense.analytics import Event

MATCH_IOU = 0.5  # a box is on the person at this overlap with their label or more
DETECT_LOOK = 0.5  # seconds either side of the crossing
TRACK_LOOK = 1.0  # seconds before, and after, the crossing

DETECTED = ("at least half", "some", "never")
TRACKED = ("no track", "one ID, no match", "starts or ends", "ID changes", "swap")


def _best(box: np.ndarray, boxes: np.ndarray) -> int | None:
    """Index of the box overlapping `box` most, if by at least MATCH_IOU."""
    if not len(boxes):
        return None
    overlap = sv.box_iou_batch(box[None], boxes)[0]
    return int(overlap.argmax()) if overlap.max() >= MATCH_IOU else None


def _box(people: sv.Detections | None, person: int) -> np.ndarray | None:
    if people is None or people.tracker_id is None:
        return None
    hits = np.flatnonzero(people.tracker_id == person)
    return people.xyxy[hits[0]] if len(hits) else None


def detected(
    person: int,
    frame: int,
    labels: Mapping[int, sv.Detections],
    detections: Mapping[int, sv.Detections],
    fps: float,
) -> str:
    """One of DETECTED: how often a detection was on `person` around `frame`."""
    look = round(DETECT_LOOK * fps)
    seen = shown = 0
    for f in range(frame - look, frame + look + 1):
        box, found = _box(labels.get(f), person), detections.get(f)
        if box is None or found is None:
            continue
        shown += 1
        seen += _best(box, found.xyxy) is not None
    if seen and seen >= shown / 2:
        return DETECTED[0]
    return DETECTED[1] if seen else DETECTED[2]


def tracked(
    missed: Event,
    labels: Mapping[int, sv.Detections],
    tracks: Mapping[int, sv.Detections],
    wrong: Iterable[Event],
    fps: float,
) -> str:
    """One of TRACKED: what our tracks did on the person around their missed crossing."""
    look, person, frame = round(TRACK_LOOK * fps), missed.track_id, missed.frame

    def on(frames: range) -> int | None:
        ids = []
        for f in frames:
            box, ours = _box(labels.get(f), person), tracks.get(f)
            if box is not None and ours is not None:
                i = _best(box, ours.xyxy)
                if i is not None:
                    ids.append(int(ours.tracker_id[i]))
        return Counter(ids).most_common(1)[0][0] if ids else None

    before, after = on(range(frame - look, frame)), on(range(frame, frame + look + 1))
    if before is None and after is None:
        return TRACKED[0]
    if any(
        e.type == missed.type and abs(e.frame - frame) <= 2 * fps and e.track_id in (before, after)
        for e in wrong
    ):
        return TRACKED[4]
    if before is None or after is None:
        return TRACKED[2]
    return TRACKED[3] if before != after else TRACKED[1]


def explain_misses(
    missed: Iterable[Event],
    wrong: Iterable[Event],
    labels: Mapping[int, sv.Detections],
    detections: Mapping[int, sv.Detections],
    tracks: Mapping[int, sv.Detections],
    fps: float,
) -> Counter:
    """(TRACKED answer, DETECTED answer) -> missed crossings, for a crossed table."""
    wrong = list(wrong)
    return Counter(
        (
            tracked(m, labels, tracks, wrong, fps),
            detected(m.track_id, m.frame, labels, detections, fps),
        )
        for m in missed
    )


def misses_table(table: Counter) -> str:
    """explain_misses() as a Markdown table, with totals."""
    lines = [
        "| what our tracks did | " + " | ".join(f"detected: {d}" for d in DETECTED) + " | all |",
        "| --- |" + " --- |" * (len(DETECTED) + 1),
    ]
    for t in (*TRACKED, None):
        row = [table[t, d] if t else sum(table[x, d] for x in TRACKED) for d in DETECTED]
        lines.append(f"| {t or 'all'} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    return "\n".join(lines) + "\n"


__all__ = ["DETECTED", "TRACKED", "detected", "explain_misses", "misses_table", "tracked"]
