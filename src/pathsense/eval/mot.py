"""MOT Challenge files: read the labels, write our tracks, score tracking.

A MOT ground truth row is `frame, id, x, y, w, h, flag, class, visibility`, frames from 1.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import supervision as sv

# Pedestrian, person on a vehicle, static person: everyone a person counter should see.
# Distractors (reflections, mannequins) and non-people are left out.
PEOPLE = {1, 2, 7}

# What tracking_metrics returns, from the CLEAR, HOTA and Identity families.
METRICS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "MT", "ML", "CLR_FP", "CLR_FN")


def ground_truth(path: Path | str, visibility: float = 0.0) -> dict[int, sv.Detections]:
    """Frame index (from 0) -> the labelled people at least `visibility` visible, with
    their IDs as tracker_id."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.shape[1] < 9:
        raise ValueError(f"{path}: expected MOT ground truth rows of 9 values")
    rows = rows[np.isin(rows[:, 7], list(PEOPLE)) & (rows[:, 8] >= visibility)]
    frames = {}
    for frame in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == frame]
        xyxy = np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])
        frames[frame - 1] = sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=np.ones(len(r), np.float32),
            class_id=np.zeros(len(r), int),
            tracker_id=r[:, 1].astype(int),
        )
    return frames


def write_mot(path: Path | str, frames: Iterable[tuple[int, sv.Detections]]) -> None:
    """Write (frame index from 0, tracked people) pairs as a MOT results file."""
    with open(path, "w", encoding="utf-8") as out:
        for index, tracked in frames:
            for (x1, y1, x2, y2), track_id in zip(tracked.xyxy, tracked.tracker_id, strict=True):
                w, h = x2 - x1, y2 - y1
                out.write(f"{index + 1},{track_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n")


def tracking_metrics(labels: Path | str, tracks: Path | str) -> dict[str, float]:
    """HOTA, MOTA, IDF1, ID switches and more for a MOT results file.

    Uses the `trackers` library's evaluation, built to match TrackEval: only pedestrians
    marked for evaluation count, and a box on a distractor (a static person, a reflection)
    is neither right nor wrong. Ratios are from 0 to 1.
    """
    from trackers.eval import evaluate_mot_sequence

    result = evaluate_mot_sequence(labels, tracks, metrics=["CLEAR", "HOTA", "Identity"])
    values = {**vars(result.CLEAR), **vars(result.HOTA), **vars(result.Identity)}
    return {key: float(values[key]) for key in METRICS}
