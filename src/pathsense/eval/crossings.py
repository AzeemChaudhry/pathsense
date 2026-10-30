"""Score our line crossings one by one against the labelled people's crossings.

Totals hide errors that cancel: a missed OUT and a false OUT give the right count. Here
each of our crossings is tied to the labelled person under its box (`Identity`), and it is
right only when that person crossed the same way within `window` seconds. A crossing by an
ID the tracker moved onto someone else lands on the wrong person and counts as false; if
the person who really crossed has no right crossing of their own, it is also missed, so
one swap can count twice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import supervision as sv
from scipy.optimize import linear_sum_assignment

from pathsense.analytics import Event

DIRECTIONS = ("enter", "exit")


class Identity:
    """Which labelled person each of our tracks covers, frame by frame, by box overlap."""

    def __init__(self, min_iou: float = 0.3) -> None:
        self.min_iou = min_iou
        self._label: dict[tuple[int, int], int] = {}  # (track id, frame) -> label id

    def add(self, frame: int, tracked: sv.Detections, labels: sv.Detections | None) -> None:
        if labels is None or len(labels) == 0 or len(tracked) == 0:
            return
        overlap = sv.box_iou_batch(tracked.xyxy, labels.xyxy)
        rows, cols = linear_sum_assignment(-overlap)
        for r, c in zip(rows, cols, strict=True):
            if overlap[r, c] >= self.min_iou:
                self._label[(int(tracked.tracker_id[r]), frame)] = int(labels.tracker_id[c])

    def of(self, track_id: int, frame: int, frames: int) -> int | None:
        """The label the track covers most over `frames` frames from `frame` on, or None."""
        found = [self._label.get((track_id, f)) for f in range(frame, frame + frames)]
        common = Counter(label for label in found if label is not None).most_common(1)
        return common[0][0] if common else None


@dataclass
class CrossingScore:
    tp: Counter = field(default_factory=Counter)  # direction -> right crossings
    fp: Counter = field(default_factory=Counter)  # ours with no matching true crossing
    fn: Counter = field(default_factory=Counter)  # true crossings we did not count
    right: list[Event] = field(default_factory=list)  # our crossings that match a true one
    wrong: list[Event] = field(default_factory=list)  # our crossings that match none
    missed: list[Event] = field(default_factory=list)  # true crossings (label IDs) not counted

    def precision(self, direction: str | None = None) -> float:
        tp, fp = self._sum(self.tp, direction), self._sum(self.fp, direction)
        return tp / (tp + fp) if tp + fp else 1.0

    def recall(self, direction: str | None = None) -> float:
        tp, fn = self._sum(self.tp, direction), self._sum(self.fn, direction)
        return tp / (tp + fn) if tp + fn else 1.0

    def f1(self, direction: str | None = None) -> float:
        p, r = self.precision(direction), self.recall(direction)
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: CrossingScore) -> CrossingScore:
        return CrossingScore(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.right + other.right,
            self.wrong + other.wrong,
            self.missed + other.missed,
        )

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for d in DIRECTIONS:
            out |= {f"{d}_tp": self.tp[d], f"{d}_fp": self.fp[d], f"{d}_fn": self.fn[d]}
            out |= {f"{d}_precision": self.precision(d), f"{d}_recall": self.recall(d)}
        return out | {"crossing_f1": self.f1()}

    @staticmethod
    def _sum(counts: Counter, direction: str | None) -> int:
        return counts[direction] if direction else sum(counts[d] for d in DIRECTIONS)


def match_crossings(
    ours: list[Event],
    truth: list[Event],
    identity: Identity,
    fps: float,
    window: float = 2.0,
    look: float = 0.5,
) -> CrossingScore:
    """Pair our crossings with true ones of the same person and direction, closest first.

    Our crossing's person is the label its track covers for `look` seconds from the
    crossing on: after a swap, that is the person the ID is on now.
    """
    score = CrossingScore()
    frames = max(1, round(look * fps))
    for direction in DIRECTIONS:
        mine = [e for e in ours if e.type == direction]
        true = [e for e in truth if e.type == direction]
        pairs = []
        for i, event in enumerate(mine):
            label = identity.of(event.track_id, event.frame, frames)
            for j, real in enumerate(true):
                gap = abs(event.frame - real.frame) / fps
                if real.track_id == label and gap <= window:
                    pairs.append((gap, i, j))
        used_mine, used_true = set(), set()
        for _, i, j in sorted(pairs):
            if i not in used_mine and j not in used_true:
                used_mine.add(i)
                used_true.add(j)
        score.tp[direction] = len(used_mine)
        score.fp[direction] = len(mine) - len(used_mine)
        score.fn[direction] = len(true) - len(used_true)
        score.right += [e for i, e in enumerate(mine) if i in used_mine]
        score.wrong += [e for i, e in enumerate(mine) if i not in used_mine]
        score.missed += [e for j, e in enumerate(true) if j not in used_true]
    return score


__all__ = ["DIRECTIONS", "CrossingScore", "Identity", "match_crossings"]
