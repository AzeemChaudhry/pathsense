"""Evaluation against hand labelled video (MOT17, MOT20): tracking metrics and crossings.

    labels = ground_truth("data/mot/gt/MOT20-03.txt")       # frame index -> labelled people
    identity = Identity()                                     # which labelled person each
    identity.add(frame, tracked, labels.get(frame))           #   of our tracks follows
    score = match_crossings(our_events, true_events, identity, fps)
    metrics = tracking_metrics("data/mot/gt/MOT20-03.txt", "tracks.txt")

The data is CC BY-NC-SA: evaluation only, never redistributed, no frames in public docs.
"""

from pathsense.eval.crossings import CrossingScore, Identity, match_crossings
from pathsense.eval.mot import PEOPLE, ground_truth, tracking_metrics, write_mot

__all__ = [
    "PEOPLE",
    "CrossingScore",
    "Identity",
    "ground_truth",
    "match_crossings",
    "tracking_metrics",
    "write_mot",
]
