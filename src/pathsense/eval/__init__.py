"""Evaluation against hand labelled video (MOT17, MOT20): tracking metrics and crossings.

    labels = ground_truth("data/mot/gt/MOT20-03.txt")       # frame index -> labelled people
    identity = Identity()                                     # which labelled person each
    identity.add(frame, tracked, labels.get(frame))           #   of our tracks follows
    truth = label_events(config.analytics, labels, frames, fps)  # what really happened
    score = match_crossings(our_events, truth, identity, fps)
    explain_misses(score.missed, score.wrong, labels, detections, tracks, fps)  # why missed
    metrics = tracking_metrics("data/mot/gt/MOT20-03.txt", "tracks.txt")

The data is CC BY-NC-SA: evaluation only, never redistributed, no frames in public docs.
"""

from pathsense.eval.crossings import CrossingScore, Identity, match_crossings
from pathsense.eval.misses import explain_misses, misses_table
from pathsense.eval.mot import PEOPLE, ground_truth, tracking_metrics, write_mot
from pathsense.eval.truth import label_events

__all__ = [
    "PEOPLE",
    "CrossingScore",
    "Identity",
    "explain_misses",
    "ground_truth",
    "label_events",
    "match_crossings",
    "misses_table",
    "tracking_metrics",
    "write_mot",
]
