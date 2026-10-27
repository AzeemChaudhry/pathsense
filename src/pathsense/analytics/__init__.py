"""Movement analytics built on tracked detections.

Every result is an `Event` (a person crossed a line, entered a zone...), so counts, dwell
times and the plain language answers of Phase 6 all come from one event log.

The logic is plain arithmetic on box coordinates on purpose: the browser app reimplements
it in TypeScript, and a parity test compares both (docs/PLAN.md).
"""

from pathsense.analytics.events import Event
from pathsense.analytics.line import LineCounter, anchor_points, side_of_line

__all__ = ["Event", "LineCounter", "anchor_points", "side_of_line"]
