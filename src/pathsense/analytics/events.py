"""The event log: one record per thing that happened, the interface between vision and
language. Answers in Phase 6 are computed from these records, never by the model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Event:
    t: float  # seconds since the first frame (Frame.timestamp)
    frame: int  # Frame.index
    track_id: int
    type: str  # "enter" or "exit" for a line; zone events come later in Phase 2

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)
