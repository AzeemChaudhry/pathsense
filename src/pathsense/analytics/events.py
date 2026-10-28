"""The event log: one record per thing that happened, the interface between vision and
language. Answers in Phase 6 are computed from these records, never by the model.

On disk it is JSON Lines, one event per line:

    {"t": 6.64, "frame": 166, "track_id": 23, "type": "enter"}
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Event:
    t: float  # seconds since the first frame (Frame.timestamp)
    frame: int  # Frame.index
    track_id: int
    type: str  # "enter" or "exit" for a line; zone events come later in Phase 2

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class EventLog:
    """Writes events to a JSON Lines file as they happen, so a run stopped with Ctrl+C or a
    crash keeps everything up to that point.

    Lines are in the order events were confirmed, which is not always the order they
    happened: a crossing is dated when the person flipped side but confirmed later.
    `load_events` sorts by time.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self.written = 0

    def write(self, events: Iterable[Event]) -> None:
        for event in events:
            self._file.write(json.dumps(event.to_dict()) + "\n")
            self.written += 1
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_events(path: Path | str) -> list[Event]:
    """Every event in a log file, in time order."""
    with Path(path).open(encoding="utf-8") as file:
        events = [Event(**json.loads(line)) for line in file if line.strip()]
    return sorted(events, key=lambda e: (e.t, e.frame))


def count_events(events: Iterable[Event]) -> Counter[str]:
    """How many events of each type: count_events(events)["enter"]."""
    return Counter(e.type for e in events)
