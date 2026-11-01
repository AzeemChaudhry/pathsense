"""What really happened in a scene, for tuning to score against: the `truth:` block of a
scene file.

Either hand labels, every person boxed in every frame (MOT format):

    truth:
      labels: data/mot/gt/MOT20-03.txt

or hand counts, for a clip nobody labelled box by box:

    truth:
      enter: 11          # line crossings each way
      exit: 7
      zones: {aisle: 1}  # people who entered each zone

Hand counts can list each crossing, which must add up to `enter` and `exit`:

      crossings:         # seconds, direction, x in pixels
        - [0.5, enter, 1522]
        - [1.5, exit, 713]

Labels and listed crossings score every crossing one by one (pathsense.eval); counts alone
only score the totals, so a false crossing and a missed one can cancel out there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathsense.config import ConfigError, read_yaml
from pathsense.eval.crossings import DIRECTIONS, Marked


@dataclass(frozen=True)
class Truth:
    labels: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)  # "enter", "exit", "zone:<name>"
    crossings: tuple[Marked, ...] = ()  # listed by hand, with counts

    def __post_init__(self) -> None:
        if (self.labels is None) == (not self.counts):
            raise ConfigError("truth needs either labels or counts, not both or neither")


def count_key(event_type: str, zone: str | None = None) -> str | None:
    """The `Truth.counts` key an event adds to, or None if counts do not track it."""
    if event_type in ("enter", "exit"):
        return event_type
    return f"zone:{zone}" if event_type == "zone_enter" else None


def scene_truth(scene: Path | str) -> Truth | None:
    """The scene file's `truth:` block, or None if it has none."""
    raw: Any = read_yaml(Path(scene)).get("truth")
    if raw is None:
        return None
    if not isinstance(raw, dict) or not set(raw) <= {
        "labels",
        "enter",
        "exit",
        "zones",
        "crossings",
    }:
        raise ConfigError(f"{scene}: truth must have labels, or enter, exit, zones and crossings")
    if "labels" in raw:
        if len(raw) > 1:
            raise ConfigError(f"{scene}: truth has labels, so it cannot also have counts")
        return Truth(labels=Path(raw["labels"]))
    counts = {}
    for key in ("enter", "exit"):
        if key in raw:
            counts[key] = _count(scene, key, raw[key])
    zones = raw.get("zones") or {}
    if not isinstance(zones, dict):
        raise ConfigError(f"{scene}: truth.zones must map zone names to counts")
    for name, value in zones.items():
        counts[f"zone:{name}"] = _count(scene, f"zones.{name}", value)
    return Truth(counts=counts, crossings=_crossings(scene, raw.get("crossings"), counts))


def _crossings(scene: Path | str, raw: Any, counts: dict[str, int]) -> tuple[Marked, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{scene}: truth.crossings must be a list of [seconds, direction, x]")
    marked = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or item[1] not in DIRECTIONS
            or isinstance(item[0], bool)
            or isinstance(item[2], bool)
            or not isinstance(item[0], int | float)
            or not isinstance(item[2], int | float)
        ):
            raise ConfigError(f"{scene}: truth.crossings needs [seconds, direction, x], got {item}")
        marked.append(Marked(float(item[0]), item[1], float(item[2])))
    for direction in DIRECTIONS:
        listed = sum(m.type == direction for m in marked)
        if listed != counts.get(direction):
            raise ConfigError(
                f"{scene}: truth.crossings lists {listed} {direction}, "
                f"truth.{direction} says {counts.get(direction)}"
            )
    return tuple(marked)


def _count(scene: Path | str, key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{scene}: truth.{key} must be a whole number >= 0, got {value!r}")
    return value


__all__ = ["Truth", "count_key", "scene_truth"]
