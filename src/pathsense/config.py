"""Pipeline configuration: one YAML file, plus `section.key=value` overrides.

Swapping the detector or tracker is one line, either in the file or on the command line:

    detector:
      name: rfdetr-nano        # was yolo26n

    pathsense run --set tracker.name=botsort

Unknown sections or keys are errors, not silently ignored, so a typo cannot quietly fall
back to a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """The config file or an override is invalid."""


@dataclass(frozen=True)
class DetectorConfig:
    name: str = "yolo26n"  # a key of pathsense.detect.DETECTORS
    weights: str | None = None  # None: the model's own pretrained weights
    imgsz: int | None = None  # None: the model's default input size (YOLO 640, RF-DETR Nano 384)
    # Low on purpose: ByteTrack's second matching pass uses boxes between this and 0.6 to keep
    # following people who are partly hidden (see NOTES.md, Phase 0).
    conf: float = 0.1
    device: str | None = None  # None: GPU if available. "cpu", "cuda", "0"...


@dataclass(frozen=True)
class TrackerConfig:
    name: str = "bytetrack"  # a key of pathsense.track.TRACKERS
    # Extra keyword arguments for the tracker class, e.g. {lost_track_buffer: 60}.
    # configs/default.yaml sets track_activation_threshold: 0.6 (ByteTrack's own is 0.7).
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraConfig:
    # Measure a new camera once and reuse the result (pathsense.io.camera). Off: OpenCV's
    # defaults, e.g. raw 640x480.
    calibrate: bool = True
    max_width: int = 1280  # largest picture to consider; the detector shrinks to 640 anyway
    fps: float = 30.0  # target frame rate
    # Many webcams drop to 20 FPS in dim light to expose longer when this is on. False:
    # steady FPS, darker picture. None: leave the camera's setting alone.
    dynamic_framerate: bool | None = False
    store: str | None = None  # None: ~/.config/pathsense/cameras.json


Points = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    polygon: Points


@dataclass(frozen=True)
class AnalyticsConfig:
    # Counting line in pixels, [[x1, y1], [x2, y2], ...]: two points, or more to follow a
    # curved edge. Standing on the first point looking at the next, crossing from your left
    # to your right is "enter" (pathsense.analytics.line).
    line: Points | None = None
    line_margin: float = 5.0  # feet closer than this to the line do not change side
    line_margin_height: float = 0.0  # if > 0, the margin is at least this share of height
    line_confirm: float = 0.5  # seconds on the new side before a crossing counts
    # Zones: [{name: door, polygon: [[x1, y1], [x2, y2], [x3, y3], ...]}, ...]
    zones: tuple[ZoneSpec, ...] = ()
    zone_confirm: float = 0.5  # seconds inside (or outside) before an enter (or exit) counts
    zone_count_present: bool = True  # people inside when the video starts count as entered
    # Legs hidden (a cart, another person): a box this much shorter than the person's full
    # height gets its feet from the head (pathsense.analytics.feet). 0: off.
    hidden_legs: float = 0.25

    def __post_init__(self) -> None:
        if self.line is not None:
            object.__setattr__(self, "line", parse_line(self.line))
        object.__setattr__(self, "zones", parse_zones(self.zones))
        for key in ("line_margin", "line_margin_height", "line_confirm", "zone_confirm"):
            object.__setattr__(self, key, non_negative(f"analytics.{key}", getattr(self, key)))
        hidden = non_negative("analytics.hidden_legs", self.hidden_legs)
        if hidden >= 1:
            raise ConfigError(f"analytics.hidden_legs must be below 1, got {self.hidden_legs!r}")
        object.__setattr__(self, "hidden_legs", hidden)
        if not isinstance(self.zone_count_present, bool):
            value = self.zone_count_present
            raise ConfigError(f"analytics.zone_count_present must be true or false, got {value!r}")


def non_negative(name: str, value: Any) -> float:
    """A finite number >= 0, or a ConfigError naming the setting."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if isinstance(value, bool) or not (math.isfinite(number) and number >= 0):
        raise ConfigError(f"{name} must be a number >= 0, got {value!r}")
    return number


def parse_points(name: str, value: Any, minimum: int) -> Points:
    """[[x, y], ...] with at least `minimum` finite points -> tuple of (x, y)."""
    try:
        points = tuple((float(x), float(y)) for x, y in value)
        if len(points) < minimum or not all(math.isfinite(v) for p in points for v in p):
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigError(
            f"{name} must be [[x, y], ...] with at least {minimum} points, got {value!r}"
        ) from None
    if any(a == b for a, b in pairwise(points)):
        raise ConfigError(f"{name} repeats a point, got {value!r}")
    return points


def parse_line(value: Any) -> Points:
    return parse_points("analytics.line", value, minimum=2)


def parse_zones(value: Any) -> tuple[ZoneSpec, ...]:
    """[{name, polygon}, ...] -> ZoneSpecs; names must be unique."""
    if isinstance(value, tuple) and all(isinstance(z, ZoneSpec) for z in value):
        return value
    if not isinstance(value, list | tuple):
        raise ConfigError(f"analytics.zones must be a list, got {value!r}")
    zones = []
    for i, zone in enumerate(value):
        if not isinstance(zone, dict) or set(zone) != {"name", "polygon"}:
            raise ConfigError(
                f"analytics.zones[{i}] must have exactly name and polygon, got {zone!r}"
            )
        name = zone["name"]
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"analytics.zones[{i}].name must be a non empty text, got {name!r}")
        polygon = parse_points(f"analytics.zones[{i}].polygon", zone["polygon"], minimum=3)
        zones.append(ZoneSpec(name, polygon))
    names = [z.name for z in zones]
    if len(set(names)) != len(names):
        raise ConfigError(f"analytics.zones names must be unique, got {names}")
    return tuple(zones)


@dataclass(frozen=True)
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)


SECTIONS = {
    "detector": DetectorConfig,
    "tracker": TrackerConfig,
    "camera": CameraConfig,
    "analytics": AnalyticsConfig,
}


##### Loading ##############################################################################


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Read a YAML config (default: configs/default.yaml) and apply `a.b=value` overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    for override in overrides or []:
        apply_override(raw, override)
    return from_dict(raw)


def apply_override(raw: dict[str, Any], override: str) -> None:
    """Set `section.key=value` in a raw config dict. The value is parsed as YAML."""
    key, sep, value = override.partition("=")
    parts = key.strip().split(".")
    if not sep or len(parts) < 2 or not all(parts):
        raise ConfigError(f"Override {override!r} must look like section.key=value")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Override {override!r}: {part} is not a section")
    node[parts[-1]] = yaml.safe_load(value)  # "0.3" -> 0.3, "null" -> None, "{a: 1}" -> dict


def from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
    config = Config()
    for section, cls in SECTIONS.items():
        values = raw.get(section) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] must be a mapping")
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ConfigError(
                f"Unknown keys in [{section}]: {sorted(unknown)}; allowed: {sorted(allowed)}"
            )
        config = replace(config, **{section: cls(**values)})
    if config.detector.device == "auto":  # accepted for readability in YAML
        config = replace(config, detector=replace(config.detector, device=None))
    return config
