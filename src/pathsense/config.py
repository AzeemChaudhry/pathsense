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
    # Detect on overlapping tiles of this many pixels (pathsense.detect.slice): finds small
    # and crowded people in large frames, several times slower. None: whole frame.
    slice: int | None = None
    # Drop a box that is a partial view of a person also boxed whole (upper body behind a
    # shopping cart; pathsense.detect.base.drop_partial_duplicates). Off by default: in a
    # crowd seen from above it also drops people standing behind others.
    drop_partial: bool = False

    def __post_init__(self) -> None:
        if self.slice is not None and (
            isinstance(self.slice, bool) or not isinstance(self.slice, int) or self.slice < 160
        ):
            raise ConfigError(
                f"detector.slice must be a tile size of 160 px or more, got {self.slice!r}"
            )


@dataclass(frozen=True)
class TrackerConfig:
    name: str = "bytetrack"  # a key of pathsense.track.TRACKERS
    # Extra keyword arguments for the tracker class, e.g. {lost_track_buffer: 60}.
    # configs/default.yaml sets track_activation_threshold: 0.6 (ByteTrack's own is 0.7).
    params: dict[str, Any] = field(default_factory=dict)
    # Give a person the tracker lost for up to this many seconds their old ID back when they
    # reappear where they were heading (pathsense.track.stitch). 0: off.
    stitch_seconds: float = 2.0
    stitch_distance: float = 1.0  # how far from the expected spot, in box heights
    stitch_size: float = 0.5  # box height within 1/(1 + this) to 1 + this times theirs

    def __post_init__(self) -> None:
        for key in ("stitch_seconds", "stitch_distance", "stitch_size"):
            object.__setattr__(self, key, non_negative(f"tracker.{key}", getattr(self, key)))


APPEARANCE_MATCHES = ("off", "botsort", "veto", "blend", "adaptive")


@dataclass(frozen=True)
class AppearanceConfig:
    """Tell people apart by how they look (pathsense.appearance). Off: on the MOT scenes it
    did not earn its cost (docs/TRACKER_STUDY.md, Appearance). Each use below is 0 or "off"
    until set, so `enabled` alone changes nothing but the time spent."""

    enabled: bool = False
    weights: str = "data/models/osnet_x0_25_msmt17.pt"
    device: str | None = None  # None: GPU if available
    batch: int = 256  # boxes at a time through the network
    # The tracker's first pass, ByteTrack only: off | botsort | veto | blend | adaptive
    match: str = "off"
    closeness: float = 0.25  # botsort: how near in look; 0 same, 1 opposite
    proximity: float = 0.5  # botsort: how near in overlap, as 1 - IoU
    blend: float = 0.5  # blend: share of the score that is look
    gate: float = 0.0  # veto, blend: a look below this refuses the match; 0: off
    adaptive_weight: float = 0.75  # adaptive: base weight of look (Deep OC-SORT's MOT value)
    adaptive_cap: float = 0.5  # adaptive: most a clear best match adds to that weight
    memory: float = 0.9  # how much of a track's remembered look each new box keeps
    dynamic: bool = False  # a less confident box changes that memory less (Deep OC-SORT)
    skip_overlap: float = 0.0  # no memory from a box overlapping another this much; 0: off
    stitch_gate: float = 0.0  # the stitcher refuses a link that looks below this; 0: off
    line_same: float = 0.0  # a bounce at or above this look counts every crossing; 0: off
    line_different: float = 0.0  # a reversal below this look is dropped at once; 0: off
    line_memory: float = 0.8  # as `memory`, for the look the line compares

    def __post_init__(self) -> None:
        for key in ("enabled", "dynamic"):
            if not isinstance(getattr(self, key), bool):
                raise ConfigError(
                    f"appearance.{key} must be true or false, got {getattr(self, key)!r}"
                )
        if self.match not in APPEARANCE_MATCHES:
            raise ConfigError(
                f"appearance.match must be one of {APPEARANCE_MATCHES}, got {self.match!r}"
            )
        if isinstance(self.batch, bool) or not isinstance(self.batch, int) or self.batch < 1:
            raise ConfigError(f"appearance.batch must be a whole number >= 1, got {self.batch!r}")
        keys = (
            "closeness proximity blend gate adaptive_weight adaptive_cap memory skip_overlap "
            "stitch_gate line_same line_different"
        )
        for key in keys.split():
            object.__setattr__(self, key, non_negative(f"appearance.{key}", getattr(self, key)))
        object.__setattr__(
            self, "line_memory", non_negative("appearance.line_memory", self.line_memory)
        )
        for key in ("blend", "memory", "line_memory"):
            if getattr(self, key) >= 1:
                raise ConfigError(f"appearance.{key} must be below 1, got {getattr(self, key)!r}")
        for key in ("gate", "stitch_gate", "line_same", "line_different"):  # similarities
            if getattr(self, key) > 1:
                raise ConfigError(f"appearance.{key} must be 1 or less, got {getattr(self, key)!r}")
        if self.match == "veto" and self.gate <= 0:
            raise ConfigError("appearance.match veto needs appearance.gate above 0")


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
    # Seconds to hold a crossing that reverses the ID's last one; dropped if the ID crosses
    # back within that time (an ID swap in a crowd, pathsense.analytics.line). 0: off.
    line_bounce: float = 12.0
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
        keys = ("line_margin", "line_margin_height", "line_confirm", "line_bounce", "zone_confirm")
        for key in keys:
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
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)


SECTIONS = {
    "detector": DetectorConfig,
    "tracker": TrackerConfig,
    "camera": CameraConfig,
    "analytics": AnalyticsConfig,
    "appearance": AppearanceConfig,
}


##### Loading ##############################################################################


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    scene: str | Path | None = None,
) -> Config:
    """Read a YAML config (default: configs/default.yaml), lay a scene file over it, then
    apply `a.b=value` overrides.

    A scene (configs/scenes/*.yaml) holds what belongs to one camera: its lines, zones and
    settings, and optionally its `source` and `truth` (what really happened, for
    `pathsense tune`). It changes only the keys it names, so shared settings stay in the
    base config.
    """
    raw = read_yaml(Path(path) if path else DEFAULT_CONFIG)
    if scene:
        layer = read_yaml(Path(scene))
        layer.pop("source", None)
        layer.pop("truth", None)  # read by pathsense.tune.truth, not part of the pipeline
        for section, values in layer.items():
            if isinstance(values, dict) and isinstance(raw.get(section), dict):
                raw[section] = {**raw[section], **values}
            else:
                raw[section] = values
    for override in overrides or []:
        apply_override(raw, override)
    return from_dict(raw)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return raw


def scene_source(scene: str | Path) -> str | None:
    """The video a scene file names (`source:`), if any."""
    source = read_yaml(Path(scene)).get("source")
    return None if source is None else str(source)


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
