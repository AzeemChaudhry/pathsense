"""Look at a video for a moment and choose its settings, with no labels.

    analysis = analyze("clip.mp4", load_rules())
    analysis.overrides       # section.key=value settings, as `pathsense run --set` takes
    print(analysis.report())

Scans evenly spaced frames of the first two minutes with one quick detector, describes
the scene in numbers (pathsense.tune.profile), and takes the first rule in
configs/analyzer.yaml that fits. A fixed camera's scene changes little, so a short scan is
enough. The rules are data: fit on labelled scenes, and fit again after Phase 4's training,
with no change here.

`pathsense tune` scores setups against truth and is better when truth exists. This is for
the video that has none. It never sets a line or a zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pathsense.config import ConfigError, DetectorConfig, read_yaml
from pathsense.detect import Detector, build_detector
from pathsense.io import VideoSource
from pathsense.tune.cache import SavedDetections
from pathsense.tune.profile import SceneProfile, scene_profile

RULES = Path(__file__).resolve().parents[3] / "configs" / "analyzer.yaml"
SCAN_SECONDS = 120.0  # a camera's scene changes little; decoding more costs more
MEASURES = {"people", "people_busy", "height", "height_small", "spacing"}


@dataclass(frozen=True)
class Rule:
    name: str
    why: str
    overrides: tuple[str, ...]
    below: dict[str, float]  # measure -> the scene must be under this
    at_least: dict[str, float]  # measure -> the scene must be this or more

    def fits(self, profile: SceneProfile) -> bool:
        low = all(getattr(profile, k) < v for k, v in self.below.items())
        return low and all(getattr(profile, k) >= v for k, v in self.at_least.items())


@dataclass(frozen=True)
class Rules:
    scan: DetectorConfig
    frames: int
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class Analysis:
    profile: SceneProfile
    rule: Rule

    @property
    def overrides(self) -> tuple[str, ...]:
        return self.rule.overrides

    def report(self) -> str:
        lines = ["The scene:", *(f"- {line}" for line in self.profile.lines()[:4]), ""]
        lines += [f"Chosen: {self.rule.name}, {self.rule.why}", "", "Settings:"]
        return "\n".join(lines + [f"  --set {o}" for o in self.overrides] or ["  as they are"])


def load_rules(path: Path | str = RULES) -> Rules:
    raw = read_yaml(Path(path))
    rules = []
    for item in raw.get("rules") or []:
        extra = set(item.get("below", {})) | set(item.get("at_least", {}))
        if extra - MEASURES:
            raise ConfigError(f"{path}: unknown measures {sorted(extra - MEASURES)}")
        rules.append(
            Rule(
                str(item["name"]),
                str(item["why"]),
                tuple(item.get("set", ())),
                dict(item.get("below", {})),
                dict(item.get("at_least", {})),
            )
        )
    if not rules or rules[-1].below or rules[-1].at_least:
        raise ConfigError(f"{path}: the last rule must have no conditions, so one always fits")
    return Rules(DetectorConfig(**raw["scan"]), int(raw["frames"]), tuple(rules))


def choose(profile: SceneProfile, rules: Rules) -> Rule:
    return next(rule for rule in rules.rules if rule.fits(profile))


def scan(video: Path | str, detector: Detector, frames: int) -> SavedDetections:
    """The detector's boxes on `frames` evenly spaced frames."""
    boxes: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    size = (0, 0)
    with VideoSource(str(video)) as source:
        fps = source.fps
        count = min(source.frame_count, int(SCAN_SECONDS * fps))
        wanted = {int(k) for k in np.linspace(0, max(count - 1, 0), frames)} if count else None
        last = max(wanted) if wanted else frames - 1
        for frame in source:
            if wanted is None or frame.index in wanted:
                found = detector(frame.image)
                boxes.append(found.xyxy.astype(np.float32))
                confidences.append(found.confidence.astype(np.float32))
                size = (frame.image.shape[1], frame.image.shape[0])
            if frame.index >= last:  # a stream has no length: its first frames
                break
    if not boxes:
        raise ConfigError(f"no frames to scan in {video}")
    n = len(boxes)
    return SavedDetections(
        xyxy=np.concatenate(boxes),
        confidence=np.concatenate(confidences),
        counts=np.array([len(b) for b in boxes], int),
        index=np.arange(n),
        timestamp=np.arange(n) / fps,
        fps=fps,
        size=size,
    )


def analyze(video: Path | str, rules: Rules, build: Any = build_detector) -> Analysis:
    profile = scene_profile(scan(video, build(rules.scan), rules.frames))
    return Analysis(profile, choose(profile, rules))


__all__ = ["Analysis", "Rule", "Rules", "analyze", "choose", "load_rules", "scan"]
