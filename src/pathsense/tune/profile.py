"""Describe a scene in numbers, so a tuning result can be read against what the scene is.

    profile = scene_profile(saved, tracks)
    print("\\n".join(profile.lines()))

From the saved detections: how many people, how tall, how close together. From one
replay's tracks: how fast people move. Distances are in person heights, so a close-up and
a wide view compare.

No verdict is drawn from these numbers on purpose. Box overlap and neighbour distance did
not tell the MOT20 crowds from the MOT17 streets (seen from above, packed people barely
overlap); the number of people did. Scoring candidates against the truth is what decides.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import supervision as sv

from pathsense.tune.cache import SavedDetections

SURE = 0.5  # detections at least this confident count as people here; lower ones are noise


@dataclass(frozen=True)
class SceneProfile:
    frames: int
    seconds: float
    fps: float
    size: tuple[int, int]  # width, height in pixels; (0, 0) if unknown
    people: float  # median people per frame
    people_busy: float  # 90th percentile per frame
    height: float  # median person height, px
    height_small: float  # 10th percentile height, px
    spacing: float  # median distance from a person's feet to the nearest other's, in heights
    speed: float  # median walking speed, heights per second (nan without tracks)

    def lines(self) -> list[str]:
        width, height = self.size
        return [
            f"{self.frames} frames, {self.seconds:.0f} s at {self.fps:.0f} FPS"
            + (f", {width}x{height}" if width else ""),
            f"people per frame: {self.people:.0f} typical, {self.people_busy:.0f} busy",
            f"person height: {self.height:.0f} px typical, {self.height_small:.0f} px small",
            f"nearest other person: {self.spacing:.2f} heights away",
            f"walking speed: {self.speed:.2f} heights per second",
        ]


def _feet(xyxy: np.ndarray) -> np.ndarray:
    return np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])


def _spacing(xyxy: np.ndarray) -> np.ndarray:
    """Each person's distance to the nearest other, in their own heights."""
    if len(xyxy) < 2:
        return np.zeros(0)
    feet = _feet(xyxy)
    distance = np.linalg.norm(feet[:, None] - feet[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1) / np.maximum(xyxy[:, 3] - xyxy[:, 1], 1)


def _speed(tracks: Mapping[int, sv.Detections], fps: float) -> float:
    """Median feet speed between consecutive frames of the same track, in heights per s."""
    last: dict[int, tuple[int, np.ndarray, float]] = {}  # id -> frame, feet, height
    speeds = []
    for index in sorted(tracks):
        people = tracks[index]
        for feet, box, track_id in zip(
            _feet(people.xyxy), people.xyxy, people.tracker_id, strict=True
        ):
            height = float(box[3] - box[1])
            seen = last.get(int(track_id))
            if seen and seen[0] == index - 1 and height > 0:
                speeds.append(float(np.linalg.norm(feet - seen[1])) / height * fps)
            last[int(track_id)] = (index, feet, height)
    return float(np.median(speeds)) if speeds else float("nan")


def scene_profile(
    saved: SavedDetections, tracks: Mapping[int, sv.Detections] | None = None
) -> SceneProfile:
    sure = saved.confidence >= SURE
    starts = saved.starts
    per_frame, spacing = [], []
    for k in range(len(saved)):
        rows = slice(starts[k], starts[k + 1])
        boxes = saved.xyxy[rows][sure[rows]]
        per_frame.append(len(boxes))
        spacing.append(_spacing(boxes))
    heights = saved.xyxy[sure, 3] - saved.xyxy[sure, 1]
    gaps = np.concatenate(spacing) if spacing else np.zeros(0)

    def median(values) -> float:
        return float(np.median(values)) if len(values) else float("nan")

    return SceneProfile(
        frames=len(saved),
        seconds=len(saved) / saved.fps,
        fps=saved.fps,
        size=saved.size,
        people=median(per_frame),
        people_busy=float(np.percentile(per_frame, 90)) if per_frame else float("nan"),
        height=median(heights),
        height_small=float(np.percentile(heights, 10)) if len(heights) else float("nan"),
        spacing=median(gaps),
        speed=_speed(tracks, saved.fps) if tracks else float("nan"),
    )


__all__ = ["SceneProfile", "scene_profile"]
