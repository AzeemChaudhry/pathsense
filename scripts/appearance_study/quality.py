"""How good the vectors themselves are, scored against the labels.

- `people_table`: is a person's later box closer than the neighbours beside them?
- `switches_table`: at the tracker's real ID switches, did the look change?
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import supervision as sv

from pathsense.config import load_config, scene_source
from pathsense.pipeline import Pipeline
from pathsense.tune.cache import cache_path
from pathsense.tune.embeddings import embeddings_path, load_embeddings
from pathsense.tune.replay import no_detector
from pathsense.tune.run import _load_scene, _Scene

HOLD = 15  # frames an ID must stay on a person, before and after, to count as a switch
MEMORY = 0.9
WORKERS = 4


@cache
def study_scene(name: str) -> _Scene:
    path = Path(f"configs/scenes/mot/{name}.yaml")
    return _load_scene(path, None, cache_path(scene_source(path)))


def _vectors(scene: _Scene) -> np.ndarray:
    look = load_config(scene=scene.path).appearance
    return load_embeddings(embeddings_path(scene.saved_path, scene.saved, look))


def _matched(boxes: sv.Detections, truth: sv.Detections | None) -> dict[int, int]:
    """Row of `boxes` -> label, where each is the other's best at IoU 0.5 or more."""
    if truth is None or len(truth) == 0 or len(boxes) == 0:
        return {}
    overlap = sv.box_iou_batch(boxes.xyxy, truth.xyxy)
    best = zip(*np.nonzero(overlap >= 0.5), strict=True)
    return {
        int(r): int(truth.tracker_id[c])
        for r, c in best
        if overlap[r, c] == overlap[:, c].max() and overlap[r, c] == overlap[r].max()
    }


def _people(name: str) -> str:
    scene = study_scene(name)
    vectors = _vectors(scene)
    starts = scene.saved.starts
    who: dict[int, dict] = {}
    for k, (frame, boxes) in enumerate(scene.saved):
        found = {}
        for row, label in _matched(boxes, scene.labels.get(frame.index)).items():
            x1, _, x2, y2 = boxes.xyxy[row]
            height = y2 - boxes.xyxy[row][1]
            look = np.asarray(vectors[starts[k] + row], np.float32)
            found[label] = (look, (x1 + x2) / 2, y2, height)
        who[frame.index] = found
    cells = [name]
    for gap in (1, round(scene.saved.fps), round(5 * scene.saved.fps)):
        right, total = 0, 0
        for f in sorted(who)[::5]:
            later = who.get(f + gap, {})
            for person, (look, *_) in who[f].items():
                if person not in later:
                    continue
                same, x, y, height = later[person]
                near = [
                    float(look @ v[0])
                    for q, v in later.items()
                    if q != person and np.hypot(v[1] - x, v[2] - y) < height
                ]
                if near:
                    total += 1
                    right += float(look @ same) > max(near)
        cells.append(f"{100 * right / max(total, 1):.0f}% ({total})")
    return "| " + " | ".join(cells) + " |"


def people_table(scenes: list[str]) -> str:
    head = "| scene | 1 frame (pairs) | 1 s | 5 s |\n| --- | --- | --- | --- |"
    with ProcessPoolExecutor(WORKERS) as pool:
        return "\n".join([head, *pool.map(_people, scenes)])


def _switches(name: str) -> tuple[list[float], list[float], list[float]]:
    scene = study_scene(name)
    vectors = _vectors(scene)
    config = load_config(scene=scene.path)
    pipeline = Pipeline.from_config(config, scene.saved.fps, detector=no_detector)
    starts = scene.saved.starts
    memory: dict[int, np.ndarray] = {}
    last: dict[int, int] = {}
    runs: dict[int, list] = {}  # track -> [[label, frames, look on the first frame], ...]
    clear, overlapped = [], []
    for k, (frame, boxes) in enumerate(scene.saved):
        looks = np.asarray(vectors[starts[k] : starts[k + 1]], np.float32)
        boxes.data["row"] = np.arange(len(boxes))
        tracked = pipeline.track(boxes, frame).tracked
        if len(tracked) == 0:
            continue
        found = _matched(tracked, scene.labels.get(frame.index))
        around = sv.box_iou_batch(tracked.xyxy, tracked.xyxy)
        np.fill_diagonal(around, 0)
        for r, track in enumerate(tracked.tracker_id):
            track, look = int(track), looks[tracked.data["row"][r]]
            old = memory.get(track)
            if old is not None and r in found:
                alike = float(old @ look)
                seq = runs.setdefault(track, [])
                if seq and seq[-1][0] == found[r]:
                    seq[-1][1] += 1
                else:
                    seq.append([found[r], 1, alike])
                if last.get(track) == found[r]:
                    (overlapped if around[r].max() > 0.3 else clear).append(alike)
            if r in found:
                last[track] = found[r]
            mixed = look if old is None else MEMORY * old + (1 - MEMORY) * look
            memory[track] = mixed / (np.linalg.norm(mixed) + 1e-12)
    switches = []
    for seq in runs.values():
        held = [x for x in seq if x[1] >= HOLD]
        switches += [b[2] for a, b in zip(held, held[1:], strict=False) if a[0] != b[0]]
    return switches, clear, overlapped


def switches_table(scenes: list[str]) -> str:
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(_switches, scenes))
    lines = ["| boxes | n | below 0.5 | below 0.6 | below 0.7 |", "| --- | --- | --- | --- | --- |"]
    groups = ("real ID switches", "same person, clear", "same person, overlapped")
    for i, label in enumerate(groups):
        values = np.array([v for result in results for v in result[i]])
        shares = [f"{100 * (values < limit).mean():.1f}%" for limit in (0.5, 0.6, 0.7)]
        lines.append(f"| {label} | {len(values)} | " + " | ".join(shares) + " |")
    return "\n".join(lines)
