"""Phase 3 detector study: model, input size and confidence, scored against labels.

    uv run python scripts/personpath22.py             # scenes from PersonPath22
    uv run python scripts/detector_study detect       # GPU, once per model and size
    uv run python scripts/detector_study stage1       # every model and size (CPU)
    uv run python scripts/detector_study stage2       # the best, with every setting (CPU)
    uv run python scripts/detector_study report       # tables

Defined in configs/study/detectors.yaml, written up in docs/DETECTOR_STUDY.md. Saved boxes
go to data/cache/study/, results to runs/detector-study/ (not in git). Every step can be
stopped and run again: it skips what is done. Replays use the code a live video runs.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pathsense.config import load_config, read_yaml, scene_source  # noqa: E402
from pathsense.tune.cache import SavedDetections, cache_path, cached_detections  # noqa: E402
from pathsense.tune.run import _load_scene, load_candidates  # noqa: E402

STUDY = Path("configs/study/detectors.yaml")
SCENES = Path("data/personpath22/scenes")
CACHE = Path("data/cache/study")
OUT = Path("runs/detector-study")
WORKERS = 8
# A line can look busy to scripts/personpath22.py yet see too few crossings by the scorer's
# rules, and one crossing then swings a scene's F1. Too few: not scored, for any setup.
MIN_TRUE = 4
FIELDS = "detector imgsz confidence candidate scene group tp wrong missed ids".split()


def plan() -> dict:
    return yaml.safe_load(STUDY.read_text())


def people_per_frame(gt: Path) -> float:
    frames = np.loadtxt(gt, delimiter=",", usecols=0, dtype=int)
    return len(frames) / len(np.unique(frames))


def sample(study: dict) -> dict[str, str]:
    """Scene file -> group: MOT, then PersonPath22 spread evenly from sparse to busy."""
    groups = {f"configs/scenes/mot/{name}.yaml": "mot" for name in study["mot"]}
    splits: dict[str, list[tuple[float, str]]] = {"train": [], "test": []}
    for path in sorted(SCENES.glob("*.yaml")):
        split = path.read_text().split("\n")[0].split(", ")[1].split(";")[0]
        scene = read_yaml(path)
        if _frames(scene["source"]) <= study["max_frames"]:
            splits[split].append((people_per_frame(Path(scene["truth"]["labels"])), str(path)))
    for split, scenes in splits.items():
        scenes.sort()
        take = min(study["sample"][split], len(scenes))
        for k in np.linspace(0, len(scenes) - 1, take).round().astype(int):
            groups[scenes[k][1]] = split
    return groups


def _frames(video: str) -> int:
    import cv2

    capture = cv2.VideoCapture(video)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frames


def _folder(detector: dict) -> Path:
    return CACHE / f"{detector['name']}-{detector['imgsz']}"


def _config(scene: str, detector: dict):
    sets = [f"detector.name={detector['name']}", f"detector.imgsz={detector['imgsz']}"]
    return load_config(scene=scene, overrides=sets)


def detect(study: dict) -> None:
    from pathsense.detect import build_detector

    scenes = sample(study)
    for detector in study["detectors"]:
        model = None

        def build(config):
            nonlocal model
            model = model or build_detector(config)
            return model

        try:
            for k, scene in enumerate(scenes):
                video = scene_source(scene)
                saved = cached_detections(
                    video, _config(scene, detector).detector, _folder(detector), build=build
                )
                print(f"{detector} {k + 1}/{len(scenes)} {saved.detect_ms:.0f} ms", flush=True)
        except Exception as error:  # out of memory at a large size: keep the others
            print(f"{detector} failed: {type(error).__name__}: {str(error)[:100]}", flush=True)


@lru_cache(maxsize=4)  # scenes are large; a worker needs few at once
def _scene(scene: str, folder: str):
    return _load_scene(Path(scene), None, cache_path(scene_source(scene), folder))


def above(saved: SavedDetections, confidence: float) -> SavedDetections:
    """The saved boxes at this confidence or more, as a detector set to it would give."""
    keep = saved.confidence >= confidence
    frames = np.repeat(np.arange(len(saved.counts)), saved.counts)
    counts = np.bincount(frames[keep], minlength=len(saved.counts))
    return dataclasses.replace(
        saved, xyxy=saved.xyxy[keep], confidence=saved.confidence[keep], counts=counts
    )


def _score(job: tuple) -> dict | None:
    detector, confidence, candidate, scene, group = job
    try:
        loaded = _scene(scene, str(_folder(detector)))
    except FileNotFoundError:
        return None  # this detector never finished the scene
    loaded = dataclasses.replace(loaded, saved=above(loaded.saved, confidence))
    score, _ = loaded.run(candidate)
    c = score.crossings
    return {
        "detector": detector["name"],
        "imgsz": detector["imgsz"],
        "confidence": confidence,
        "candidate": candidate.name,
        "scene": Path(scene).stem,
        "group": group,
        "tp": sum(c.tp.values()),
        "wrong": sum(c.fp.values()),
        "missed": sum(c.fn.values()),
        "ids": score.ids,
    }


def _key(row: dict) -> tuple:
    return tuple(str(row[k]) for k in FIELDS[:5])


def run(jobs: list[tuple], path: Path) -> None:
    """Score `jobs`, adding to `path`, skipping those already there."""
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with path.open() as saved:
            done = {_key(row) for row in csv.DictReader(saved)}
    todo = []
    for job in jobs:
        detector, confidence, candidate, scene, _ = job
        names = (detector["name"], detector["imgsz"], confidence, candidate.name, Path(scene).stem)
        if tuple(str(v) for v in names) not in done:
            todo.append(job)
    print(f"{len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    new = not path.exists()
    with path.open("a", newline="") as out, ProcessPoolExecutor(WORKERS) as pool:
        writer = csv.DictWriter(out, FIELDS)
        if new:
            writer.writeheader()
        for k, row in enumerate(pool.map(_score, todo, chunksize=4)):
            if row:
                writer.writerow(row)
            if k % 200 == 0:
                out.flush()
                print(f"{k}/{len(todo)}", flush=True)


def rows(*paths: Path) -> list[dict]:
    found = []
    for path in paths:
        if path.exists():
            with path.open() as saved:
                found += list(csv.DictReader(saved))
    return found


def totals(found: list[dict], group: set[str]) -> dict[tuple, tuple]:
    """(detector, size, confidence, candidate) -> tp, wrong, missed, ids, F1, mean scene F1."""
    table: dict[tuple, list] = {}
    for row in found:
        if row["group"] in group and int(row["tp"]) + int(row["missed"]) >= MIN_TRUE:
            key = (row["detector"], row["imgsz"], row["confidence"], row["candidate"])
            table.setdefault(key, []).append(row)
    out = {}
    for key, part in table.items():
        tp, wrong, missed, ids = (
            sum(int(r[k]) for r in part) for k in ("tp", "wrong", "missed", "ids")
        )
        each = [
            2 * int(r["tp"]) / max(2 * int(r["tp"]) + int(r["wrong"]) + int(r["missed"]), 1)
            for r in part
        ]
        out[key] = (
            tp,
            wrong,
            missed,
            ids,
            2 * tp / max(2 * tp + wrong + missed, 1),
            float(np.mean(each)),
            len(part),
        )
    return out


def best_detectors(study: dict) -> list[dict]:
    """Stage 1's best models and sizes on the choosing scenes, by mean F1 per scene: each
    camera counts once, so a crowd with many crossings does not decide alone."""
    table = totals(rows(OUT / "stage1.csv"), {"mot", "train"})
    score: dict[tuple, float] = {}
    for (name, size, _, _), value in table.items():
        score[(name, size)] = max(score.get((name, size), 0.0), value[5])
    ranked = sorted(score, key=score.get, reverse=True)[: study["stage2_detectors"]]
    return [{"name": name, "imgsz": int(size)} for name, size in ranked]


def report(study: dict) -> None:
    found = rows(OUT / "stage1.csv", OUT / "stage2.csv")
    lines = []
    for title, group in (("MOT", {"mot"}), ("PersonPath22 train", {"train"})):
        head = "detector size conf tracker scene_F1 F1 found wrong missed IDs scenes".split()
        lines += [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        table = totals(found, group)
        for key, v in sorted(table.items(), key=lambda kv: -kv[1][5])[:25]:
            cells = [*key, f"{v[5]:.3f}", f"{v[4]:.3f}", v[0], v[1], v[2], v[3], v[6]]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    command = sys.argv[1]
    study = plan()
    scenes = sample(study)
    choosing = {s: g for s, g in scenes.items() if g != "test"}
    candidates = {c.name: c for c in load_candidates()}
    if command == "detect":
        detect(study)
    elif command == "stage1":
        chosen = [candidates[n] for n in study["stage1_candidates"]]
        jobs = [
            (d, 0.1, c, s, g)
            for d in study["detectors"]
            for c in chosen
            for s, g in choosing.items()
        ]
        run(jobs, OUT / "stage1.csv")
    elif command == "stage2":
        best = best_detectors(study)
        print(best, flush=True)
        jobs = [(d, conf, c, s, g) for d in best for conf in study["confidence"]
                for c in candidates.values() for s, g in choosing.items()]  # fmt: skip
        run(jobs, OUT / "stage2.csv")
    elif command == "report":
        report(study)


if __name__ == "__main__":
    main()
