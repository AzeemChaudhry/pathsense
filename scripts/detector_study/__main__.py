"""Phase 3 detector study: model, input size and confidence, scored against labels.

    uv run python scripts/personpath22.py             # scenes from PersonPath22
    uv run python scripts/detector_study detect       # GPU, once per model and size
    uv run python scripts/detector_study stage1       # every model and size (CPU)
    uv run python scripts/detector_study stage2       # the best, with every setting (CPU)
    uv run python scripts/detector_study report       # tables
    uv run python scripts/detector_study test         # the held out scenes, once
    uv run python scripts/detector_study models       # more models (configs/study/models.yaml)
    uv run python scripts/detector_study density      # those results by people per frame

Defined in configs/study/detectors.yaml, written up in docs/DETECTOR_STUDY.md. Saved boxes
go to data/cache/study/, results to runs/detector-study/ (not in git). Every step can be
stopped and run again: it skips what is done. Replays use the code a live video runs.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import sys
from collections import Counter
from typing import NamedTuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pathsense.config import load_config, read_yaml, scene_source  # noqa: E402
from pathsense.tune.cache import cache_path, cached_detections  # noqa: E402
from pathsense.tune.run import _load_scene, load_candidates  # noqa: E402

STUDY = Path("configs/study/detectors.yaml")
MODELS = Path("configs/study/models.yaml")
SCENES = Path("data/personpath22/scenes")
CACHE = Path("data/cache/study")
OUT = Path("runs/detector-study")
WORKERS = 8
# A line can look busy to scripts/personpath22.py yet see too few crossings by the scorer's
# rules, and one crossing then swings a scene's F1. Too few: not scored, for any setup.
MIN_TRUE = 4
TIE = 0.03  # a smaller gap in mean scene F1 is noise (docs/DETECTOR_STUDY.md)
BOOTSTRAP = 2000
FIELDS = "detector imgsz confidence candidate scene group tp wrong missed ids".split()
MODEL_FIELDS = [*FIELDS, "line_error", "zone_error", "ms"]


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


def _label(detector: dict) -> str:
    return detector.get("label", detector["name"])


def _folder(detector: dict) -> Path:
    return CACHE / f"{_label(detector)}-{detector['imgsz']}"


def _config(scene: str, detector: dict):
    sets = [f"detector.name={detector['name']}", f"detector.imgsz={detector['imgsz']}"]
    return load_config(scene=scene, overrides=sets + detector.get("sets", []))


def detect(study: dict, scenes: dict[str, str] | None = None) -> None:
    from pathsense.detect import build_detector

    scenes = scenes or sample(study)
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


def _score(job: tuple) -> dict | None:
    detector, confidence, candidate, scene, group = job
    try:
        loaded = _scene(scene, str(_folder(detector)))
    except FileNotFoundError:
        return None  # this detector never finished the scene
    loaded = dataclasses.replace(loaded, saved=loaded.saved.above(confidence))
    score, result = loaded.run(candidate)
    c = score.crossings
    return {
        "detector": _label(detector),
        "imgsz": detector["imgsz"],
        "confidence": confidence,
        "candidate": candidate.name,
        "scene": Path(scene).stem,
        "group": group,
        "tp": sum(c.tp.values()),
        "wrong": sum(c.fp.values()),
        "missed": sum(c.fn.values()),
        "ids": score.ids,
        "line_error": score.line_error,
        "zone_error": score.count_error - score.line_error,
        "ms": round(loaded.saved.detect_ms + result.track_ms, 1),
    }


def _key(row: dict) -> tuple:
    return tuple(str(row[k]) for k in FIELDS[:5])


def run(jobs: list[tuple], path: Path, fields: list[str] = FIELDS) -> None:
    """Score `jobs`, adding to `path`, skipping those already there."""
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with path.open() as saved:
            done = {_key(row) for row in csv.DictReader(saved)}
    todo = []
    for job in jobs:
        detector, confidence, candidate, scene, _ = job
        names = (_label(detector), detector["imgsz"], confidence, candidate.name, Path(scene).stem)
        if tuple(str(v) for v in names) not in done:
            todo.append(job)
    print(f"{len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    new = not path.exists()
    with path.open("a", newline="") as out, ProcessPoolExecutor(WORKERS) as pool:
        writer = csv.DictWriter(out, fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        for k, row in enumerate(pool.map(_score, todo, chunksize=4)):
            if row:
                writer.writerow(row)
            if k % 200 == 0:
                out.flush()
                print(f"{k}/{len(todo)}", flush=True)


def rows(*paths: Path) -> list[dict]:
    found: dict[tuple, dict] = {}  # a run can be in both stages: once each
    for path in paths:
        if path.exists():
            with path.open() as saved:
                found |= {_key(row): row for row in csv.DictReader(saved)}
    return list(found.values())


class Total(NamedTuple):
    tp: int
    wrong: int
    missed: int
    ids: int
    f1: float
    scene_f1: float
    scenes: int


def totals(found: list[dict], group: set[str]) -> dict[tuple, Total]:
    """(detector, size, confidence, candidate) -> its Total, over the scenes with enough truth."""
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
        each = [_scene_f1(r) for r in part]
        f1 = 2 * tp / max(2 * tp + wrong + missed, 1)
        out[key] = Total(tp, wrong, missed, ids, f1, float(np.mean(each)), len(part))
    return out


def best_detectors(study: dict) -> list[dict]:
    """Stage 1's best models and sizes on the choosing scenes, by mean F1 per scene: each
    camera counts once, so a crowd with many crossings does not decide alone."""
    table = totals(rows(OUT / "stage1.csv"), {"mot", "train"})
    score: dict[tuple, float] = {}
    for (name, size, _, _), value in table.items():
        score[(name, size)] = max(score.get((name, size), 0.0), value.scene_f1)
    ranked = sorted(score, key=score.get, reverse=True)[: study["stage2_detectors"]]
    return [{"name": name, "imgsz": int(size)} for name, size in ranked]


def report(study: dict) -> None:
    found = rows(OUT / "stage1.csv", OUT / "stage2.csv")
    lines = []
    for title, group in (("MOT", {"mot"}), ("PersonPath22 train", {"train"})):
        head = "detector size conf tracker scene_F1 F1 found wrong missed IDs scenes".split()
        lines += [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        table = totals(found, group)
        for key, v in sorted(table.items(), key=lambda kv: -kv[1].scene_f1)[:25]:
            cells = [*key, f"{v.scene_f1:.3f}", f"{v.f1:.3f}", *v[:4], v.scenes]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def _scene_f1(row: dict) -> float:
    tp, wrong, missed = int(row["tp"]), int(row["wrong"]), int(row["missed"])
    return 2 * tp / max(2 * tp + wrong + missed, 1)


def paired(a: np.ndarray, b: np.ndarray) -> str:
    """How `a` differs from `b`, scene by scene, with a bootstrap 95 % interval. Scenes hold
    about 16 crossings each, so a gap under TIE, or an interval over 0, decides nothing. Read
    "tie" as no difference found, not as equal, and "differs" as soft: `a` or `b` is often the
    best of several, which favours a gap. Under about five scenes the interval means little."""
    gaps = a - b
    picks = np.random.default_rng(0).integers(len(gaps), size=(BOOTSTRAP, len(gaps)))
    low, high = np.percentile(gaps[picks].mean(1), [2.5, 97.5])
    tied = abs(gaps.mean()) < TIE or low <= 0 <= high
    return f"{gaps.mean():+.3f} [{low:+.3f}, {high:+.3f}] {'tie' if tied else 'differs'}"


def compare(path: Path, pairs: list[list[str]], listed: list[str]) -> None:
    """Each model at its own best setting, on the scenes every model finished: a model that
    failed part way must not be scored on an easier set. Best of many settings on these
    scenes, so read the gaps, and confirm any winner on held out scenes."""
    found = [r for r in rows(path) if int(r["tp"]) + int(r["missed"]) >= MIN_TRUE]
    found = [r for r in found if r["detector"] in listed]  # a dropped model may have rows
    done: dict[str, set[str]] = {name: set() for name in listed}
    for row in found:
        done[row["detector"]].add(row["scene"])
    started = {name: scenes for name, scenes in done.items() if scenes}
    if not started:
        print("nothing scored yet", flush=True)
        return
    every = set.union(*started.values())
    short = [f"{n} ({len(s)} of {len(every)})" for n, s in done.items() if len(s) < len(every)]
    common = set.intersection(*started.values())
    print(f"{len(common)} scenes; short: {', '.join(short) or 'none'}", flush=True)
    if len(common) < 2:
        print("too few shared scenes to compare", flush=True)
        return
    found = [r for r in found if r["scene"] in common]
    best: dict[str, tuple] = {}
    for key, total in totals(found, {"mot", "train"}).items():
        if total.scene_f1 > best.get(key[0], (0.0,))[0]:
            best[key[0]] = (total.scene_f1, key)
    chosen = {
        name: sorted(
            (
                r
                for r in found
                if (r["detector"], r["imgsz"], r["confidence"], r["candidate"]) == key
            ),
            key=lambda r: r["scene"],
        )
        for name, (_, key) in best.items()
    }
    each = {name: np.array([_scene_f1(r) for r in part]) for name, part in chosen.items()}
    leader = max(best, key=lambda name: best[name][0])
    wins = Counter(np.argmax(np.stack([each[n] for n in best]), axis=0).tolist())
    print(
        f"{'model':16} scene_F1 wins line_err zone_err  ms  against {leader}, best setting",
        flush=True,
    )
    for k, name in enumerate(best):
        gap = "" if name == leader else paired(each[name], each[leader])
        key = best[name][1]
        line = np.median([int(r["line_error"]) for r in chosen[name]])
        zone = np.median([int(r["zone_error"]) for r in chosen[name]])
        ms = np.mean([float(r["ms"]) for r in chosen[name]])
        row = f"{name:16} {best[name][0]:.3f} {wins[k]:6} {line:7.0f} {zone:8.0f} {ms:5.0f}"
        print(f"{row}  {gap}  {key[2]} {key[3]}", flush=True)
    for a, b in pairs:
        if a in each and b in each:
            print(f"{a} against {b}: {paired(each[a], each[b])}", flush=True)


def band_report(found: list[dict], bands: dict[str, set[str]]) -> None:
    """Per density band, each model's setting chosen with the scored scene left out, so a
    setting cannot be flattered by the scene it is scored on, against its one pooled setting.
    A mean over scenes hides a crowd: sparse scenes outnumber it and pick the setting."""
    table: dict[str, dict[tuple, dict[str, dict]]] = {}
    for row in found:
        setting = (row["confidence"], row["candidate"])
        table.setdefault(row["detector"], {}).setdefault(setting, {})[row["scene"]] = row

    def mean(rows_of: dict[str, dict], scenes: list[str]) -> float:
        return float(np.mean([_scene_f1(rows_of[s]) for s in scenes]))

    for band, members in bands.items():
        scenes = sorted(members)
        print(f"\n{band}: {len(scenes)} scenes", flush=True)
        if len(scenes) < 2:
            continue
        held: dict[str, np.ndarray] = {}
        lines = {}
        for name, settings in table.items():
            kept = []
            for scene in scenes:
                others = [s for s in scenes if s != scene]
                pick = max(settings, key=lambda k: mean(settings[k], others))
                kept.append(_scene_f1(settings[pick][scene]))
            held[name] = np.array(kept)
            fit = max(settings, key=lambda k: mean(settings[k], scenes))
            pooled = max(settings, key=lambda k: mean(settings[k], sorted(settings[k])))
            lines_ = [int(settings[fit][s]["line_error"]) for s in scenes]
            zones = [int(settings[fit][s]["zone_error"]) for s in scenes]
            ids = np.mean([int(settings[fit][s]["ids"]) for s in scenes])
            lines[name] = (
                mean(settings[pooled], scenes),
                np.median(lines_),
                np.median(zones),
                ids,
                fit,
            )
        leader = max(held, key=lambda name: held[name].mean())
        print(
            f"{'model':14} left_out pooled line_err zone_err IDs  against {leader}, setting",
            flush=True,
        )
        for name, (pooled, line, zone, ids, fit) in lines.items():
            gap = "" if name == leader else paired(held[name], held[leader])
            row = f"{name:14} {held[name].mean():.3f}    {pooled:.3f}  {line:6.0f} {zone:8.0f}"
            row = f"{row} {ids:5.0f}"
            print(f"{row}  {gap}  {fit[0]} {fit[1]}", flush=True)


def density(scenes: dict[str, str]) -> None:
    """The sweep's results by people per frame, from `models.csv`; no GPU."""
    low, high = yaml.safe_load(MODELS.read_text())["density"]
    found = [r for r in rows(OUT / "models.csv") if int(r["tp"]) + int(r["missed"]) >= MIN_TRUE]
    finished = Counter(r["detector"] for r in found)
    found = [r for r in found if finished[r["detector"]] == max(finished.values())]
    scored = {r["scene"] for r in found}
    bands: dict[str, set[str]] = {
        f"sparse, under {low}": set(),
        f"busy, {low} to {high}": set(),
        f"crowd, {high} or more": set(),
    }
    names = list(bands)
    for scene in scenes:
        stem = Path(scene).stem
        if stem in scored:
            people = people_per_frame(Path(read_yaml(Path(scene))["truth"]["labels"]))
            bands[names[(people >= low) + (people >= high)]].add(stem)
    band_report(found, bands)


def models(study: dict, scenes: dict[str, str], candidates: dict) -> None:
    """One model at a time, cheapest first: a run stopped early has every finished model."""
    extra = yaml.safe_load(MODELS.read_text())
    chosen = [candidates[name] for name in extra["candidates"]]
    for detector in extra["detectors"]:
        detect({**study, "detectors": [detector]}, scenes)
        jobs = [(detector, conf, c, s, g) for conf in extra["confidence"] for c in chosen
                for s, g in scenes.items()]  # fmt: skip
        run(jobs, OUT / "models.csv", MODEL_FIELDS)
        compare(OUT / "models.csv", extra["pairs"], [_label(d) for d in extra["detectors"]])


def main() -> None:
    command = sys.argv[1]
    study = plan()
    scenes = sample(study)
    choosing = {s: g for s, g in scenes.items() if g != "test"}
    candidates = {c.name: c for c in load_candidates()}
    if command == "detect":
        detect(study)
    elif command == "models":
        models(study, choosing, candidates)
    elif command == "density":
        density(choosing)
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
    elif command == "test":
        held = {s: g for s, g in scenes.items() if g == "test"}
        jobs = [
            (d, final["confidence"], candidates[final["candidate"]], s, g)
            for final in study["final"]
            for d in [final]
            for s, g in held.items()
        ]
        run(jobs, OUT / "test.csv")
        table = totals(rows(OUT / "test.csv"), {"test"})
        for key, v in sorted(table.items(), key=lambda kv: -kv[1].scene_f1):
            print(key, f"scene F1 {v.scene_f1:.3f} F1 {v.f1:.3f}", *v[:4], f"scenes {v.scenes}")
    elif command == "report":
        report(study)


if __name__ == "__main__":
    main()
