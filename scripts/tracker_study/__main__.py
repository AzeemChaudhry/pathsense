"""Phase 3 tracker study: trackers swapped, then our layers stacked on top, scored on MOT.

    uv run python scripts/tracker_study cache    # detector once per scene -> data/cache
    uv run python scripts/tracker_study sweep    # choose settings on the tune scenes
    uv run python scripts/tracker_study run      # every setup on every scene, saved
    uv run python scripts/tracker_study render   # a video of every run
    uv run python scripts/tracker_study report   # tables and charts
    uv run python scripts/tracker_study check    # where the misses come from; matcher rescoring
    uv run python scripts/tracker_study all      # the lot, in that order (not cache)

Everything goes to runs/tracker-study/ (not in git):

    sweep/sweep.csv              every setting tried, per tune scene
    sweep/best.json              the settings chosen for each tracker and layer
    runs/<setup>/<scene>/        tracks.txt (MOT format), events.jsonl, crossings.json,
                                 metrics.json, video.mp4
    results.csv                  one row per setup and scene
    report/                      charts (PNG) and tables (Markdown); misses.md, matcher.md

The study is defined in configs/study/trackers.yaml and written up in
docs/TRACKER_STUDY.md. MOT data is CC BY-NC-SA: the videos stay local, never in docs.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per worker, for fair timings

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from replay import Setup, replay  # noqa: E402

from pathsense.track import accepted_params  # noqa: E402

STUDY = Path("configs/study/trackers.yaml")
OUT = Path("runs/tracker-study")
WORKERS = max(1, (os.cpu_count() or 2) - 2)
POOL: ProcessPoolExecutor


def load_study() -> dict:
    return yaml.safe_load(STUDY.read_text())


Rows = list[dict]


def mean(rows: Rows, key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def objective(rows: Rows) -> tuple[float, float]:
    """Crossing F1 with every crossing on the tune scenes pooled, then IDF1 for ties."""
    tp = sum(r[f"{d}_tp"] for r in rows for d in ("enter", "exit"))
    fp = sum(r[f"{d}_fp"] for r in rows for d in ("enter", "exit"))
    fn = sum(r[f"{d}_fn"] for r in rows for d in ("enter", "exit"))
    return 2 * tp / max(2 * tp + fp + fn, 1), mean(rows, "IDF1")


def write_csv(path: Path, rows: Rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def score_all(setups: list[Setup], scenes: list[str], full: bool, save: bool) -> Rows:
    jobs = [(s, scene) for s in setups for scene in scenes]
    futures = [
        POOL.submit(replay, s, scene, OUT / "runs" / s.name / scene if save else None, full)
        for s, scene in jobs
    ]
    rows = []
    for n, future in enumerate(futures, 1):
        rows.append(future.result())
        if n % 10 == 0 or n == len(futures):
            print(f"  {n}/{len(futures)}", flush=True)
    return rows


def best_of(setups: list[Setup], rows: Rows) -> Setup:
    return max(setups, key=lambda s: objective([r for r in rows if r["setup"] == s.name]))


def sweep() -> None:
    study = load_study()
    tune, rows, best = study["tune"], [], {}
    for name, fixed in study["trackers"].items():
        takes = accepted_params(name)
        grid = {k: v for k, v in study["sweep"].items() if k in takes}
        setups = []
        for values in itertools.product(*grid.values()):
            params = {**fixed, **dict(zip(grid, values, strict=True))}
            label = ",".join(f"{k}={v}" for k, v in zip(grid, values, strict=True))
            setups.append(Setup(f"{name}[{label}]", name, tuple(params.items())))
        print(f"{name}: {len(setups)} settings x {len(tune)} scenes", flush=True)
        table = score_all(setups, tune, full=True, save=False)
        rows += table
        tuned = best_of(setups, table)
        if "iou" in takes:  # then the overlap measure, on the best settings
            variants = [
                Setup(f"{tuned.name}+{m}", name, tuple({**dict(tuned.params), "iou": m}.items()))
                for m in study["sweep_iou"]
                if m != "iou"
            ]
            rows += score_all(variants, tune, full=True, save=False)
            tuned = best_of([tuned, *variants], rows)
        layered = {"stitch_seconds": 0.0, "bounce": 0.0, "guard": None}
        for layer, options in study["layers"].items():
            key, values = next(iter(options.items()))
            field = "stitch_seconds" if layer == "stitch" else key
            setups = [
                Setup(
                    f"{name}-tuned+{layer}={v}",
                    name,
                    tuned.params,
                    **{**layered, field: tuple(v) if isinstance(v, list) else float(v)},
                )
                for v in values
            ]
            if layer == "guard":  # judged with the other two layers on, as it would run
                setups = [
                    Setup(
                        s.name,
                        name,
                        s.params,
                        layered["stitch_seconds"],
                        layered["bounce"],
                        s.guard,
                    )
                    for s in setups
                ]
            table = score_all(setups, tune, full=True, save=False)
            rows += table
            chosen = best_of(setups, table)
            if layer != "guard":
                layered[field] = getattr(chosen, field)
            best.setdefault(name, {})[layer] = getattr(chosen, field)
        best[name]["params"] = dict(tuned.params)
        print(f"  chosen: {best[name]}", flush=True)
    write_csv(OUT / "sweep" / "sweep.csv", rows)
    (OUT / "sweep" / "best.json").write_text(json.dumps(best, indent=1))


def setups() -> list[Setup]:
    """The reference, then for each tracker the ladder from configs/study/trackers.yaml."""
    study = load_study()
    best = json.loads((OUT / "sweep" / "best.json").read_text())
    ref = study["reference"]
    out = [Setup(ref["name"], ref["tracker"], tuple(ref["params"].items()), ref["stitch_seconds"])]
    for name, fixed in study["trackers"].items():
        chosen = best[name]
        for rung in study["ladder"]:
            params = fixed if "default" in rung else chosen["params"]
            out.append(
                Setup(
                    name if "default" in rung else "+".join([f"{name}-tuned", *rung[1:]]),
                    name,
                    tuple(params.items()),
                    chosen["stitch"] if "stitch" in rung else 0.0,
                    chosen["bounce"] if "bounce" in rung else 0.0,
                    tuple(chosen["guard"]) if "guard" in rung else None,
                )
            )
    return out


def run() -> None:
    study = load_study()
    table = score_all(setups(), study["scenes"], full=True, save=True)
    write_csv(OUT / "results.csv", table)
    print(f"{len(table)} runs -> {OUT / 'results.csv'}")


def render() -> None:
    from render import render_run

    folders = sorted(p.parent for p in (OUT / "runs").glob("*/*/metrics.json"))
    for n, _ in enumerate(POOL.map(render_run, folders), 1):
        if n % 10 == 0 or n == len(folders):
            print(f"  {n}/{len(folders)} videos", flush=True)


def report() -> None:
    from report import make_report

    make_report(OUT, load_study())


# The setups the matcher rescoring compares: the reference, each tracker shipped and tuned
# with bounce, and the chosen setup.
RESCORED = ["v0.3.0", "bytetrack-tuned+stitch+bounce"] + [
    f"{t}{s}"
    for t in ("sort", "bytetrack", "botsort", "ocsort", "cbiou")
    for s in ("", "-tuned+bounce")
]


def check() -> None:
    from analysis import misses, rescore

    study = load_study()
    scenes, chosen = study["scenes"], "bytetrack-tuned+stitch+bounce"
    check_scenes = [s for s in scenes if s not in study["tune"]]
    report = OUT / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "misses.md").write_text(misses(OUT, chosen, scenes))
    (report / "matcher.md").write_text(rescore(OUT, RESCORED, scenes, check_scenes, POOL))
    print((report / "misses.md").read_text() + "\n" + (report / "matcher.md").read_text())


def cache() -> None:
    """The detector once per scene, shared with `pathsense tune` (pathsense.tune.cache)."""
    from pathsense.config import load_config, scene_source
    from pathsense.tune import cached_detections

    for scene in load_study()["scenes"]:
        path = Path("configs/scenes/mot") / f"{scene}.yaml"
        saved = cached_detections(scene_source(path), load_config(scene=path).detector)
        print(f"{scene}: {len(saved)} frames, {saved.detect_ms:.1f} ms per frame", flush=True)


def main() -> None:
    commands = {"cache": [cache], "sweep": [sweep], "run": [run], "render": [render],
                "report": [report], "check": [check],
                "all": [sweep, run, render, report, check]}  # fmt: skip
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=commands)
    global POOL
    with ProcessPoolExecutor(WORKERS) as POOL:  # one pool: each worker loads a scene once
        for step in commands[parser.parse_args().command]:
            step()


if __name__ == "__main__":
    main()
