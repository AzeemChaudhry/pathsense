"""Phase 3 step 5 study: does telling people apart by looks recover crossings?

    uv run python scripts/appearance_study embed     # one vector per saved box, GPU if any
    uv run python scripts/appearance_study run       # every variant on every scene
    uv run python scripts/appearance_study people    # do looks tell neighbours apart?
    uv run python scripts/appearance_study switches  # would looks notice real ID switches?

The variants are in configs/study/appearance.yaml, and the results go to
runs/appearance-study/ (not in git). Written up in docs/TRACKER_STUDY.md, Appearance. MOT
data is CC BY-NC-SA: the videos stay local, never in docs.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per worker, for fair timings

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from quality import people_table, study_scene, switches_table  # noqa: E402

from pathsense.config import load_config, scene_source  # noqa: E402
from pathsense.tune.embeddings import cached_embeddings  # noqa: E402
from pathsense.tune.run import Candidate  # noqa: E402

STUDY = Path("configs/study/appearance.yaml")
OUT = Path("runs/appearance-study")
WORKERS = 6  # each holds a scene's labels and boxes; the vectors are shared by memory map


def study() -> dict:
    return yaml.safe_load(STUDY.read_text())


def embed(scenes: list[str]) -> None:
    for name in scenes:
        scene = study_scene(name)
        look = load_config(scene=scene.path).appearance
        video = scene_source(scene.path)
        vectors = cached_embeddings(video, scene.saved, look, scene.saved_path)
        print(f"{name}: {len(vectors)} vectors", flush=True)


def _run(job: tuple[str, str, list[str]]) -> dict:
    name, variant, overrides = job
    candidate = Candidate(variant, "", tuple(overrides))
    score, _ = study_scene(name).run(candidate)
    crossings = score.crossings
    return {
        "variant": variant,
        "scene": name,
        "tp": sum(crossings.tp.values()),
        "wrong": sum(crossings.fp.values()),
        "missed": sum(crossings.fn.values()),
        "ids": score.ids,
        "track_ms": round(score.track_ms, 2),
    }


def table(rows: list[dict], tune: set[str], names: list[str]) -> str:
    lines = [
        "| variant | tune F1 | wrong | missed | check F1 | wrong | missed | IDs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant in names:
        mine = [r for r in rows if r["variant"] == variant]
        cells = [variant]
        for group in (True, False):
            part = [r for r in mine if (r["scene"] in tune) == group]
            tp, wrong, missed = (sum(r[k] for r in part) for k in ("tp", "wrong", "missed"))
            cells += [f"{2 * tp / max(2 * tp + wrong + missed, 1):.3f}", str(wrong), str(missed)]
        cells.append(str(sum(r["ids"] for r in mine)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def tune_f1(rows: list[dict], tune: set[str], variant: str) -> float:
    part = [r for r in rows if r["variant"] == variant and r["scene"] in tune]
    tp, wrong, missed = (sum(r[k] for r in part) for k in ("tp", "wrong", "missed"))
    return 2 * tp / max(2 * tp + wrong + missed, 1)


def run(plan: dict) -> None:
    variants = plan["variants"]
    jobs = [(s, v, o) for v, o in variants.items() for s in plan["scenes"]]
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(_run, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tune = set(plan["tune"])
    chosen = max(plan["chosen_from"], key=lambda v: tune_f1(rows, tune, v))
    text = table(rows, tune, list(variants)) + f"\nChosen on the tune scenes: **{chosen}**\n"
    (OUT / "report.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("embed", "run", "people", "switches"))
    args = parser.parse_args()
    plan = study()
    if args.command == "embed":
        embed(plan["scenes"])
    elif args.command == "run":
        run(plan)
    elif args.command == "people":
        print(people_table(plan["scenes"]))
    else:
        print(switches_table(plan["scenes"]))


if __name__ == "__main__":
    main()
