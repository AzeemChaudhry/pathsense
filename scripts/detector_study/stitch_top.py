"""Does linking by the top of the box help, and what does it cost? From saved boxes.

    uv run python scripts/detector_study/stitch_top.py

Replays the chosen setup with `tracker.stitch_top` off and at each value below, on the
sample clips with listed crossings, the MOT scenes and the PersonPath22 sample. Writes
runs/detector-study/stitch_top.csv and prints wrong, missed and mean scene F1 by group.
"""

from __future__ import annotations

import csv
import dataclasses
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gate import CONFIDENCE, CROWD_LOW_BAR

from pathsense.config import scene_source
from pathsense.tune.cache import cache_path
from pathsense.tune.run import Candidate, _load_scene

FOLDER = Path("data/cache/study/yolo26s-1920")
SAMPLES = "people-walking grocery-store market-square".split()
MOT = "MOT17-02 MOT17-04 MOT17-09 MOT20-01 MOT20-02 MOT20-03 MOT20-05".split()
TOPS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.35)
OUT = Path("runs/detector-study/stitch_top.csv")


def scenes() -> dict[str, str]:
    out = {f"configs/scenes/{name}.yaml": "sample" for name in SAMPLES}
    out |= {f"configs/scenes/mot/{name}.yaml": "mot" for name in MOT}
    for name in ("models.csv", "test.csv"):
        rows = Path(f"runs/detector-study/{name}").read_text().splitlines()
        for row in csv.DictReader(rows):
            if row["group"] != "mot":
                out[f"data/personpath22/scenes/{row['scene']}.yaml"] = row["group"]
    return out


def score(job: tuple[str, str, float]) -> dict:
    scene, group, top = job
    path = cache_path(scene_source(scene), FOLDER)
    loaded = _load_scene(Path(scene), None, path)
    loaded = dataclasses.replace(loaded, saved=loaded.saved.above(CONFIDENCE))
    overrides = (*CROWD_LOW_BAR.overrides, f"tracker.stitch_top={top}")
    result, _ = loaded.run(Candidate(f"top {top}", "", overrides))
    c = result.crossings
    return {
        "scene": Path(scene).stem,
        "group": group,
        "top": top,
        "tp": sum(c.tp.values()),
        "wrong": sum(c.fp.values()),
        "missed": sum(c.fn.values()),
        "f1": round(c.f1(), 4),
        "line": result.line_error,
        "zones": result.count_error - result.line_error,
        "ids": result.ids,
    }


def main() -> None:
    jobs = [(scene, group, top) for scene, group in scenes().items() for top in TOPS]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(score, jobs))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row["tp"] + row["missed"] >= 4 or row["group"] == "sample":
            groups[row["group"], row["top"]].append(row)
    for (group, top), found in sorted(groups.items()):
        wrong, missed = sum(r["wrong"] for r in found), sum(r["missed"] for r in found)
        f1 = np.mean([r["f1"] for r in found])
        line = f"{group:8} top {top:<5} {len(found):3} scenes"
        print(f"{line} wrong {wrong:5} missed {missed:5} F1 {f1:.3f}")


if __name__ == "__main__":
    main()
