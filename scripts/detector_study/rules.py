"""Can a scan choose better than one setup? One split on one measure, with scenes held out.

    uv run python scripts/detector_study/rules.py

Needs stage1 and stage2 of the study. Measures come from the analyzer's own scan, made here
from the saved `yolo26n` at 1280 boxes. Prints the best one setup, the best possible choice
per scene, the rule fit on every scene, and that rule's score over five held out folds.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import runpy
from pathlib import Path

import numpy as np

from pathsense.config import scene_source
from pathsense.tune.analyze import MEASURES, SCAN_SECONDS, load_rules
from pathsense.tune.cache import SavedDetections, cache_path
from pathsense.tune.profile import scene_profile

study = runpy.run_path("scripts/detector_study/__main__.py", run_name="study")
OUT = Path("runs/detector-study")
SHORTLIST = 30  # setups tried in a rule, best first
GAIN = 0.005  # a rule must add this to the mean F1 to count
NAMES = sorted(MEASURES)


def measures(scene: str, frames: int) -> list[float]:
    video = scene_source(scene)
    saved = SavedDetections.load(cache_path(video, "data/cache/study/yolo26n-1280"))
    count = min(len(saved), int(SCAN_SECONDS * saved.fps))
    keep = np.unique(np.linspace(0, count - 1, frames).astype(int))
    s = saved.starts
    rows = np.concatenate([np.arange(s[k], s[k + 1]) for k in keep])
    part = dataclasses.replace(
        saved,
        xyxy=saved.xyxy[rows],
        confidence=saved.confidence[rows],
        counts=saved.counts[keep],
        index=saved.index[keep],
        timestamp=saved.timestamp[keep],
    )
    profile = scene_profile(part)
    return [float(np.nan_to_num(getattr(profile, name))) for name in NAMES]


def f1(row: dict) -> float:
    tp, wrong, missed = int(row["tp"]), int(row["wrong"]), int(row["missed"])
    return 2 * tp / max(2 * tp + wrong + missed, 1)


def best_rule(scores, x, seen, top):
    """(mean F1, setup below, setup at or above, measure, threshold) on the scenes `seen`."""
    base = scores[:, seen].mean(1).argmax()
    best = (scores[base, seen].mean(), base, base, 0, 0.0)
    for a, b in itertools.permutations(top, 2):
        for measure in range(x.shape[1]):
            for t in np.unique(np.percentile(x[seen, measure], [25, 33, 50, 67, 75])):
                low = x[seen, measure] < t
                if low.sum() < 6 or (~low).sum() < 6:
                    continue
                value = (scores[a, seen][low].sum() + scores[b, seen][~low].sum()) / len(seen)
                if value > best[0] + GAIN:
                    best = (value, a, b, measure, t)
    return best


def main() -> None:
    plan = study["plan"]()
    groups = {Path(s).stem: s for s, g in study["sample"](plan).items() if g != "test"}
    found: dict[tuple, dict] = {}
    for name in ("stage1", "stage2"):
        with (OUT / f"{name}.csv").open() as saved:
            for row in csv.DictReader(saved):
                found[tuple(row[k] for k in study["FIELDS"][:5])] = row
    scored = {
        k[4] for k, r in found.items() if int(r["tp"]) + int(r["missed"]) >= study["MIN_TRUE"]
    }
    scenes = sorted(scored & set(groups))
    setups = sorted({k[:4] for k in found})
    setups = [c for c in setups if all((*c, s) in found for s in scenes)]
    scores = np.array([[f1(found[(*c, s)]) for s in scenes] for c in setups])
    frames = load_rules().frames
    x = np.array([measures(groups[s], frames) for s in scenes])
    single = scores.mean(1).argmax()
    print(f"{len(setups)} setups, {len(scenes)} scenes")
    print(f"best one setup {setups[single]}: {scores[single].mean():.3f}")
    print(f"best per scene: {scores.max(0).mean():.3f}")
    top = list(np.argsort(-scores.mean(1))[:SHORTLIST])
    everyone = np.arange(len(scenes))
    value, a, b, m, t = best_rule(scores, x, everyone, top)
    print(f"fit on all: {value:.3f}: below {NAMES[m]} {t:.1f} {setups[a]}, else {setups[b]}")
    got, one = [], []
    for fold in np.array_split(np.random.default_rng(0).permutation(len(scenes)), 5):
        seen = np.setdiff1d(everyone, fold)
        _, a, b, m, t = best_rule(scores, x, seen, top)
        base = scores[:, seen].mean(1).argmax()
        got += [scores[a if x[k, m] < t else b, k] for k in fold]
        one += [scores[base, k] for k in fold]
        print(f"  fold: below {NAMES[m]} {t:.1f}: {setups[a][:2]}, else {setups[b][:2]}")
    print(f"held out: rule {np.mean(got):.3f}, one setup {np.mean(one):.3f}")


if __name__ == "__main__":
    main()
