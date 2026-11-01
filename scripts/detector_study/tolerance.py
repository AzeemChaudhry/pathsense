"""How much the listed crossing scores depend on the matching tolerances.

    uv run python scripts/detector_study/tolerance.py

Scores the sample clips, as shipped and with the chosen setup, at three time and place
limits. Times in the scene files come from the chosen run, so tight limits favour it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from gate import CHOSEN, CONFIDENCE, CROWD_LOW_BAR

from pathsense.config import load_config, scene_source
from pathsense.eval import match_marked
from pathsense.eval.crossings import DIRECTIONS
from pathsense.tune.cache import cache_path, cached_detections
from pathsense.tune.run import CURRENT, _load_scene, _place
from pathsense.tune.truth import scene_truth

CLIPS = "people-walking market-square grocery-store".split()
LIMITS = [(2.0, 200.0), (1.0, 100.0), (0.5, 100.0)]
SETUPS = [
    ("as shipped", Path("data/cache"), [], CURRENT, None),
    ("chosen", Path("data/cache/study/yolo26s-1920"), CHOSEN, CROWD_LOW_BAR, CONFIDENCE),
]


def main() -> None:
    for clip in CLIPS:
        scene = f"configs/scenes/{clip}.yaml"
        truth = scene_truth(scene)
        for setup, folder, overrides, candidate, bar in SETUPS:
            config = load_config(scene=scene, overrides=[*overrides, *candidate.overrides])
            saved = cached_detections(scene_source(scene), config.detector, folder)
            saved = saved.above(bar) if bar else saved
            path = cache_path(scene_source(scene), folder)
            loaded = dataclasses.replace(_load_scene(Path(scene), None, path), saved=saved)
            _, result = loaded.run(candidate)
            ours = [e for e in result.events if e.type in DIRECTIONS]
            places = [_place(e, result.tracks) for e in ours]
            for window, reach in LIMITS:
                c = match_marked(ours, places, truth.crossings, saved.fps, window, reach)
                wrong, missed = sum(c.fp.values()), sum(c.fn.values())
                print(clip, setup, window, reach, wrong, missed, f"{c.f1():.3f}")


if __name__ == "__main__":
    main()
