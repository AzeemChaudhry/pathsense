"""Phase 3's gate table: all eleven scenes, as shipped and with the chosen detector.

    uv run python scripts/detector_study/gate.py

Writes runs/detector-study/gate.md. The chosen setup is what `pathsense run --auto` uses,
on each scene's own line and zones: detector, confidence and tracker all change. subway has
no truth, so it shows counts and nothing to score them
against. FPS is the detector and everything after it, one frame at a time.
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pathsense.config import load_config, scene_source
from pathsense.detect import build_detector
from pathsense.eval import tracking_metrics
from pathsense.eval.mot import write_mot
from pathsense.io import VideoSource
from pathsense.pipeline import Pipeline
from pathsense.tune.cache import cache_path, cached_detections
from pathsense.tune.replay import no_detector, replay
from pathsense.tune.run import CURRENT, _load_scene, load_candidates

MOT = "MOT17-02 MOT17-04 MOT17-09 MOT20-01 MOT20-02 MOT20-03 MOT20-05".split()
SCENES = [f"configs/scenes/mot/{name}.yaml" for name in MOT]
SAMPLES = "people-walking grocery-store market-square subway".split()
SCENES += [f"configs/scenes/{name}.yaml" for name in SAMPLES]
CHOSEN = ["detector.name=yolo26s", "detector.imgsz=1920"]
CONFIDENCE = 0.3
CROWD_LOW_BAR = next(c for c in load_candidates() if c.name == "crowd, low bar")
CPU_FRAMES = 30
OUT = Path("runs/detector-study")


def _saved(scene: str, overrides: list[str], folder: Path):
    config = load_config(scene=scene, overrides=overrides)
    return cached_detections(scene_source(scene), config.detector, folder), config


def _row(job: tuple[str, str, Path]) -> dict:
    scene, setup, folder = job
    overrides = CHOSEN if setup == "chosen" else []
    chosen = setup != "as shipped"
    overrides = [*overrides, *CROWD_LOW_BAR.overrides] if chosen else overrides
    saved, config = _saved(scene, overrides, folder)
    saved = saved.above(CONFIDENCE) if chosen else saved
    row = {"scene": Path(scene).stem, "setup": setup, "gpu_fps": 1000 / (saved.detect_ms + 1)}
    if "subway" in scene:
        pipeline = Pipeline.from_config(config, saved.fps, detector=no_detector)
        result = replay(pipeline, saved)
        row |= {"ours": f"{pipeline.line_counter.entered} / {pipeline.line_counter.exited}"}
        row["gpu_fps"] = 1000 / (saved.detect_ms + result.track_ms)
        row["ids"] = result.ids
        return row
    loaded = _load_scene(Path(scene), None, cache_path(scene_source(scene), folder))
    if chosen:
        loaded = dataclasses.replace(loaded, saved=saved)
    score, result = loaded.run(CROWD_LOW_BAR if chosen else CURRENT)
    enter, exit_ = (score.counts.get(k, 0) for k in ("enter", "exit"))
    true = f"{score.truth.get('enter', '?')} / {score.truth.get('exit', '?')}"
    row |= {
        "ours": f"{enter} / {exit_}",
        "true": true,
        "error": score.count_error,
        "ids": score.ids,
    }
    row["gpu_fps"] = 1000 / (saved.detect_ms + result.track_ms)
    c = score.crossings
    if c is not None:
        row |= {"wrong": sum(c.fp.values()), "missed": sum(c.fn.values())}
        for direction in ("enter", "exit"):
            row[direction] = f"{c.precision(direction):.2f} / {c.recall(direction):.2f}"
        with tempfile.TemporaryDirectory() as folder_:
            tracks = Path(folder_) / "tracks.txt"
            write_mot(tracks, sorted(result.tracks.items()))
            metrics = tracking_metrics(loaded_truth(scene), tracks)
        row |= {k: metrics[k] for k in ("HOTA", "IDF1", "MOTA", "IDSW", "Frag")}
    return row


def loaded_truth(scene: str) -> Path:
    return Path(f"data/mot/gt/{Path(scene).stem}.txt")


def cpu_fps(name: str, imgsz: int) -> float:
    config = load_config(
        overrides=[f"detector.name={name}", f"detector.imgsz={imgsz}", "detector.device=cpu"]
    )
    detector = build_detector(config.detector)
    with VideoSource("data/samples/people-walking.mp4") as source:
        images = [frame.image for _, frame in zip(range(CPU_FRAMES), source, strict=False)]
    detector(images[0])  # warm up
    start = time.perf_counter()
    for image in images:
        detector(image)
    return CPU_FRAMES / (time.perf_counter() - start)


def main() -> None:
    jobs = []
    for scene in SCENES:
        jobs.append((scene, "as shipped", Path("data/cache")))
        jobs.append((scene, "chosen", Path("data/cache/study/yolo26s-1920")))
    for scene in SCENES:  # any box not saved yet, on the GPU, before the workers start
        _saved(scene, [], Path("data/cache"))
        _saved(scene, CHOSEN, Path("data/cache/study/yolo26s-1920"))
    with ProcessPoolExecutor(5) as pool:
        rows = list(pool.map(_row, jobs))
    keys = "scene setup ours true error wrong missed enter exit".split()
    keys += "HOTA IDF1 MOTA IDSW Frag ids gpu_fps".split()
    lines = ["| " + " | ".join(keys) + " |", "|" + " --- |" * len(keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            if key in ("HOTA", "IDF1", "MOTA") and value != "":
                value = f"{100 * value:.1f}"
            elif isinstance(value, float):
                value = f"{value:.0f}" if key != "gpu_fps" else f"{value:.1f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "CPU FPS, detector only, 30 frames of people-walking:", ""]
    for name, imgsz in (("yolo26n", 640), ("yolo26n", 1280), ("yolo26n", 1920), ("yolo26s", 1920)):
        lines.append(f"- {name} at {imgsz}: {cpu_fps(name, imgsz):.1f}")
    text = "\n".join(lines) + "\n"
    (OUT / "gate.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
