"""Score line and zone counts against MOT ground truth.

MOT17 and MOT20 label every person in every frame with a box and an ID, drawn by hand. Fed
through the same line and zone counters as the pipeline, those tracks give the true counts
for any line or zone, so a scene's count error needs no hand counting.

The data is CC BY-NC-SA: evaluation only, never redistributed, no frames in public docs.

    uv run python scripts/mot_counts.py prepare     # unzip data/mot/*.zip, make the videos
    uv run python scripts/mot_counts.py score configs/scenes/mot/*.yaml --set detector.imgsz=1280

Two truths are printed. `truth` is every labelled person, also while fully hidden: what
really happened. `seen` keeps only boxes at least `SEEN` visible: the best a counter that
must see people could do. Truth minus seen is the scene's limit (people hidden behind
others); seen minus ours is our error. Truth uses the feet of the labelled box as they are
(no hidden legs estimate). Classes 2 and 7 (person on a vehicle, static person) are flagged
"ignore" by the benchmark but are people, so they count.

`prepare` expects MOT16.zip (frames), MOT17Labels.zip (the corrected labels for the same
sequences) and MOT20.zip in data/mot/, from https://motchallenge.net/data/. It writes
data/mot/videos/<sequence>.mp4 and data/mot/gt/<sequence>.txt for each sequence below.
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import supervision as sv

from pathsense.config import load_config, scene_source
from pathsense.eval import ground_truth
from pathsense.io import VideoSource
from pathsense.pipeline import Pipeline, build_counters

ROOT = Path("data/mot")
# Fixed cameras only: a moving camera has no fixed line or zone to count at.
SEQUENCES = {
    "MOT17-02": ("MOT16/train/MOT16-02", "MOT17Labels/train/MOT17-02-FRCNN"),
    "MOT17-04": ("MOT16/train/MOT16-04", "MOT17Labels/train/MOT17-04-FRCNN"),
    "MOT17-09": ("MOT16/train/MOT16-09", "MOT17Labels/train/MOT17-09-FRCNN"),
    "MOT20-01": ("MOT20/train/MOT20-01", "MOT20/train/MOT20-01"),
    "MOT20-02": ("MOT20/train/MOT20-02", "MOT20/train/MOT20-02"),
    "MOT20-03": ("MOT20/train/MOT20-03", "MOT20/train/MOT20-03"),
    "MOT20-05": ("MOT20/train/MOT20-05", "MOT20/train/MOT20-05"),
}
SEEN = 0.25  # visibility a box needs for the `seen` truth


def prepare() -> None:
    for name in ("MOT16", "MOT17Labels", "MOT20"):
        archive, folder = ROOT / f"{name}.zip", ROOT / name
        if folder.exists():
            continue
        print(f"unzipping {archive}")
        partial = ROOT / f"{name}.partial"
        shutil.rmtree(partial, ignore_errors=True)  # left by an interrupted run
        with zipfile.ZipFile(archive) as z:
            z.extractall(partial)
        inner = partial / name  # some archives hold a top folder of their own name
        (inner if inner.is_dir() else partial).rename(folder)
        shutil.rmtree(partial, ignore_errors=True)
    (ROOT / "videos").mkdir(exist_ok=True)
    (ROOT / "gt").mkdir(exist_ok=True)
    for sequence, (frames, labels) in SEQUENCES.items():
        info = configparser.ConfigParser()
        info.read(ROOT / frames / "seqinfo.ini")
        fps = info["Sequence"]["frameRate"]
        shutil.copy(ROOT / labels / "gt/gt.txt", ROOT / "gt" / f"{sequence}.txt")
        video = ROOT / "videos" / f"{sequence}.mp4"
        if video.exists():
            continue
        print(f"making {video} at {fps} FPS")
        partial = video.with_suffix(".partial.mp4")  # renamed once ffmpeg has finished
        command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps]
        command += ["-i", str(ROOT / frames / "img1/%06d.jpg"), "-c:v", "libx264"]
        # x264 needs even sizes; padding right and bottom keeps the labels' coordinates.
        command += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        command += ["-crf", "16", "-pix_fmt", "yuv420p", str(partial)]
        subprocess.run(command, check=True)
        partial.rename(video)


class Tally:
    """One set of counters and what they saw: IDs, events, occupancy per frame."""

    def __init__(self, line, zones) -> None:
        self.line, self.zones = line, zones
        self.ids: set[int] = set()
        self.dwell = {zone.name: 0.0 for zone in zones}
        self.inside = {zone.name: 0 for zone in zones}  # summed over frames
        self.frames = 0

    def update(self, tracked: sv.Detections, events) -> None:
        if tracked.tracker_id is not None:
            self.ids.update(int(i) for i in tracked.tracker_id)
        self.add(events)
        self.frames += 1
        for zone in self.zones:
            self.inside[zone.name] += zone.occupancy

    def add(self, events) -> None:
        for event in events:
            if event.type == "dwell":
                self.dwell[event.zone] += event.duration

    def feed(self, tracked: sv.Detections, t: float, frame: int) -> None:
        counters = ([self.line] if self.line else []) + self.zones
        self.update(tracked, [e for c in counters for e in c.update(tracked, t, frame)])

    def finish(self) -> None:
        counters = ([self.line] if self.line else []) + self.zones
        self.add([e for c in counters for e in c.finish()])

    def rows(self) -> dict[str, float]:
        out: dict[str, float] = {"people (IDs)": len(self.ids)}
        if self.line:
            out["line in"], out["line out"] = self.line.entered, self.line.exited
        for zone in self.zones:
            out[f"{zone.name} in"], out[f"{zone.name} out"] = zone.entered, zone.exited
            out[f"{zone.name} dwell s"] = round(self.dwell[zone.name], 1)
            out[f"{zone.name} mean inside"] = round(self.inside[zone.name] / max(self.frames, 1), 1)
        return out


def score(scene: Path, overrides: list[str]) -> None:
    config = load_config(overrides=overrides, scene=scene)
    video = Path(scene_source(scene))
    labels = ROOT / "gt" / f"{video.stem}.txt"
    tracks = {"truth": ground_truth(labels), "seen": ground_truth(labels, SEEN)}
    # The labelled boxes already end at the feet: no hidden legs estimate for truth.
    analytics = dataclasses.replace(config.analytics, hidden_legs=0.0)
    empty = sv.Detections.empty()
    empty.tracker_id = np.array([], int)
    with VideoSource(str(video)) as source:
        pipeline = Pipeline.from_config(config, frame_rate=source.fps)
        stitch = config.tracker.stitch_seconds
        lost = max(pipeline.tracker.lost_seconds, stitch) + 1 / source.fps  # as the pipeline
        truths = {name: Tally(*build_counters(analytics, lost)) for name in tracks}
        ours = Tally(pipeline.line_counter, pipeline.zones)
        for frame in source:
            result = pipeline.process(frame)
            ours.update(result.tracked, result.events)
            for name, tally in truths.items():
                tally.feed(tracks[name].get(frame.index, empty), frame.timestamp, frame.index)
    ours.add(pipeline.finish())
    for tally in truths.values():
        tally.finish()
    truth, seen, mine = truths["truth"].rows(), truths["seen"].rows(), ours.rows()
    print(f"\n{video.stem}  ({scene})")
    print(f"  {'':22}{'truth':>8}{'seen':>8}{'ours':>8}{'error':>9}")
    for key, true in truth.items():
        print(f"  {key:22}{true:>8g}{seen[key]:>8g}{mine[key]:>8g}{mine[key] - true:>+9.4g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    scores = sub.add_parser("score")
    scores.add_argument("scenes", nargs="+", type=Path)
    scores.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        for scene in args.scenes:
            score(scene, args.overrides)


if __name__ == "__main__":
    main()
