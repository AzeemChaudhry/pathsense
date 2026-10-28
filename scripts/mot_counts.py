"""Score line and zone counts against MOT ground truth.

MOT17 and MOT20 label every person in every frame with a box and an ID, drawn by hand. Fed
through the same line and zone counters as the pipeline, those tracks give the true counts
for any line or zone, so a scene's count error needs no hand counting.

The data is CC BY-NC-SA: evaluation only, never redistributed, no frames in public docs.

    uv run python scripts/mot_counts.py prepare     # unzip data/mot/*.zip, make the videos
    uv run python scripts/mot_counts.py score configs/scenes/mot/*.yaml --set detector.imgsz=1280

`prepare` expects MOT16.zip (frames), MOT17Labels.zip (the corrected labels for the same
sequences) and MOT20.zip in data/mot/, from https://motchallenge.net/data/. It writes
data/mot/videos/<sequence>.mp4 and data/mot/gt/<sequence>.txt for each sequence below.
"""

from __future__ import annotations

import argparse
import configparser
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import supervision as sv

from pathsense.config import load_config, scene_source
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
# Pedestrian, person on a vehicle, static person: everyone a person counter should see.
# Distractors (reflections, mannequins) and non-people are left out.
PEOPLE = {1, 2, 7}


def prepare() -> None:
    for name in ("MOT16", "MOT17Labels", "MOT20"):
        archive, folder = ROOT / f"{name}.zip", ROOT / name
        if not folder.exists():
            print(f"unzipping {archive}")
            with zipfile.ZipFile(archive) as z:
                z.extractall(folder)
            # Some archives hold a top folder of their own name: lift its contents.
            inner = folder / name
            if inner.is_dir():
                for item in inner.iterdir():
                    item.rename(folder / item.name)
                inner.rmdir()
    (ROOT / "videos").mkdir(exist_ok=True)
    (ROOT / "gt").mkdir(exist_ok=True)
    for sequence, (frames, labels) in SEQUENCES.items():
        info = configparser.ConfigParser()
        info.read(ROOT / frames / "seqinfo.ini")
        fps = info["Sequence"]["frameRate"]
        shutil.copy(ROOT / labels / "gt/gt.txt", ROOT / "gt" / f"{sequence}.txt")
        video = ROOT / "videos" / f"{sequence}.mp4"
        if not video.exists():
            print(f"making {video} at {fps} FPS")
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-framerate",
                    fps,
                    "-i",
                    str(ROOT / frames / "img1/%06d.jpg"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "16",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )


def ground_truth(path: Path) -> dict[int, sv.Detections]:
    """Frame index (from 0) -> the labelled people in it, with their IDs as tracker_id."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    rows = rows[np.isin(rows[:, 7], list(PEOPLE))]
    frames = {}
    for frame in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == frame]
        xyxy = np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])
        frames[frame - 1] = sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=np.ones(len(r), np.float32),
            class_id=np.zeros(len(r), int),
            tracker_id=r[:, 1].astype(int),
        )
    return frames


def totals(line, zones) -> dict[str, int]:
    counts = {}
    if line:
        counts["line in"], counts["line out"] = line.entered, line.exited
    for zone in zones:
        counts[f"{zone.name} in"], counts[f"{zone.name} out"] = zone.entered, zone.exited
    return counts


def score(scene: Path, overrides: list[str]) -> None:
    config = load_config(overrides=overrides, scene=scene)
    video = Path(scene_source(scene))
    truth_tracks = ground_truth(ROOT / "gt" / f"{video.stem}.txt")
    with VideoSource(str(video)) as source:
        pipeline = Pipeline.from_config(config, frame_rate=source.fps)
        lost = pipeline.zones[0].lost if pipeline.zones else pipeline.line_counter.lost
        truth_line, truth_zones = build_counters(config.analytics, lost)
        truth_counters = ([truth_line] if truth_line else []) + truth_zones
        empty = sv.Detections.empty()
        empty.tracker_id = np.array([], int)
        ours_ids = set()
        for frame in source:
            result = pipeline.process(frame)
            if result.tracked.tracker_id is not None:
                ours_ids.update(int(i) for i in result.tracked.tracker_id)
            for counter in truth_counters:
                counter.update(truth_tracks.get(frame.index, empty), frame.timestamp, frame.index)
    pipeline.finish()
    for counter in truth_counters:
        counter.finish()
    truth = totals(truth_line, truth_zones)
    ours = totals(pipeline.line_counter, pipeline.zones)
    people = len({int(i) for d in truth_tracks.values() for i in d.tracker_id})
    print(f"\n{video.stem}  ({scene})")
    print(f"  {'':16}{'truth':>7}{'ours':>7}{'error':>8}")
    print(f"  {'people (IDs)':16}{people:>7}{len(ours_ids):>7}{len(ours_ids) - people:>+8}")
    for key, true in truth.items():
        print(f"  {key:16}{true:>7}{ours[key]:>7}{ours[key] - true:>+8}")


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
