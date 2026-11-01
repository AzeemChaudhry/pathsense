"""A first look at models and settings, in minutes: boxes against labels on sampled frames.

    uv run python scripts/detector_study/frames.py

Scores FRAMES evenly spaced frames from each of three scenes, box by box at IoU 0.5, each
setup at each confidence. Not crossings: use it to choose what `detector_study models`
spends GPU time on. Results: docs/DETECTOR_STUDY.md, Head and overlap.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from pathsense.config import load_config
from pathsense.detect import build_detector

SCENES = ["MOT20-05", "MOT17-04", "MOT17-09"]
FRAMES = 40
OVERLAP = 0.5
CONFIDENCE = (0.1, 0.2, 0.3, 0.4)
FREE, LOOSE, TIGHT = ["detector.nms=false"], ["detector.iou=0.9"], ["detector.iou=0.5"]
SETUPS = {
    "yolo26n": [("", []), (" free", FREE), (" iou.5", TIGHT), (" iou.9", LOOSE)],
    "yolo26s": [("", []), (" free", FREE), (" iou.5", TIGHT), (" iou.9", LOOSE)],
    "yolo11s": [("", [])],
    "yolo26m": [("", []), (" free", FREE), (" iou.9", LOOSE)],
    "yolo11m": [("", [])],
    "yolo26l": [("", []), (" free", FREE)],
    "yolo26x": [("", []), (" free", FREE)],
}
OUT = Path("runs/detector-study/frames.csv")


def sampled(scene: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(frame, its labelled people as xyxy), FRAMES of them."""
    files = glob.glob(f"data/mot/**/{scene}-FRCNN/gt/gt.txt", recursive=True)
    files = files or glob.glob(f"data/mot/**/{scene}/gt/gt.txt", recursive=True)
    labels = np.loadtxt(files[0], delimiter=",")
    labels = labels[(labels[:, 6] == 1) & (labels[:, 7] == 1)]  # people who count
    capture = cv2.VideoCapture(f"data/mot/videos/{scene}.mp4")
    last = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    out = []
    for index in np.linspace(0, last, FRAMES).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        _, frame = capture.read()
        boxes = labels[labels[:, 0] == index + 1, 2:6].copy()
        boxes[:, 2:] += boxes[:, :2]
        out.append((frame, boxes))
    capture.release()
    return out


def hits(found: np.ndarray, truth: np.ndarray) -> int:
    """People matched one to one."""
    if len(found) == 0 or len(truth) == 0:
        return 0
    overlap = sv.box_iou_batch(found, truth)
    rows, columns = linear_sum_assignment(-overlap)
    return int((overlap[rows, columns] >= OVERLAP).sum())


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {scene: sampled(scene) for scene in SCENES}
    with OUT.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow("model scene confidence tp wrong missed count_error".split())
        for name, setups in SETUPS.items():
            for suffix, sets in setups:
                overrides = [f"detector.name={name}", "detector.imgsz=1920", *sets]
                detector = build_detector(load_config(overrides=overrides).detector)
                for scene, frames in data.items():
                    boxes = [detector(frame) for frame, _ in frames]
                    for bar in CONFIDENCE:
                        tp = wrong = missed = error = 0
                        for found, (_, truth) in zip(boxes, frames, strict=True):
                            kept = found.xyxy[found.confidence >= bar]
                            hit = hits(kept, truth)
                            tp, wrong = tp + hit, wrong + len(kept) - hit
                            missed += len(truth) - hit
                            error += abs(len(kept) - len(truth))
                        writer.writerow([name + suffix, scene, bar, tp, wrong, missed, error])
                print("done", name + suffix, flush=True)


if __name__ == "__main__":
    main()
