"""Run the scene's detector once and save every frame's boxes: data/mot/dets/<seq>.npz."""

import sys
import time
from pathlib import Path

import numpy as np

from pathsense.config import load_config, scene_source
from pathsense.detect import build_detector
from pathsense.detect.base import drop_partial_duplicates
from pathsense.io import VideoSource

for scene in map(Path, sys.argv[1:]):
    config = load_config(scene=scene)
    video = Path(scene_source(scene))
    detector = build_detector(config.detector)
    boxes, confs, frames, times, ms = [], [], [], [], []
    with VideoSource(str(video)) as source:
        fps = source.fps
        for frame in source:
            t0 = time.perf_counter()
            d = detector(frame.image)
            ms.append((time.perf_counter() - t0) * 1000)
            if config.detector.drop_partial:
                d = drop_partial_duplicates(d)
            boxes.append(d.xyxy.astype(np.float32))
            confs.append(d.confidence.astype(np.float32))
            frames.append(frame.index)
            times.append(frame.timestamp)
    out = Path("data/mot/dets") / f"{video.stem}.npz"
    out.parent.mkdir(exist_ok=True)
    n = np.array([len(b) for b in boxes])
    np.savez(
        out,
        xyxy=np.concatenate(boxes),
        conf=np.concatenate(confs),
        n=n,
        frame=np.array(frames),
        t=np.array(times),
        fps=fps,
        detect_ms=np.median(ms[5:]),
    )
    print(
        video.stem, len(n), "frames", int(n.sum()), "boxes", f"{np.median(ms[5:]):.1f} ms ->", out
    )
