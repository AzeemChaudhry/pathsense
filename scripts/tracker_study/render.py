"""Draw one run on its video: boxes by ID, the line and zones, running counts against the
truth, and every crossing marked right (green), wrong (red) or missed (yellow ring)."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from replay import labels, true_events

from pathsense.config import load_config, scene_source

WIDTH = 960  # videos are drawn at this width to keep them small
FLASH = 1.0  # seconds a crossing mark stays on screen
GREEN, RED, YELLOW, WHITE = (80, 200, 80), (60, 60, 230), (0, 210, 255), (255, 255, 255)


def colour(track_id: int) -> tuple[int, int, int]:
    """A steady, distinct colour per ID (golden ratio hue steps)."""
    hue = int((track_id * 0.618034 % 1) * 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def read_tracks(path: Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    frames = defaultdict(list)
    for row in np.loadtxt(path, delimiter=",", ndmin=2):
        x, y, w, h = row[2:6]
        frames[int(row[0]) - 1].append((int(row[1]), np.array([x, y, x + w, y + h])))
    return frames


def render_run(folder: Path) -> Path:
    metrics = json.loads((folder / "metrics.json").read_text())
    setup, scene = metrics["setup"]["name"], metrics["scene"]
    out = folder / "video.mp4"
    if out.exists() and out.stat().st_mtime > (folder / "metrics.json").stat().st_mtime:
        return out  # already drawn for these results
    source = scene_source(Path("configs/scenes/mot") / f"{scene}.yaml")
    analytics = load_config(scene=Path("configs/scenes/mot") / f"{scene}.yaml").analytics
    tracks = read_tracks(folder / "tracks.txt")
    marks = json.loads((folder / "crossings.json").read_text())
    ours = [m for m in marks if m["mark"] != "missed"]
    truth = [e for e in true_events(scene) if e.type in ("enter", "exit")]

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = WIDTH / width
    size = (WIDTH, int(round(height * scale / 2)) * 2)
    flash = int(FLASH * fps)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    command += ["-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264"]
    command += ["-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", str(out)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    line = np.int32(np.array(analytics.line) * scale) if analytics.line else None
    zones = [np.int32(np.array(z.polygon) * scale) for z in analytics.zones]

    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        image = cv2.resize(image, (WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
        image = cv2.copyMakeBorder(image, 0, size[1] - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for zone in zones:
            cv2.polylines(image, [zone], True, (255, 170, 60), 1, cv2.LINE_AA)
        if line is not None:
            cv2.polylines(image, [line], False, WHITE, 2, cv2.LINE_AA)
        boxes = {}
        for track_id, box in tracks.get(index, []):
            x1, y1, x2, y2 = (box * scale).astype(int)
            boxes[track_id] = (x1, y1, x2, y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour(track_id), 1)
            label = str(track_id if track_id < 1_000_000 else f"g{track_id - 1_000_000}")
            cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 0.7, WHITE, 1)
        for m in ours:
            if m["frame"] <= index < m["frame"] + flash and m["id"] in boxes:
                x1, y1, x2, y2 = boxes[m["id"]]
                c = GREEN if m["mark"] == "right" else RED
                cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), c, 2)
                arrow = "IN" if m["type"] == "enter" else "OUT"
                cv2.putText(image, arrow, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        people = labels(scene).get(index)
        for m in marks:
            if m["mark"] == "missed" and m["frame"] <= index < m["frame"] + flash and people:
                hit = np.flatnonzero(people.tracker_id == m["id"])
                if len(hit):
                    x1, y1, x2, y2 = people.xyxy[hit[0]] * scale
                    centre = (int((x1 + x2) / 2), int(y2))
                    cv2.circle(image, centre, max(6, int((y2 - y1) / 3)), YELLOW, 2)
        done = [m for m in ours if m["frame"] <= index]
        real = [e for e in truth if e.frame <= index]
        panel = [
            f"{setup}   {scene}   frame {index}",
            "IN  ours {:3d}  true {:3d}".format(
                sum(m["type"] == "enter" for m in done), sum(e.type == "enter" for e in real)
            ),
            "OUT ours {:3d}  true {:3d}".format(
                sum(m["type"] == "exit" for m in done), sum(e.type == "exit" for e in real)
            ),
            "right {}  wrong {}  missed {}".format(
                sum(m["mark"] == "right" for m in done),
                sum(m["mark"] == "wrong" for m in done),
                sum(m["mark"] == "missed" and m["frame"] <= index for m in marks),
            ),
        ]
        cv2.rectangle(image, (0, 0), (330, 16 * len(panel) + 6), (0, 0, 0), -1)
        for i, text in enumerate(panel):
            cv2.putText(image, text, (6, 16 * i + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
        encoder.stdin.write(image.tobytes())
        index += 1
    capture.release()
    encoder.stdin.close()
    encoder.wait()
    return out
