"""Turn PersonPath22 into scenes PathSense can score: more cameras than MOT alone.

    uv run python scripts/personpath22.py

Needs data/personpath22/annotation/anno_amodal_2022/*.json and splits.json, and the videos
in data/personpath22/raw/videos (docs/DATASETS.md). Writes, not in git:

    data/personpath22/gt/<video>.txt      labels in MOT format
    data/personpath22/scenes/<video>.yaml a scene file with its source, line and truth

Labels exist at about 5 FPS, so positions between two labels of a person are interpolated.

No one drew a line for these cameras, so each gets one from its own labelled tracks: square
to the direction people mostly move in, at the place the most of them cross, and as long
as the area they use. A video with fewer than MIN_CROSSINGS crossings gets no scene. This
replaces a line placed for a real use: it scores counting everywhere people flow, and says
nothing about any one doorway.

CC BY-NC 4.0, evaluation only: no frames in docs. Its authors ask that the data is not
used to identify people or make biometric information, so appearance stays off on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("data/personpath22")
MIN_CROSSINGS = 4
MIN_TRAVEL = 0.75  # a track must move this many of its heights to count towards the flow
MAX_GAP = 15  # frames: a longer gap between labels is left unlabelled, not guessed


def interpolate(entities: list[dict]) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Person id -> (frame, x y w h) for every frame between their labels."""
    by_person: dict[int, list[tuple[int, np.ndarray]]] = {}
    for item in entities:
        box = np.array(item["bb"], float)
        by_person.setdefault(item["id"], []).append((item["blob"]["frame_idx"], box))
    tracks = {}
    for person, labels in by_person.items():
        labels.sort(key=lambda label: label[0])
        filled = [labels[0]]
        for (f0, b0), (f1, b1) in zip(labels, labels[1:], strict=False):
            if f1 - f0 <= MAX_GAP:
                filled += [(f, b0 + (b1 - b0) * (f - f0) / (f1 - f0)) for f in range(f0 + 1, f1)]
            filled.append((f1, b1))
        tracks[person] = filled
    return tracks


def feet(track: list[tuple[int, np.ndarray]]) -> np.ndarray:
    return np.array([[b[0] + b[2] / 2, b[1] + b[3]] for _, b in track])


def choose_line(tracks: dict, width: float, height: float):
    """The line, and how many tracks cross it; (None, 0) if people do not flow anywhere."""
    moves = []
    for track in tracks.values():
        path = feet(track)
        step = path[-1] - path[0]
        size = np.median([b[3] for _, b in track])
        if np.hypot(*step) >= MIN_TRAVEL * size:
            moves.append(step * (1 if step @ (moves[0] if moves else step) >= 0 else -1))
    if len(moves) < MIN_CROSSINGS:
        return None, 0
    _, _, axes = np.linalg.svd(np.array(moves), full_matrices=False)
    flow = axes[0] / np.linalg.norm(axes[0])
    along = {p: feet(t) @ flow for p, t in tracks.items()}
    low = min(a.min() for a in along.values())
    high = max(a.max() for a in along.values())
    margin = (high - low) * 0.1
    best, crossings = None, 0
    for place in np.linspace(low + margin, high - margin, 41):
        crossed = sum(a.min() < place < a.max() for a in along.values())
        if crossed > crossings:
            best, crossings = place, crossed
    if crossings < MIN_CROSSINGS:
        return None, 0
    side = np.array([-flow[1], flow[0]])
    points = np.concatenate([feet(t) for t in tracks.values()])
    across = points @ side
    reach = np.percentile(across, [1, 99])
    pad = (reach[1] - reach[0]) * 0.1
    ends = [flow * best + side * (reach[0] - pad), flow * best + side * (reach[1] + pad)]
    ends = [[float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))] for x, y in ends]
    return ([[round(v, 1) for v in end] for end in ends] if ends[0] != ends[1] else None), crossings


def main() -> None:
    splits = json.loads((ROOT / "annotation/splits.json").read_text())
    (ROOT / "gt").mkdir(exist_ok=True)
    (ROOT / "scenes").mkdir(exist_ok=True)
    written = skipped = absent = 0
    for path in sorted((ROOT / "annotation/anno_amodal_2022").glob("*.json")):
        raw = json.loads(path.read_text())
        name = raw["metadata"]["data_path"]
        stem = name.removesuffix(".mp4")
        video = ROOT / "raw/videos" / name
        if not video.exists():
            absent += 1  # Amazon may not redistribute these (external datasets)
            continue
        size = raw["metadata"]["resolution"]
        tracks = interpolate(raw["entities"])
        line, crossings = choose_line(tracks, size["width"], size["height"])
        if line is None:
            skipped += 1
            continue
        with (ROOT / "gt" / f"{stem}.txt").open("w") as out:
            for person, track in tracks.items():
                for frame, (x, y, w, h) in track:
                    out.write(f"{frame + 1},{person + 1},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,1,1\n")
        split = "test" if name in splits["test"] else "train"
        scene = {
            "source": str(video),
            "analytics": {"line": line},
            "truth": {"labels": str(ROOT / "gt" / f"{stem}.txt")},
        }
        note = f"# PersonPath22 {stem}, {split}; line from its labels, about {crossings} cross.\n"
        (ROOT / "scenes" / f"{stem}.yaml").write_text(note + yaml.safe_dump(scene))
        written += 1
    print(f"{written} scenes, {skipped} with too little flow, {absent} videos not present")


if __name__ == "__main__":
    main()
