import os

import cv2
import numpy as np
import pytest
import supervision as sv

# Opt in markers: tests that need more than CI has, and the variable that runs them.
OPT_IN = {
    "slow": ("PATHSENSE_SLOW", "loads real model weights"),
    "camera": ("PATHSENSE_CAMERA", "needs a webcam and a person in front of it"),
}


def pytest_collection_modifyitems(config, items):
    for marker, (variable, reason) in OPT_IN.items():
        if os.environ.get(variable) == "1":
            continue
        skip = pytest.mark.skip(reason=f"{reason}; set {variable}=1 to run")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


def detections(boxes, confidence=0.9, class_id=0):
    """sv.Detections from a list of [x1, y1, x2, y2] boxes."""
    xyxy = np.array(boxes, dtype=float).reshape(-1, 4)
    n = len(xyxy)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full(n, confidence, dtype=float),
        class_id=np.full(n, class_id, dtype=int),
    )


@pytest.fixture
def tiny_video(tmp_path):
    """A 12 frame, 10 FPS, 64x48 mp4 whose pixel value encodes the frame index."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    return path
