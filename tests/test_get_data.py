import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "get_data", Path(__file__).resolve().parents[1] / "train" / "get_data.py"
)
get_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_data)

RECORD = {
    "ID": "x",
    "gtboxes": [
        {"tag": "person", "fbox": [100, 50, 200, 400], "vbox": [100, 50, 200, 300], "extra": {}},
        {"tag": "person", "fbox": [-20, 300, 100, 400], "vbox": [0, 300, 80, 200], "extra": {}},
        {"tag": "mask", "fbox": [10, 10, 50, 50], "vbox": [10, 10, 50, 50], "extra": {"ignore": 1}},
        {
            "tag": "person",
            "fbox": [500, 10, 40, 80],
            "vbox": [500, 10, 40, 80],
            "extra": {"ignore": 1},
        },
        {"tag": "person", "fbox": [600, 100, 4, 4], "vbox": [600, 100, 4, 4], "extra": {}},
    ],
}


def test_drops_masks_ignored_and_tiny_boxes():
    lines = get_data.odgt_to_yolo(RECORD, 800, 600)
    assert len(lines) == 2


def test_box_is_normalised_and_clipped():
    first, second = (list(map(float, ln.split())) for ln in get_data.odgt_to_yolo(RECORD, 800, 600))
    assert first == pytest.approx([0, 0.25, 250 / 600, 0.25, 400 / 600], abs=1e-6)
    # second box starts at x=-20 and ends past the bottom edge; both get clipped
    cls, cx, cy, w, h = second
    assert cx - w / 2 >= 0 and cy + h / 2 <= 1.0
    assert abs(w - 80 / 800) < 1e-6 and abs(h - 300 / 600) < 1e-6


def test_openimages_maps_people_and_drops_groups_and_depictions():
    dets = [
        {"label": "Man", "bbox": [0.1, 0.2, 0.2, 0.4], "group": False, "depiction": False},
        {"label": "Woman", "bbox": [0.9, 0.5, 0.3, 0.3], "group": False, "depiction": False},
        {"label": "Person", "bbox": [0.0, 0.0, 1.0, 1.0], "group": True, "depiction": False},
        {"label": "Girl", "bbox": [0.3, 0.3, 0.1, 0.1], "group": False, "depiction": True},
        {"label": "Car", "bbox": [0.3, 0.3, 0.1, 0.1], "group": False, "depiction": False},
        {"label": "Boy", "bbox": [0.5, 0.5, 0.001, 0.2], "group": False, "depiction": False},
    ]
    lines = get_data.openimages_to_yolo(dets)
    assert len(lines) == 2
    man = list(map(float, lines[0].split()))
    assert man == pytest.approx([0, 0.2, 0.4, 0.2, 0.4], abs=1e-6)
    cls, cx, cy, w, h = map(float, lines[1].split())  # Woman box clipped at the right edge
    assert cx + w / 2 <= 1.0 + 1e-9 and w == pytest.approx(0.1, abs=1e-6)
