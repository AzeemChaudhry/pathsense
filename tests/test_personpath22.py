"""Labels at 5 FPS to every frame, and a line from a video's own tracks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from personpath22 import MAX_GAP, choose_line, interpolate  # noqa: E402


def label(person, frame, x, y=100):
    return {"id": person, "bb": [x, y, 40, 100], "blob": {"frame_idx": frame}}


def test_positions_between_two_labels_are_filled_in():
    frames = interpolate([label(0, 10, 100), label(0, 0, 0)])[0]
    assert [f for f, _ in frames] == list(range(11))
    assert frames[5][1][0] == 50


def test_a_gap_at_the_limit_is_filled_and_a_longer_one_is_not():
    near = interpolate([label(0, 0, 0), label(0, MAX_GAP, 10)])[0]
    far = interpolate([label(0, 0, 0), label(0, MAX_GAP + 1, 10)])[0]
    assert len(near) == MAX_GAP + 1 and len(far) == 2


def test_a_person_labelled_once_stays_one_label():
    assert len(interpolate([label(3, 7, 50)])[3]) == 1


def walkers(count, down=True):
    tracks = {}
    for person in range(count):
        rows = range(0, 600, 20) if down else range(580, -1, -20)
        tracks[person] = [
            (k, np.array([100.0 + 80 * person, float(y), 40.0, 100.0])) for k, y in enumerate(rows)
        ]
    return tracks


def test_the_line_is_square_to_the_flow_where_people_cross():
    line, crossings = choose_line(walkers(6), 1920, 1080)
    assert crossings == 6
    (x1, y1), (x2, y2) = line
    assert abs(y1 - y2) < 1 and abs(x1 - x2) > 300  # across, as people walk down


def test_opposite_directions_are_one_flow():
    tracks = {**walkers(3), **{k + 3: v for k, v in walkers(3, down=False).items()}}
    line, crossings = choose_line(tracks, 1920, 1080)
    assert line is not None and crossings == 6


def test_no_line_where_nobody_moves():
    still = {
        p: [(k, np.array([100.0 * p, 100.0, 40.0, 100.0])) for k in range(30)] for p in range(9)
    }
    assert choose_line(still, 1920, 1080) == (None, 0)
