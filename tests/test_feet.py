import numpy as np
import pytest
from conftest import detections

from pathsense.analytics import Feet


def person(track_id, top, bottom, x=100.0, width=40.0):
    tracked = detections([[x - width / 2, top, x + width / 2, bottom]])
    tracked.tracker_id = np.array([track_id])
    return tracked


def test_feet_are_the_bottom_centre_while_the_whole_body_is_seen():
    feet = Feet(hidden_legs=0.25)
    points, heights = feet(person(1, 100, 300), t=0.0)
    assert points.tolist() == [[100.0, 300.0]] and heights.tolist() == [200.0]


def test_hidden_legs_are_estimated_from_the_head():
    # The grocery-store shopper: full height ~700 px, then a cart hides the legs.
    feet = Feet(hidden_legs=0.25)
    feet(person(1, 820, 1530), t=13.0)  # whole: 710 px
    points, heights = feet(person(1, 811, 1240), t=13.5)  # 429 px: legs hidden
    assert points[0, 1] == pytest.approx(811 + 710) and heights[0] == 710


def test_a_long_occlusion_keeps_the_last_full_height():
    feet = Feet(hidden_legs=0.25, window=2.0)
    feet(person(1, 100, 800), t=0.0)
    for t in np.arange(0.1, 13.0, 0.1):  # 13 s behind a cart
        points, _ = feet(person(1, 100, 450), t=float(t))
    assert points[0, 1] == pytest.approx(800)


def test_a_slowly_changing_height_is_trusted():
    # Walking towards the camera: the box grows, never more than 25 % below its recent best.
    feet = Feet(hidden_legs=0.25)
    for i, height in enumerate(range(400, 700, 10)):
        points, _ = feet(person(1, 100, 100 + height), t=i / 10)
    assert points[0, 1] == 100 + 690


def test_walking_away_shrinks_the_box_without_hiding_the_legs():
    # Review of PR 10: a box 40 % shorter for good stayed at the old height. Walking away
    # shrinks the width too, so the feet stay at the bottom of the box.
    feet = Feet(hidden_legs=0.25)
    feet(person(1, 100, 800, width=200), t=0.0)
    points, heights = feet(person(1, 300, 720, width=120), t=5.0)  # 0.6 times the size
    assert points[0, 1] == 720 and heights[0] == 420


def test_zero_turns_the_estimate_off_and_forget_clears_memory():
    feet = Feet(hidden_legs=0)
    feet(person(1, 100, 800), t=0.0)
    assert feet(person(1, 100, 300), t=0.1)[0][0, 1] == 300
    on = Feet(hidden_legs=0.25)
    on(person(2, 100, 800), t=0.0)
    on.forget(2)
    assert on(person(2, 100, 300), t=5.0)[0][0, 1] == 300  # a new person with that id
    with pytest.raises(ValueError, match="hidden_legs"):
        Feet(hidden_legs=1.0)
