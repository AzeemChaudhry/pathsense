import numpy as np
from conftest import detections

from pathsense.track.stitch import Stitcher


def boxes(*people):
    """(tracker_id, feet_x, feet_y, height) per person; none gives an empty frame."""
    tracked = detections([[x - 20, y - h, x + 20, y] for _, x, y, h in people])
    tracked.tracker_id = np.array([tid for tid, _, _, _ in people], dtype=int)
    return tracked


def ids(tracked):
    return tracked.tracker_id.tolist()


def test_a_person_found_again_where_they_were_heading_keeps_their_id():
    stitch = Stitcher(seconds=2.0)
    for frame in range(6):  # walking right at 100 px/s, 200 px tall
        stitch(boxes((1, 100 + 10 * frame, 500, 200)), t=frame / 10)
    stitch(boxes(), t=1.0)  # hidden for 1.5 s; the tracker gives up on ID 1
    back = stitch(boxes((7, 300, 500, 200)), t=2.0)  # where 100 px/s would take them
    assert ids(back) == [1] and stitch.stitched == 1
    assert ids(stitch(boxes((7, 310, 500, 200)), t=2.1)) == [1]  # and stays 1


def test_someone_new_elsewhere_or_too_late_keeps_their_new_id():
    stitch = Stitcher(seconds=2.0, distance=1.0)
    stitch(boxes((1, 100, 500, 200)), t=0.0)
    assert ids(stitch(boxes((2, 900, 500, 200)), t=0.5)) == [2]  # 4 heights away
    stitch2 = Stitcher(seconds=2.0)
    stitch2(boxes((1, 100, 500, 200)), t=0.0)
    assert ids(stitch2(boxes((3, 100, 500, 200)), t=2.5)) == [3]  # gone too long


def test_a_very_different_size_is_someone_else():
    stitch = Stitcher(seconds=2.0, size=0.5)
    stitch(boxes((1, 100, 500, 200)), t=0.0)
    assert ids(stitch(boxes((2, 100, 500, 450)), t=0.5)) == [2]  # 2.25 times taller


def test_someone_still_in_view_is_never_given_away():
    stitch = Stitcher(seconds=2.0)
    stitch(boxes((1, 100, 500, 200)), t=0.0)
    both = stitch(boxes((1, 100, 500, 200), (2, 120, 500, 200)), t=0.1)
    assert ids(both) == [1, 2]


def test_each_vanished_person_is_matched_once_and_to_the_closest():
    stitch = Stitcher(seconds=2.0)
    stitch(boxes((1, 100, 500, 200), (2, 400, 500, 200)), t=0.0)
    stitch(boxes(), t=0.5)
    back = stitch(boxes((8, 390, 500, 200), (9, 110, 500, 200)), t=1.0)
    assert ids(back) == [2, 1]


def test_an_old_id_coming_back_next_to_its_stitched_one_is_kept_apart():
    stitch = Stitcher(seconds=2.0)
    stitch(boxes((1, 100, 500, 200)), t=0.0)
    stitch(boxes((5, 100, 500, 200)), t=0.5)  # stitched to 1
    both = stitch(boxes((1, 100, 500, 200), (5, 100, 500, 200)), t=0.6)
    assert ids(both) == [1, 5]  # ID 1 is back, so 5 is its own person again


def test_an_old_id_coming_back_first_in_the_frame_keeps_its_person():
    # Review of PR 10: with the stitched ID listed first, both came out as person 1.
    stitch = Stitcher(seconds=2.0)
    stitch(boxes((1, 100, 500, 200)), t=0.0)
    stitch(boxes((5, 100, 500, 200)), t=0.5)  # stitched to 1
    both = stitch(boxes((5, 100, 500, 200), (1, 100, 500, 200)), t=0.6)
    assert ids(both) == [5, 1]
