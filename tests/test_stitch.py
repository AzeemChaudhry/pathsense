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


def _upper_body_after_a_gap(top):
    """A 400 px person walks at 100 px/s; 0.5 s later their box is 200 px, same head."""
    stitch = Stitcher(seconds=2.0, top=top)
    for frame in range(5):
        stitch(boxes((1, 100 + 10 * frame, 800, 400)), t=frame / 10)
    stitch(boxes(), t=0.5)
    return stitch, stitch(boxes((2, 190, 600, 200)), t=0.9)


def test_off_a_box_cut_in_half_is_a_new_person():
    assert ids(_upper_body_after_a_gap(top=0.0)[1]) == [2]


def test_the_top_of_the_box_links_a_person_whose_legs_got_hidden():
    stitch, back = _upper_body_after_a_gap(top=0.25)
    assert ids(back) == [1] and stitch.stitched == 1


def test_the_top_of_the_box_does_not_link_someone_elsewhere():
    stitch = Stitcher(seconds=2.0, top=0.25)
    stitch(boxes((1, 100, 800, 400)), t=0.0)
    assert ids(stitch(boxes((2, 400, 600, 200)), t=0.5)) == [2]  # the head is 300 px off


def test_the_top_of_the_box_is_only_for_a_box_that_changed_height():
    stitch = Stitcher(seconds=2.0, top=0.25)
    stitch(boxes((1, 100, 800, 400)), t=0.0)
    assert ids(stitch(boxes((2, 100, 1400, 400)), t=0.5)) == [2]  # same height, feet far off


def test_the_top_of_the_box_refuses_a_box_of_another_width():
    stitch = Stitcher(seconds=2.0, top=0.25)
    stitch(boxes((1, 100, 800, 400)), t=0.0)
    wide = detections([[0, 400, 200, 600]])
    wide.tracker_id = np.array([2])
    assert ids(stitch(wide, t=0.5)) == [2]  # 200 px wide against 40


def test_a_top_of_0_gives_what_the_stitcher_gave_before_it_existed():
    stitch = Stitcher(seconds=2.0, top=0.0)
    seen = [ids(stitch(boxes((1, 100 + 10 * f, 800, 400)), t=f / 10)) for f in range(5)]
    stitch(boxes(), t=0.5)
    seen.append(ids(stitch(boxes((2, 190, 600, 200)), t=0.9)))
    seen.append(ids(stitch(boxes((3, 150, 800, 400)), t=1.0)))
    assert seen == [[1], [1], [1], [1], [1], [2], [1]]  # from the code before `top`


def test_the_top_of_the_box_links_a_box_growing_back_to_a_whole_body():
    stitch = Stitcher(seconds=2.0, top=0.25)
    stitch(boxes((1, 100, 600, 200)), t=0.0)
    assert ids(stitch(boxes((2, 100, 800, 400)), t=0.5)) == [1]


def test_the_top_of_the_box_ignores_a_box_change_inside_the_speed_window():
    stitch = Stitcher(seconds=2.0, top=0.1)
    stitch(boxes((1, 100, 800, 400)), t=0.0)  # head at y 400, standing still
    stitch(boxes((1, 100, 600, 200)), t=0.2)  # the feet jump 200 px, the head does not
    stitch(boxes(), t=0.3)
    assert ids(stitch(boxes((2, 100, 800, 400)), t=0.5)) == [1]
