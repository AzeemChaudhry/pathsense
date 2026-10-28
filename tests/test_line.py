import numpy as np
import pytest
from conftest import detections

from pathsense.analytics import Event, LineCounter, anchor_points, side_of_line

# A horizontal line across a 1920 wide frame at y = 540, drawn left to right.
START, END = (0.0, 540.0), (1920.0, 540.0)


def person(track_id, feet_x, feet_y):
    """One tracked person whose feet (bottom centre of the box) are at (feet_x, feet_y)."""
    tracked = detections([[feet_x - 20, feet_y - 100, feet_x + 20, feet_y]])
    tracked.tracker_id = np.array([track_id])
    return tracked


def walk(counter, track_id, feet_x, ys):
    """Feed one person through frames 0, 1, 2... at 10 FPS; return all events."""
    events = []
    for frame, y in enumerate(ys):
        events += counter.update(person(track_id, feet_x, y), t=frame / 10, frame=frame)
    return events


def test_anchor_is_the_bottom_centre_of_the_box():
    tracked = detections([[100, 200, 140, 300]])
    assert anchor_points(tracked).tolist() == [[120.0, 300.0]]


def test_side_is_a_signed_distance_in_pixels():
    assert side_of_line((960, 600), START, END) == pytest.approx(60)  # below the line
    assert side_of_line((960, 500), START, END) == pytest.approx(-40)  # above
    assert side_of_line((960, 540), START, END) == 0


def test_walking_down_across_the_line_is_one_enter():
    counter = LineCounter(START, END)
    events = walk(counter, 7, 960, [500, 520, 545, 560, 580])
    assert events == [Event(0.3, 3, 7, "enter")]  # seen clearly below at frame 3
    assert (counter.entered, counter.exited) == (1, 0)


def test_walking_up_is_an_exit_and_swapping_the_points_swaps_the_meaning():
    up = [580, 560, 520, 500]
    assert [e.type for e in walk(LineCounter(START, END), 1, 960, up)] == ["exit"]
    assert [e.type for e in walk(LineCounter(END, START), 1, 960, up)] == ["enter"]


def test_someone_standing_on_the_line_is_not_counted_again_and_again():
    counter = LineCounter(START, END, margin=5)
    # Feet jitter around y = 540 by up to 4 px: always within the margin.
    walk(counter, 3, 960, [500, 538, 543, 537, 544, 539, 541])
    assert (counter.entered, counter.exited) == (0, 0)


def test_crossing_and_coming_back_counts_both():
    counter = LineCounter(START, END)
    events = walk(counter, 4, 960, [500, 580, 500])
    assert [e.type for e in events] == ["enter", "exit"]


def test_first_sighting_is_not_a_crossing():
    # Someone who appears below the line was never seen crossing it.
    counter = LineCounter(START, END)
    assert walk(counter, 5, 960, [600, 620]) == []


def test_crossing_beyond_the_ends_of_the_line_does_not_count():
    door = LineCounter((800.0, 540.0), (1100.0, 540.0))  # only a 300 px doorway
    assert walk(door, 6, 1500, [500, 580]) == []  # walks past, outside the doorway
    assert [e.type for e in walk(door, 8, 900, [500, 580])] == ["enter"]


def test_each_person_has_their_own_side():
    counter = LineCounter(START, END)
    frames = [
        ([1, 2], [500, 600]),  # person 1 above, person 2 below
        ([1, 2], [580, 500]),  # both cross, in opposite directions
    ]
    events = []
    for index, (ids, ys) in enumerate(frames):
        tracked = detections([[940, y - 100, 980, y] for y in ys])
        tracked.tracker_id = np.array(ids)
        events += counter.update(tracked, t=index / 10, frame=index)
    assert sorted((e.track_id, e.type) for e in events) == [(1, "enter"), (2, "exit")]


def test_untracked_detections_and_bad_lines():
    assert LineCounter(START, END).update(detections([[0, 0, 10, 10]]), 0.0, 0) == []
    with pytest.raises(ValueError, match="two different points"):
        LineCounter((5, 5), (5, 5))


##### Confirmation ########################################################################


def test_a_waving_arm_that_flips_back_quickly_is_not_counted():
    counter = LineCounter(START, END, confirm=1.0)
    # The box's bottom centre jumps across and back every 0.3 s (frames at 10 FPS).
    ys = [500] * 3 + ([580] * 3 + [500] * 3) * 5
    assert walk(counter, 0, 960, ys) == [] and counter.finish() == []
    assert (counter.entered, counter.exited) == (0, 0)


def test_a_held_crossing_counts_once_with_the_time_it_happened():
    counter = LineCounter(START, END, confirm=1.0)
    events = walk(counter, 7, 960, [500, 500, 580] + [600] * 12)
    assert events == [Event(0.2, 2, 7, "enter")]  # reported at 1.2 s, dated 0.2 s
    assert counter.finish() == []


def test_someone_who_crosses_and_walks_out_of_view_still_counts():
    counter = LineCounter(START, END, confirm=1.0)
    walk(counter, 3, 960, [500, 580])  # crossed at 0.1 s, then gone
    empty = detections([])
    empty.tracker_id = np.array([], dtype=int)
    assert counter.update(empty, t=0.5, frame=5) == []  # gone 0.4 s: maybe just occluded
    assert counter.update(empty, t=1.2, frame=12) == [Event(0.1, 1, 3, "enter")]


def test_finish_counts_crossings_still_waiting_when_the_video_ends():
    counter = LineCounter(START, END, confirm=1.0)
    walk(counter, 4, 960, [500, 580, 590])  # crossed at 0.1 s, video ends at 0.2 s
    assert counter.finish() == [Event(0.1, 1, 4, "enter")]
    assert counter.entered == 1


def test_coming_back_before_confirmation_cancels_and_a_later_crossing_counts():
    counter = LineCounter(START, END, confirm=1.0)
    ys = [500, 580, 500] + [500] * 5 + [580] * 12  # step over and back, then walk across
    events = walk(counter, 5, 960, ys)
    assert events == [Event(0.8, 8, 5, "enter")]
