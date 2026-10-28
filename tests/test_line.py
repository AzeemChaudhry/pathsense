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
    with pytest.raises(ValueError, match="two or more points"):
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


def empty():
    tracked = detections([])
    tracked.tracker_id = np.array([], dtype=int)
    return tracked


def test_a_diagonal_line_uses_the_same_sides_and_segment():
    # From the top left to the bottom right corner of a 100 x 100 square.
    counter = LineCounter((0.0, 0.0), (100.0, 100.0), margin=1)
    assert side_of_line((20, 80), (0, 0), (100, 100)) > 0  # below the diagonal: right of it
    feet = [(80, 20), (20, 80)]  # from above the diagonal to below it
    events = []
    for frame, (x, y) in enumerate(feet):
        tracked = detections([[x - 5, y - 10, x + 5, y]])
        tracked.tracker_id = np.array([1])
        events += counter.update(tracked, t=frame / 10, frame=frame)
    assert [e.type for e in events] == ["enter"]


def test_the_ends_of_the_segment_count_and_just_beyond_does_not():
    door = (800.0, 540.0), (1100.0, 540.0)
    assert [e.type for e in walk(LineCounter(*door), 1, 800, [500, 580])] == ["enter"]
    assert [e.type for e in walk(LineCounter(*door), 2, 1100, [500, 580])] == ["enter"]
    assert walk(LineCounter(*door), 3, 1101, [500, 580]) == []


def test_standing_on_the_line_after_a_flip_does_not_confirm_it():
    counter = LineCounter(START, END, margin=5, confirm=0.5)
    # Flip at 0.1 s, then stand on the line (within 5 px) for a second: not yet counted.
    assert walk(counter, 1, 960, [500, 560] + [542] * 10) == []
    # Then step clearly onto the new side: counted, dated at the flip.
    assert counter.update(person(1, 960, 580), t=1.2, frame=12) == [Event(0.1, 1, 1, "enter")]


def test_someone_hidden_for_less_than_lost_is_not_counted_on_the_old_side():
    counter = LineCounter(START, END, confirm=0.5, lost=1.0)
    walk(counter, 2, 960, [500, 580])  # flip at 0.1 s
    assert counter.update(empty(), t=0.8, frame=8) == []  # hidden 0.7 s: past confirm
    # Back, on the old side: the flip is cancelled, nothing was ever counted.
    assert counter.update(person(2, 960, 500), t=0.9, frame=9) == []
    assert counter.finish() == [] and (counter.entered, counter.exited) == (0, 0)


def test_someone_hidden_then_back_on_the_new_side_counts_once():
    counter = LineCounter(START, END, confirm=0.5, lost=1.0)
    walk(counter, 3, 960, [500, 580])
    counter.update(empty(), t=0.8, frame=8)
    assert counter.update(person(3, 960, 590), t=0.9, frame=9) == [Event(0.1, 1, 3, "enter")]


def test_a_flip_counts_on_the_same_frame_without_a_hold():
    counter = LineCounter(START, END, confirm=0.0)
    assert counter.update(person(4, 960, 500), t=0.0, frame=0) == []
    assert counter.update(person(4, 960, 580), t=0.1, frame=1) == [Event(0.1, 1, 4, "enter")]


def test_float_rounding_does_not_delay_confirmation():
    counter = LineCounter(START, END, confirm=0.5)
    counter.update(person(5, 960, 500), t=0.0, frame=0)
    counter.update(person(5, 960, 580), t=0.1, frame=1)
    assert counter.update(person(5, 960, 580), t=0.1 + 0.49999999999999994, frame=6) != []


def test_finish_twice_counts_once():
    counter = LineCounter(START, END, confirm=5.0)
    walk(counter, 6, 960, [500, 580])
    assert len(counter.finish()) == 1 and counter.finish() == []


def test_two_people_waiting_one_confirms_one_steps_back():
    counter = LineCounter(START, END, confirm=0.5)
    feet = [(500, 500), (580, 580)] + [(580, 500)] * 6  # both flip at 0.1 s; #2 steps back
    events = []
    for frame, ys in enumerate(feet):
        tracked = detections([[940, y - 100, 980, y] for y in ys])
        tracked.tracker_id = np.array([1, 2])
        events += counter.update(tracked, t=frame / 10, frame=frame)
    assert events == [Event(0.1, 1, 1, "enter")]


def test_a_frame_without_track_ids_still_lets_vanished_people_count():
    counter = LineCounter(START, END, confirm=0.5, lost=1.0)
    walk(counter, 7, 960, [500, 580])
    no_ids = detections([[0, 0, 10, 10]])  # tracker_id is None
    assert counter.update(no_ids, t=1.2, frame=12) == [Event(0.1, 1, 7, "enter")]


def test_forgotten_tracks_are_dropped_from_memory():
    counter = LineCounter(START, END, lost=1.0)
    walk(counter, 8, 960, [500, 542])  # one frame clearly above, one on the line
    walk(counter, 9, 960, [542])  # only ever on the line: no side
    counter.update(empty(), t=2.0, frame=20)
    assert counter._side == {} and counter._last_seen == {}


def test_a_track_dropped_right_after_a_flip_counts():
    # Chosen trade off (module docstring): one clear frame on the new side, then gone.
    counter = LineCounter(START, END, confirm=0.5, lost=1.0)
    walk(counter, 10, 960, [500, 580])
    assert counter.update(empty(), t=1.1, frame=11) == [Event(0.1, 1, 10, "enter")]


def test_an_id_back_after_lost_is_a_new_first_sighting():
    # The pipeline sets lost past the tracker's buffer, so this only happens with a hand
    # built counter; the returning id is not counted as a crossing.
    counter = LineCounter(START, END, lost=1.0)
    walk(counter, 11, 960, [500])
    counter.update(empty(), t=1.5, frame=15)  # forgotten
    assert counter.update(person(11, 960, 580), t=1.6, frame=16) == []


def test_finish_counts_a_waiting_flip_even_with_the_feet_on_the_line():
    counter = LineCounter(START, END, margin=5, confirm=5.0)
    walk(counter, 12, 960, [500, 580, 542])
    assert counter.finish() == [Event(0.1, 1, 12, "enter")]


##### Polylines, size relative margin, hidden legs #########################################


def test_a_polyline_follows_a_bend_and_counts_once():
    # A bent line: from (0, 540) to (960, 540), then down to (1920, 900).
    counter = LineCounter((0.0, 540.0), (960.0, 540.0), (1920.0, 900.0))
    assert [e.type for e in walk(counter, 1, 400, [500, 580])] == ["enter"]
    assert [e.type for e in walk(counter, 2, 1600, [700, 820])] == ["enter"]  # second leg
    assert walk(counter, 3, 1600, [600, 620]) == []  # above the second leg, never crosses


def test_beyond_the_ends_of_a_polyline_does_not_count():
    counter = LineCounter((100.0, 540.0), (500.0, 540.0), (900.0, 600.0))
    assert walk(counter, 1, 50, [500, 580]) == []  # before the first point
    assert walk(counter, 2, 1000, [550, 650]) == []  # past the last point


def test_polyline_side_matches_a_straight_line_for_two_points():
    from pathsense.analytics import polyline_side

    assert polyline_side((960, 600), [START, END]) == (pytest.approx(60), True)
    assert polyline_side((2000, 600), [START, END])[1] is False


def test_the_margin_can_grow_with_the_person():
    # People 680 px tall shuffling 40 px either side of the line: 15 % gives 102 px.
    def tall(track_id, ys):
        events = []
        for frame, y in enumerate(ys):
            tracked = detections([[940, y - 680, 980, y]])
            tracked.tracker_id = np.array([track_id])
            events += counter.update(tracked, t=frame / 10, frame=frame)
        return events

    counter = LineCounter(START, END, margin=5, margin_height=0.15)
    assert tall(1, [540 - 40, 540 + 40, 540 - 40, 540 + 40]) == []
    assert [e.type for e in tall(2, [400, 700])] == ["enter"]  # 140 px past it: counted


def test_hidden_legs_stop_a_cart_from_making_crossings():
    # Head stays at y 400; full body 300 px (feet 700). A cart hides the legs every other
    # frame, so the box bottom jumps to 580, across a line at y 640.
    shopper = [(400, 700), (400, 580)] * 5
    for hidden_legs, expected in ((0.0, 9), (0.25, 0)):
        counter = LineCounter((0.0, 640.0), (1920.0, 640.0), hidden_legs=hidden_legs)
        events = []
        for frame, (top, bottom) in enumerate(shopper):
            tracked = detections([[940, top, 980, bottom]])
            tracked.tracker_id = np.array([1])
            events += counter.update(tracked, t=frame / 10, frame=frame)
        assert len(events) == expected
