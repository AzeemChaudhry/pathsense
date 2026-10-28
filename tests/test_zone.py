import numpy as np
import pytest
from conftest import detections

from pathsense.analytics import Event, ZoneCounter, inside_polygon

# The bottom part of a 1920x1080 frame, below y = 540.
FLOOR = [(0.0, 540.0), (1920.0, 540.0), (1920.0, 1080.0), (0.0, 1080.0)]


def people(*feet):
    """Tracked people as (track_id, feet_x, feet_y); none gives an empty frame."""
    tracked = detections([[x - 20, y - 100, x + 20, y] for _, x, y in feet])
    tracked.tracker_id = np.array([tid for tid, _, _ in feet], dtype=int)
    return tracked


def walk(zone, track_id, ys, x=960.0, start=0):
    """One person through frames start, start + 1... at 10 FPS; return all events."""
    events = []
    for frame, y in enumerate(ys, start):
        events += zone.update(people((track_id, x, y)), t=frame / 10, frame=frame)
    return events


def test_inside_polygon_handles_concave_shapes():
    assert inside_polygon((960, 800), FLOOR) and not inside_polygon((960, 300), FLOOR)
    l_shape = [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)]
    assert inside_polygon((20, 80), l_shape)
    assert not inside_polygon((80, 80), l_shape)  # the cut out corner


def test_walking_in_and_out_gives_enter_and_exit_with_the_zone_name():
    zone = ZoneCounter("floor", FLOOR)
    events = walk(zone, 1, [400, 600, 700, 400])
    assert events == [
        Event(0.1, 1, 1, "zone_enter", "floor"),
        Event(0.3, 3, 1, "zone_exit", "floor"),
        Event(0.3, 3, 1, "dwell", "floor", duration=0.2),
    ]
    assert (zone.entered, zone.exited, zone.occupancy) == (1, 1, 0)


def test_first_seen_inside_is_an_enter():
    # Walked in from the side of the picture, never crossing the zone's top edge.
    zone = ZoneCounter("floor", FLOOR)
    assert walk(zone, 2, [800, 800]) == [Event(0.0, 0, 2, "zone_enter", "floor")]
    assert zone.occupancy == 1


def test_hovering_on_the_edge_is_not_counted():
    zone = ZoneCounter("floor", FLOOR, confirm=0.5)
    walk(zone, 3, [530, 550, 530, 550, 530, 550, 530])  # in and out every 0.1 s
    assert zone.finish() == []  # ended outside, where they started
    assert (zone.entered, zone.exited) == (0, 0)


def test_a_held_step_inside_counts_with_the_time_of_the_step():
    zone = ZoneCounter("floor", FLOOR, confirm=0.5)
    events = walk(zone, 4, [400, 600] + [650] * 6)
    assert events == [Event(0.1, 1, 4, "zone_enter", "floor")]


def test_someone_who_vanishes_inside_leaves_when_last_seen():
    zone = ZoneCounter("floor", FLOOR, lost=1.0)
    walk(zone, 5, [600, 650])  # inside, last seen at 0.1 s
    assert zone.update(people(), t=0.8, frame=8) == []  # maybe just hidden
    assert zone.update(people(), t=1.1, frame=11) == [
        Event(0.1, 1, 5, "zone_exit", "floor"),
        Event(0.1, 1, 5, "dwell", "floor", duration=0.1),
    ]
    assert zone.occupancy == 0


def test_stepping_out_then_vanishing_is_dated_at_the_step():
    zone = ZoneCounter("floor", FLOOR, confirm=0.5, lost=1.0)
    walk(zone, 6, [600] * 6 + [400])  # inside from 0.0 s (confirmed at 0.5), out at 0.6
    assert zone.update(people(), t=1.6, frame=16)[0] == Event(0.6, 6, 6, "zone_exit", "floor")


def test_a_glimpse_inside_that_vanishes_is_dropped():
    zone = ZoneCounter("floor", FLOOR, confirm=0.5, lost=1.0)
    walk(zone, 7, [700])  # one frame inside
    assert zone.update(people(), t=1.0, frame=10) == []
    assert (zone.entered, zone.exited) == (0, 0)


def test_finish_confirms_waiting_changes_but_does_not_exit_people_inside():
    zone = ZoneCounter("floor", FLOOR, confirm=0.5)
    walk(zone, 8, [700] * 8)  # inside, confirmed
    walk(zone, 9, [700], start=8)  # just arrived, waiting
    assert zone.finish() == [
        Event(0.8, 8, 9, "zone_enter", "floor"),
        Event(0.7, 7, 8, "dwell", "floor", duration=0.7),  # time so far, no exit
        Event(0.8, 8, 9, "dwell", "floor", duration=0.0),
    ]
    assert (zone.entered, zone.exited, zone.occupancy) == (2, 0, 2)


def test_each_person_is_followed_separately():
    zone = ZoneCounter("floor", FLOOR)
    zone.update(people((1, 960, 400), (2, 960, 800)), t=0.0, frame=0)
    events = zone.update(people((1, 960, 800), (2, 960, 400)), t=0.1, frame=1)
    assert sorted((e.track_id, e.type) for e in events) == [
        (1, "zone_enter"), (2, "dwell"), (2, "zone_exit")
    ]  # fmt: skip


def test_a_zone_needs_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        ZoneCounter("bad", [(0, 0), (10, 10)])


def test_people_inside_at_the_start_can_be_left_uncounted():
    for count_present, entered in ((True, 1), (False, 0)):
        zone = ZoneCounter("floor", FLOOR, count_present=count_present, start=1.0)
        walk(zone, 1, [800, 800, 400])  # inside from the first frame, then walks out
        walk(zone, 2, [400, 800], start=20)  # walks in at 2.1 s: always counted
        assert (zone.entered - 1, zone.exited) == (entered, 1)
        assert zone.occupancy == 1


def test_dwell_is_the_time_between_enter_and_exit():
    zone = ZoneCounter("floor", FLOOR, confirm=0.5)
    events = walk(zone, 3, [400] + [700] * 30 + [400] * 6)  # in at 0.1 s, out at 3.1 s
    dwell = [e for e in events if e.type == "dwell"]
    assert dwell == [Event(3.1, 31, 3, "dwell", "floor", duration=3.0)]


def test_hidden_legs_keep_a_shopper_inside():
    # Head at y 300, full body 400 px (feet at 700, inside). From frame 10 a cart hides the
    # legs: the box ends at 500, outside the floor zone.
    for hidden_legs, exits in ((0.0, 1), (0.25, 0)):
        zone = ZoneCounter("floor", FLOOR, confirm=0.2, hidden_legs=hidden_legs)
        for frame in range(20):
            tracked = detections([[940, 300, 980, 700 if frame < 10 else 500]])
            tracked.tracker_id = np.array([1])
            zone.update(tracked, t=frame / 10, frame=frame)
        assert zone.exited == exits


def test_finish_twice_gives_nothing_more():
    zone = ZoneCounter("z", FLOOR)
    zone.update(people((1, 960, 800)), t=0.0, frame=0)
    assert len(zone.finish()) == 1  # a dwell for the person still inside
    assert zone.finish() == []
