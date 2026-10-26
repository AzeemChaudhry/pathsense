"""The tracker wrapper on synthetic detections, with the real trackers library."""

import numpy as np
import pytest
from conftest import detections

from pathsense.config import ConfigError
from pathsense.track import TRACKERS, Tracker, build_tracker


def walk(tracker, frames=15, people=((50, 50), (300, 60))):
    """People walking right 5 px per frame; returns the IDs seen in each frame."""
    ids = []
    for i in range(frames):
        boxes = [[x + 5 * i, y, x + 5 * i + 40, y + 100] for x, y in people]
        image = np.zeros((240, 480, 3), np.uint8)
        tracked = tracker.update(detections(boxes), frame=image, timestamp=i / 30)
        ids.append(sorted(tracked.tracker_id.tolist()))
    return ids


@pytest.mark.parametrize("name", sorted(TRACKERS))
def test_each_person_keeps_one_id(name):
    ids = walk(Tracker(name))
    settled = [frame_ids for frame_ids in ids if len(frame_ids) == 2]
    assert len(settled) >= 10, ids  # both people confirmed after a few frames
    assert all(frame_ids == settled[0] for frame_ids in settled), ids


@pytest.mark.parametrize("name", sorted(TRACKERS))
def test_unconfirmed_tracks_are_never_returned(name):
    for frame_ids in walk(Tracker(name)):
        assert -1 not in frame_ids


def test_bytetrack_needs_two_frames_before_giving_an_id():
    ids = walk(Tracker("bytetrack"), frames=2)
    assert ids[0] == [] and len(ids[1]) == 2


def test_reset_forgets_tracks():
    tracker = Tracker("bytetrack")
    first = walk(tracker)[-1]
    tracker.reset()
    assert walk(tracker)[-1] == first  # IDs restart from the beginning


def test_frame_only_reaches_botsort(monkeypatch):
    seen = {}
    for name in TRACKERS:
        tracker = Tracker(name)
        original = tracker._tracker.update

        def spy(dets, frame=None, timestamp=None, _name=name, _orig=original):
            seen[_name] = frame is not None
            return _orig(dets, frame=frame, timestamp=timestamp)

        monkeypatch.setattr(tracker._tracker, "update", spy)
        walk(tracker, frames=1)
    assert seen == {"botsort": True, "bytetrack": False, "ocsort": False}


def test_params_reach_the_tracker_class():
    # Boxes have confidence 0.9; above that activation threshold no track is ever started.
    tracker = build_tracker("bytetrack", frame_rate=30, params={"track_activation_threshold": 0.95})
    assert walk(tracker, frames=5) == [[]] * 5


def test_unknown_tracker_lists_the_choices():
    with pytest.raises(ConfigError, match="bytetrack"):
        Tracker("deepsort")
