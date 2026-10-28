"""The pipeline end to end with a fake detector and the real tracker and video reader."""

import numpy as np
import pytest
from conftest import detections

from pathsense.annotate import Annotator
from pathsense.config import load_config
from pathsense.detect import DETECTORS
from pathsense.io import VideoSource
from pathsense.pipeline import Pipeline
from pathsense.track import Tracker


class WalkingPerson:
    """Fake detector: one person moving right 4 px per frame."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        x = 4 * self.calls
        self.calls += 1
        return detections([[x, 5, x + 10, 40]])


def test_pipeline_tracks_one_person_through_a_video(tiny_video):
    pipeline = Pipeline(WalkingPerson(), Tracker("bytetrack", frame_rate=10))
    with VideoSource(str(tiny_video)) as source:
        results = list(pipeline.run(source))
    assert len(results) == 12
    ids = {int(t) for r in results for t in r.tracked.tracker_id}
    assert len(ids) == 1
    assert all(r.detect_ms >= 0 and r.track_ms >= 0 for r in results)
    assert results[3].frame.timestamp == 0.3


def test_from_config_builds_the_named_stages(monkeypatch):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    config = load_config(overrides=["detector.name=fake", "tracker.name=ocsort"])
    with pytest.warns(UserWarning, match="ocsort does not use"):  # default config's setting
        pipeline = Pipeline.from_config(config, frame_rate=25)
    assert pipeline.detector.name == "fake" and pipeline.tracker.name == "ocsort"


def test_annotator_draws_without_changing_the_input(tiny_video):
    with VideoSource(str(tiny_video)) as source:
        frame = next(iter(source))
    tracked = detections([[5, 5, 30, 40]])
    tracked.tracker_id = np.array([7])
    before = frame.image.copy()
    out = Annotator().draw(frame.image, tracked, stats="FPS 30")
    assert (frame.image == before).all() and (out != before).any()


def test_the_line_counter_runs_after_the_tracker(tiny_video):
    from pathsense.analytics import LineCounter

    # The fake walks right; feet x = 5, 9, 13... A vertical line at x = 25, drawn bottom to
    # top, so walking left to right is "enter".
    counter = LineCounter((25.0, 48.0), (25.0, 0.0))
    pipeline = Pipeline(WalkingPerson(), Tracker("bytetrack", frame_rate=10), counter)
    with VideoSource(str(tiny_video)) as source:
        results = list(pipeline.run(source))
    events = [e for r in results for e in r.events]
    assert [(e.frame, e.type) for e in events] == [(7, "enter")]  # feet 33: past the margin
    assert (counter.entered, counter.exited) == (1, 0)


def test_from_config_adds_the_line_counter_only_when_a_line_is_set(monkeypatch):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    plain = Pipeline.from_config(load_config(overrides=["detector.name=fake"]))
    assert plain.line_counter is None
    config = load_config(
        overrides=["detector.name=fake", "analytics.line=[[0, 675], [1920, 675]]",
                   "analytics.line_margin=8"]
    )  # fmt: skip
    counter = Pipeline.from_config(config).line_counter
    assert counter.start == (0.0, 675.0) and counter.margin == 8


def test_the_line_forgets_people_just_after_the_tracker_does(monkeypatch):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    line = "analytics.line=[[0, 675], [1920, 675]]"
    default = Pipeline.from_config(load_config(overrides=["detector.name=fake", line]), 25)
    assert default.line_counter.lost == pytest.approx(30 / 30 + 1 / 25)
    longer = load_config(
        overrides=["detector.name=fake", line, "tracker.params={lost_track_buffer: 60}"]
    )
    assert Pipeline.from_config(longer, 25).line_counter.lost == pytest.approx(2 + 1 / 25)


def test_draw_line_marks_the_line_and_the_totals():
    from pathsense.annotate import FLASH_COLOUR, LINE_COLOUR, draw_line

    image = np.zeros((200, 400, 3), np.uint8)
    draw_line(image, ((0.0, 150.0), (400.0, 150.0)), entered=3, exited=1)
    assert tuple(image[150, 380]) == LINE_COLOUR
    assert image[10:70, 10:330].any()  # the IN / OUT box
    draw_line(image, ((0.0, 150.0), (400.0, 150.0)), 3, 1, flash=True)
    assert tuple(image[150, 380]) == FLASH_COLOUR
