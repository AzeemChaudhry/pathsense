"""The pipeline end to end with a fake detector and the real tracker and video reader."""

import numpy as np
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
