"""Telling people apart by looks: off changes nothing, each use does what it says."""

import numpy as np
import pytest
import torch
from conftest import detections
from trackers import ByteTrackTracker

from pathsense.analytics import LineCounter
from pathsense.appearance import AppearanceByteTrack, AppearanceLine, AppearanceStitcher, Embedder
from pathsense.appearance.embed import crops
from pathsense.appearance.osnet import OSNet
from pathsense.appearance.tracker import _boost, _overlapped
from pathsense.config import ConfigError, load_config
from pathsense.pipeline import Pipeline, build_counters, tracker_factory
from pathsense.track.stitch import Stitcher
from pathsense.tune.cache import SavedDetections
from pathsense.tune.embeddings import cached_embeddings, embeddings_path

RED, BLUE = np.eye(2, 4, 0, np.float32)[0], np.eye(2, 4, 1, np.float32)[1]


def boxes(*people, confidence=0.9):
    """(id or None, x, look) per person, each box 40 wide and 100 tall."""
    found = detections([[x, 50, x + 40, 150] for _, x, _ in people], confidence)
    found.data["embedding"] = np.array([look for _, _, look in people], np.float32)
    if people and people[0][0] is not None:
        found.tracker_id = np.array([i for i, _, _ in people], int)
    return found


def random_stream(seed=0, frames=60):
    rng = np.random.default_rng(seed)
    for frame in range(frames):
        n = rng.integers(0, 12)
        xy = rng.uniform(0, 300, (n, 2))
        size = rng.uniform(20, 80, (n, 2))
        found = detections(np.hstack([xy, xy + size]).tolist())
        found.confidence = rng.uniform(0.1, 1.0, n)
        yield found, frame / 30


def test_off_gives_exactly_what_the_library_gives():
    ours, theirs = AppearanceByteTrack(mode="off"), ByteTrackTracker()
    for found, t in random_stream():
        a, b = ours.update(found, timestamp=t), theirs.update(found, timestamp=t)
        assert a.tracker_id.tolist() == b.tracker_id.tolist()
        assert a.xyxy.tolist() == b.xyxy.tolist()


def track_two(**settings):
    tracker = AppearanceByteTrack(minimum_consecutive_frames=1, **settings)
    for _ in range(3):
        tracker.update(boxes((None, 100, RED), (None, 400, BLUE)))
    return tracker


def test_veto_refuses_a_box_that_overlaps_but_looks_different():
    tracker = track_two(mode="veto", gate=0.5)
    ids = [t.tracker_id for t in tracker.tracks]
    swapped = tracker.update(boxes((None, 100, BLUE), (None, 400, RED)))
    assert not set(swapped.tracker_id) & set(ids)


def test_without_veto_overlap_alone_decides():
    tracker = track_two(mode="off")
    ids = {t.tracker_id for t in tracker.tracks}
    assert set(tracker.update(boxes((None, 100, BLUE), (None, 400, RED))).tracker_id) == ids


def test_adaptive_picks_the_look_among_pairs_that_overlap():
    tracker = track_two(mode="adaptive")
    first, second = (t.tracker_id for t in tracker.tracks)
    # Two boxes between the tracks' reach would tie on overlap; here both overlap track 1.
    both = boxes((None, 105, BLUE), (None, 95, RED), (None, 400, BLUE))
    out = tracker.update(both)
    assert out.tracker_id[out.xyxy[:, 0].tolist().index(95)] == first
    assert second in out.tracker_id


def test_adaptive_never_matches_without_enough_overlap():
    tracker = track_two(mode="adaptive")
    ids = {t.tracker_id for t in tracker.tracks}
    far = tracker.update(boxes((None, 250, RED)))
    assert not set(far.tracker_id) & ids


def test_the_boost_is_the_lead_of_best_over_second_best_capped():
    alike = np.array([[0.9, 0.1], [0.8, 0.7]])
    # Rows lead by 0.5 (capped from 0.8) and 0.1; columns by 0.1 and 0.5 (capped from 0.6).
    assert _boost(alike, 0.5) == pytest.approx(np.array([[0.3, 0.5], [0.1, 0.3]]))
    assert _boost(np.array([[0.9]]), 0.5) == pytest.approx(np.array([[0.0]]))


def test_dynamic_memory_ignores_a_box_at_the_first_pass_threshold():
    tracker = AppearanceByteTrack(mode="off", memory=0.9, dynamic=True)
    floor = tracker.high_conf_det_threshold
    assert tracker._share(floor) == pytest.approx(1.0)
    assert tracker._share(1.0) == pytest.approx(0.9)
    assert tracker._share(None) == pytest.approx(0.9)
    assert AppearanceByteTrack(mode="off", memory=0.9)._share(floor) == pytest.approx(0.9)


def test_skip_overlap_finds_boxes_showing_two_people():
    xyxy = np.array([[0, 0, 40, 100], [5, 0, 45, 100], [300, 0, 340, 100]], float)
    assert _overlapped(xyxy, 0.3).tolist() == [True, True, False]
    assert _overlapped(xyxy, 0.0).tolist() == [False, False, False]


def test_an_overlapped_box_does_not_join_the_memory():
    tracker = AppearanceByteTrack(mode="off", minimum_consecutive_frames=1, skip_overlap=0.3)
    tracker.update(boxes((None, 100, RED)))
    tracker.update(boxes((None, 100, BLUE), (None, 105, BLUE)))
    assert tracker.tracks[0].embedding == pytest.approx(RED)


def walk(stitch, look, x_from=100):
    for frame in range(6):
        stitch(boxes((1, x_from + 10 * frame, look)), t=frame / 10)


def test_the_stitcher_refuses_someone_who_looks_different():
    plain, gated = Stitcher(seconds=2.0), AppearanceStitcher(0.5, seconds=2.0)
    walk(plain, RED)
    walk(gated, RED)
    plain(detections([]), t=1.0)
    gated(detections([]), t=1.0)
    assert plain(boxes((7, 300, BLUE)), t=2.0).tracker_id.tolist() == [1]
    assert gated(boxes((7, 300, BLUE)), t=2.0).tracker_id.tolist() == [7]


def test_the_stitcher_still_links_someone_who_looks_the_same():
    gated = AppearanceStitcher(0.5, seconds=2.0)
    walk(gated, RED)
    gated(detections([]), t=1.0)
    assert gated(boxes((7, 300, RED)), t=2.0).tracker_id.tolist() == [1]


def person(track_id, y, look):
    found = detections([[100, y - 100, 140, y]])
    found.tracker_id = np.array([track_id], int)
    found.data["embedding"] = np.array([look], np.float32)
    return found


def bounce(counter, looks):
    """Cross down, back up, then down again, with looks for each stretch."""
    events, t = [], 0.0
    for y, look in ((100, looks[0]), (300, looks[0]), (100, looks[1]), (300, looks[2])):
        for _ in range(3):
            events += counter.update(person(1, y, look), t, int(t * 10))
            t += 0.1
    return [e.type for e in events + counter.finish()]


LINE = ((0, 200), (500, 200))


def test_the_line_keeps_every_crossing_when_the_look_stays_the_same():
    plain = LineCounter(*LINE, confirm=0.1, bounce=12)
    looks = AppearanceLine(*LINE, same=0.8, confirm=0.1, bounce=12)
    assert bounce(plain, (RED, RED, RED)) == ["enter", "enter"]
    assert bounce(looks, (RED, RED, RED)) == ["enter", "exit", "enter"]
    assert looks.kept == 1


def test_the_line_drops_a_reversal_at_once_when_the_look_changes():
    looks = AppearanceLine(*LINE, different=0.5, memory=0.0, confirm=0.1, bounce=12)
    assert bounce(looks, (RED, BLUE, BLUE)) == ["enter", "enter"]
    assert looks.dropped == 1


def test_a_track_lost_mid_reversal_is_still_judged_by_its_look():
    looks = AppearanceLine(*LINE, different=0.5, memory=0.0, confirm=5.0, lost=0.5, bounce=12)
    looks._crossed[1], looks._side[1] = 1, 1  # crossed down earlier, at a known look
    looks._at_crossing[1] = RED
    looks.update(person(1, 100, BLUE), 0.0, 0)  # now up, unconfirmed, as someone else
    assert [e.type for e in looks.update(detections([]), 1.0, 10)] == []
    assert looks.dropped == 1


def test_off_builds_the_plain_classes_and_no_embedder():
    config = load_config(overrides=["analytics.line=[[0, 1], [9, 1]]"])
    assert tracker_factory(config.appearance) is None
    line, _ = build_counters(config.analytics, 1.0, config.appearance)
    assert type(line) is LineCounter
    pipeline = Pipeline.from_config(config, detector=lambda image: None)
    assert pipeline.embedder is None and type(pipeline.stitcher) is Stitcher


def test_on_builds_what_the_config_asks_for():
    config = load_config(
        overrides=[
            "analytics.line=[[0, 1], [9, 1]]",
            "appearance.enabled=true",
            "appearance.match=veto",
            "appearance.stitch_gate=0.7",
            "appearance.line_same=0.7",
        ]
    )
    pipeline = Pipeline.from_config(config, detector=lambda image: None)
    assert isinstance(pipeline.tracker._tracker, AppearanceByteTrack)
    assert isinstance(pipeline.stitcher, AppearanceStitcher)
    assert isinstance(pipeline.line_counter, AppearanceLine)
    assert pipeline.embedder is None  # a given detector means replay: vectors come with boxes


def test_looks_in_matching_need_bytetrack():
    config = load_config(
        overrides=["appearance.enabled=true", "appearance.match=veto", "tracker.name=ocsort"]
    )
    with pytest.raises(ConfigError, match="bytetrack"):
        Pipeline.check(config)


@pytest.mark.parametrize(
    "override",
    ["appearance.match=fancy", "appearance.gate=-1", "appearance.blend=1", "appearance.batch=0",
     "appearance.dynamic=1", "appearance.memory=1"],
)  # fmt: skip
def test_bad_settings_are_errors(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_crops_are_rgb_normalised_and_a_box_outside_is_zero():
    image = np.zeros((100, 100, 3), np.uint8)
    image[:, :, 2] = 255  # red, in BGR
    out = crops(image, np.array([[10, 10, 50, 90], [200, 200, 240, 280]], float))
    assert out.shape == (2, 3, 256, 128)
    assert out[0, 0].mean() > out[0, 1].mean()  # the red channel comes first
    assert out[1, 0, 0, 0] == pytest.approx((0 - 0.485) / 0.229)


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "osnet.pt"
    torch.save({"state_dict": OSNet().state_dict()}, path)
    return path


def test_the_embedder_gives_unit_vectors_for_every_box(weights):
    embedder = Embedder(weights, "cpu", batch=2)
    image = np.random.default_rng(0).integers(0, 255, (120, 160, 3), np.uint8)
    found = embedder(image, np.array([[0, 0, 40, 100], [50, 0, 90, 100], [60, 0, 99, 90.0]]))
    assert found.shape == (3, 512)
    assert np.linalg.norm(found, axis=1) == pytest.approx(np.ones(3), abs=1e-4)
    assert embedder(image, np.zeros((0, 4))).shape == (0, 512)


def test_missing_weights_say_where_to_look():
    with pytest.raises(ConfigError, match="appearance.weights"):
        Embedder("nowhere.pt")


def test_saved_vectors_line_up_with_the_saved_boxes(tiny_video, tmp_path, weights):
    n = np.array([1, 0, 2] + [0] * 9)
    saved = SavedDetections(
        xyxy=np.array([[0, 0, 9, 9], [1, 1, 8, 8], [2, 2, 7, 7]], np.float32),
        confidence=np.ones(3, np.float32),
        counts=n,
        index=np.arange(12),
        timestamp=np.arange(12) / 10,
        fps=10.0,
        detector="key",
    )
    look = load_config(overrides=[f"appearance.weights={weights}"]).appearance

    class Fake:  # row number in the first value: shows the order
        def __init__(self, *args) -> None:
            self.rows = 0

        def __call__(self, image, boxes):
            out = np.zeros((len(boxes), 512), np.float32)
            out[:, 0] = np.arange(self.rows, self.rows + len(boxes))
            self.rows += len(boxes)
            return out

    path = tmp_path / "video.npz"
    vectors = cached_embeddings(tiny_video, saved, look, path, build=Fake)
    assert vectors[:, 0].tolist() == [0, 1, 2]
    assert embeddings_path(path, saved, look).exists()
    assert cached_embeddings(tiny_video, saved, look, path, build=None)[2, 0] == 2  # from file
