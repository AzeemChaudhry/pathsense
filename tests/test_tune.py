import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import detections
from test_pipeline import WalkingPerson

from pathsense.cli import main
from pathsense.config import ConfigError, DetectorConfig, load_config
from pathsense.detect import DETECTORS
from pathsense.eval import CrossingScore
from pathsense.io import VideoSource
from pathsense.pipeline import Pipeline
from pathsense.tune import (
    Candidate,
    SavedDetections,
    cached_detections,
    detect_video,
    load_candidates,
    no_detector,
    replay,
    scene_profile,
    scene_truth,
    tune,
)
from pathsense.tune.cache import cache_path, detector_key
from pathsense.tune.run import CANDIDATES, Score, Trial, TuneReport

# WalkingPerson moves right 4 px a frame across the 64x48 tiny video; a line down the
# middle at x = 20, drawn top to bottom, makes that one exit (pathsense.analytics.line).
LINE = "[[20, 0], [20, 48]]"


@pytest.fixture
def fake_detector(monkeypatch):
    built = []

    def build(config):
        built.append(config)
        return WalkingPerson()

    monkeypatch.setitem(DETECTORS, "fake", build)
    return built


def write_scene(tmp_path, video, truth, extra=""):
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        f"source: {video}\n"
        "detector: {name: fake}\n"
        f"analytics: {{line: {LINE}}}\n"
        f"{extra}"
        f"truth: {truth}\n"
    )
    return scene


def test_saved_detections_survive_a_round_trip(tmp_path):
    saved = SavedDetections(
        xyxy=np.array([[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 9, 9]], np.float32),
        confidence=np.array([0.9, 0.8, 0.7], np.float32),
        counts=np.array([2, 0, 1]),
        index=np.array([0, 1, 2]),
        timestamp=np.array([0.0, 0.1, 0.2]),
        fps=10.0,
        size=(64, 48),
        detector="key",
    )
    saved.save(tmp_path / "d.npz")
    back = SavedDetections.load(tmp_path / "d.npz")
    assert (back.size, back.detector, back.fps, len(back)) == ((64, 48), "key", 10.0, 3)
    frames = list(back)
    assert [len(boxes) for _, boxes in frames] == [2, 0, 1]
    assert frames[2][0].index == 2 and frames[2][0].image is None
    assert frames[0][1].xyxy[1].tolist() == [5, 5, 15, 25]


def test_the_detector_runs_once_per_video_and_settings(tmp_path, tiny_video, fake_detector):
    config = DetectorConfig(name="fake")
    first = cached_detections(tiny_video, config, tmp_path)
    again = cached_detections(tiny_video, config, tmp_path)
    assert len(fake_detector) == 1 and len(first) == len(again) == 12
    assert first.size == (64, 48) and first.detector == detector_key(config, tiny_video)
    cached_detections(tiny_video, DetectorConfig(name="fake", conf=0.3), tmp_path)
    assert len(fake_detector) == 2  # other settings: other boxes
    cached_detections(tiny_video, DetectorConfig(name="fake", conf=0.3, device="cpu"), tmp_path)
    assert len(fake_detector) == 2  # the device does not change the boxes
    cached_detections(tiny_video, DetectorConfig(name="fake", conf=0.3), tmp_path, refresh=True)
    assert len(fake_detector) == 3  # asked to run again


def test_saved_detections_belong_to_one_video(tmp_path, tiny_video, fake_detector):
    config = DetectorConfig(name="fake")
    other = tmp_path / "other" / tiny_video.name  # same name, another folder
    other.parent.mkdir()
    shutil.copy(tiny_video, other)
    assert cache_path(tiny_video, tmp_path) != cache_path(other, tmp_path)
    cached_detections(tiny_video, config, tmp_path)
    cached_detections(other, config, tmp_path)
    assert len(fake_detector) == 2
    os.utime(other, ns=(1, 1))  # the video replaced in place
    cached_detections(other, config, tmp_path)
    assert len(fake_detector) == 3


def test_truth_is_labels_or_counts(tmp_path):
    scene = tmp_path / "s.yaml"
    scene.write_text("truth: {enter: 11, exit: 7, zones: {aisle: 1}}\n")
    truth = scene_truth(scene)
    assert truth.labels is None and truth.counts == {"enter": 11, "exit": 7, "zone:aisle": 1}
    scene.write_text("truth: {labels: gt.txt}\n")
    assert str(scene_truth(scene).labels) == "gt.txt"
    scene.write_text("analytics: {line_confirm: 1}\n")
    assert scene_truth(scene) is None
    for bad in ("{labels: gt.txt, enter: 1}", "{}", "{enter: -1}", "{enter: 1.5}", "{door: 1}"):
        scene.write_text(f"truth: {bad}\n")
        with pytest.raises(ConfigError):
            scene_truth(scene)


def test_the_pipeline_config_ignores_the_truth_block(tmp_path):
    scene = tmp_path / "s.yaml"
    scene.write_text("truth: {enter: 1}\nanalytics: {line_confirm: 1}\n")
    assert load_config(scene=scene).analytics.line_confirm == 1


def test_the_shipped_candidates_load():
    names = [c.name for c in load_candidates(CANDIDATES)]
    assert "default" in names and "crowd" in names and "current" not in names


def test_candidates_are_checked(tmp_path):
    path = tmp_path / "c.yaml"
    for bad in (
        "candidates: []",
        "candidates: [{name: a, why: b}]",
        "candidates: [{name: a, why: b, set: tracker.name=sort}]",
        "candidates: [{name: current, why: b, set: []}]",
        "candidates: [{name: a, why: b, set: []}, {name: a, why: c, set: []}]",
        "candidates: [{name: a, why: b, set: [detector.conf=0.3]}]",  # boxes are saved
        "candidates: [{name: a, why: b, set: ['analytics.line=[[0, 0], [9, 9]]']}]",
        "candidates: [{name: a, why: b, set: ['analytics.zones=[]']}]",
        "candidates: [{name: a, why: b, set: [camera.fps=10]}]",  # only tracker, analytics
    ):
        path.write_text(bad)
        with pytest.raises(ConfigError):
            load_candidates(path)


def test_scores_compare_crossings_first_then_counts():
    hand = Score({"enter": 3}, {"enter": 4, "exit": 1}, None, ids=5, track_ms=1.0)
    assert hand.count_error == 2 and hand.objective == (-2,)
    exact = CrossingScore(tp=Counter(exit=1))
    close = Score({"exit": 2}, {"exit": 1}, exact, ids=1, track_ms=1.0)
    right = Score({"exit": 1}, {"exit": 1}, exact, ids=1, track_ms=1.0)
    assert close.objective < right.objective  # same F1: the count error decides


def test_the_chosen_block_is_ready_to_paste():
    crowd = next(c for c in load_candidates(CANDIDATES) if c.name == "crowd")
    trial = Trial(crowd, Score({}, {}, None, ids=0, track_ms=0.0))
    report = TuneReport(Path("s.yaml"), None, [trial], trial, None)
    block = yaml.safe_load(report.chosen_yaml())
    assert block == {
        "tracker": {
            "name": "bytetrack",
            "params": {
                "lost_track_buffer": 60,
                "minimum_iou_threshold": 0.4,
                "track_activation_threshold": 0.4,
                "iou": "diou",
            },
            "stitch_seconds": 1,
        }
    }


def test_tune_keeps_the_scene_unless_a_candidate_is_strictly_better(
    tmp_path, tiny_video, fake_detector
):
    scene = write_scene(tmp_path, tiny_video, "{exit: 1}")
    blind = Candidate(
        "blind", "no track ever starts", ("tracker.params={track_activation_threshold: 0.99}",)
    )
    report = tune(scene, [blind], cache_folder=tmp_path / "cache")
    assert [t.score.counts.get("exit", 0) for t in report.trials] == [1, 0]
    assert report.chosen.candidate.name == "current" and report.chosen_yaml() == ""
    assert "no candidate scores strictly better" in report.markdown()

    blind_scene = write_scene(
        tmp_path, tiny_video, "{exit: 1}", "tracker: {params: {track_activation_threshold: 0.99}}\n"
    )
    fix = Candidate("fix", "track again", ("tracker.params={track_activation_threshold: 0.5}",))
    report = tune(blind_scene, [fix], cache_folder=tmp_path / "cache")
    assert report.chosen.candidate.name == "fix"
    assert report.chosen_yaml() == "tracker:\n  params: {track_activation_threshold: 0.5}\n"
    assert len(fake_detector) == 1  # both scenes replayed the same saved boxes


def test_tune_with_labels_scores_every_crossing(tmp_path, tiny_video, fake_detector):
    labels = tmp_path / "gt.txt"
    labels.write_text("".join(f"{f + 1},7,{4 * f},5,10,35,1,1,1.0\n" for f in range(12)))
    scene = write_scene(tmp_path, tiny_video, f"{{labels: {labels}}}")
    report = tune(scene, [], cache_folder=tmp_path / "cache")
    score = report.chosen.score
    assert score.truth == {"enter": 0, "exit": 1} and score.counts == {"exit": 1}
    assert score.crossings.f1() == 1.0
    assert report.diagnosis.rules_wrong == report.diagnosis.rules_missed == 0
    assert sum(report.diagnosis.misses.values()) == 0
    assert "0 wrong crossings and 0 missed, out of 1 true" in report.markdown()


def test_tune_needs_a_truth_block(tmp_path, tiny_video):
    scene = tmp_path / "scene.yaml"
    scene.write_text(f"source: {tiny_video}\n")
    with pytest.raises(ConfigError, match="truth"):
        tune(scene, [], cache_folder=tmp_path)


def test_the_profile_describes_people_and_their_speed():
    # Two people 100 px tall, 200 px apart; one walks 10 px a frame at 10 FPS.
    boxes = [[0, 0, 40, 100], [200, 0, 240, 100]] * 2
    saved = SavedDetections(
        xyxy=np.array(boxes, np.float32),
        confidence=np.full(4, 0.9, np.float32),
        counts=np.array([2, 2]),
        index=np.array([0, 1]),
        timestamp=np.array([0.0, 0.1]),
        fps=10.0,
    )
    walker = detections([[0, 0, 40, 100]])
    walker.tracker_id = np.array([1])
    moved = detections([[10, 0, 50, 100]])
    moved.tracker_id = np.array([1])
    profile = scene_profile(saved, {0: walker, 1: moved})
    assert (profile.people, profile.height, profile.spacing) == (2, 100, 2.0)
    assert profile.speed == pytest.approx(1.0)  # 10 px a frame = 1 height a second
    assert "nearest other person: 2.00 heights away" in profile.lines()


def test_the_tune_command_saves_its_report(tmp_path, tiny_video, fake_detector, monkeypatch):
    scene = write_scene(tmp_path, tiny_video, "{exit: 1}")
    candidates = tmp_path / "c.yaml"
    candidates.write_text("candidates: [{name: same, why: test, set: []}]\n")
    monkeypatch.chdir(tmp_path)  # the cache goes to ./data/cache
    main(["tune", str(scene), "--candidates", str(candidates), "--out", "out", "--jobs", "1"])
    assert "# Tuning scene" in (tmp_path / "out" / "scene" / "report.md").read_text()


def test_a_replay_of_saved_detections_gives_the_live_result(tiny_video):
    config = load_config(overrides=["detector.name=fake", f"analytics.line={LINE}"])
    with VideoSource(str(tiny_video)) as source:
        live = Pipeline.from_config(config, 10, detector=WalkingPerson())
        results = list(live.run(source))
        events = [e for r in results for e in r.events] + live.finish()
    saved = detect_video(tiny_video, WalkingPerson())
    again = replay(Pipeline.from_config(config, saved.fps, detector=no_detector), saved)
    assert again.events == events and [e.type for e in events] == ["exit"]
    for result in results:
        got = again.tracks[result.frame.index]
        assert got.tracker_id.tolist() == result.tracked.tracker_id.tolist()
        assert np.allclose(got.xyxy, result.tracked.xyxy)


def test_tune_splits_every_missed_crossing(tmp_path, tiny_video, fake_detector):
    # Labels: person 7 is the one the fake detector sees; person 8 walks the other way
    # across the line, lower in the frame, and is never detected.
    labels = tmp_path / "gt.txt"
    rows = [f"{f + 1},7,{4 * f},5,10,35,1,1,1.0" for f in range(12)]
    rows += [f"{f + 1},8,{50 - 4 * f},30,10,17,1,1,1.0" for f in range(12)]
    labels.write_text("\n".join(rows) + "\n")
    scene = write_scene(tmp_path, tiny_video, f"{{labels: {labels}}}")
    report = tune(scene, [], cache_folder=tmp_path / "cache")
    assert report.chosen.score.truth == {"enter": 1, "exit": 1}
    split = report.diagnosis.split()
    assert split == {"detector": 1, "tracker": 0, "swap": 0, "both": 0, "unclear": 0}
    assert sum(split.values()) == sum(report.chosen.score.crossings.fn.values())
    assert "**Detector:** 1" in report.markdown()


def test_tune_says_which_counts_are_not_scored(tmp_path, tiny_video, fake_detector):
    scene = write_scene(tmp_path, tiny_video, "{enter: 0}")  # the exits are not given
    report = tune(scene, [], cache_folder=tmp_path / "cache")
    assert report.unscored == ("exit",)
    assert "exit (not scored)" in report.markdown()


def test_parallel_and_serial_tuning_agree(tmp_path, tiny_video, fake_detector):
    scene = write_scene(tmp_path, tiny_video, "{exit: 1}")
    candidates = load_candidates(CANDIDATES)[:2]
    serial = tune(scene, candidates, cache_folder=tmp_path / "cache", jobs=1)
    parallel = tune(scene, candidates, cache_folder=tmp_path / "cache", jobs=2)
    assert [t.score.counts for t in serial.trials] == [t.score.counts for t in parallel.trials]
    assert serial.chosen.candidate == parallel.chosen.candidate


def test_every_kind_of_miss_lands_in_one_group():
    from pathsense.eval.misses import DETECTED, TRACKED
    from pathsense.tune.run import Diagnosis

    every = Counter({(t, d): 1 for t in TRACKED for d in DETECTED})
    split = Diagnosis(every, 0, 0).split()
    assert sum(split.values()) == len(TRACKED) * len(DETECTED)
    assert split == {"detector": 5, "swap": 2, "tracker": 3, "both": 4, "unclear": 1}
    alone = {(t, d): Diagnosis(Counter({(t, d): 1}), 0, 0).split() for t, d in every}
    half, some, never = DETECTED
    no_track, one_id, starts, changes, swap = TRACKED
    assert alone[no_track, never]["detector"] == alone[swap, half]["swap"] == 1
    assert alone[changes, half]["tracker"] == alone[starts, some]["both"] == 1
    assert alone[one_id, half]["unclear"] == alone[one_id, some]["both"] == 1


def test_parallel_tuning_with_labels_matches_serial(tmp_path, tiny_video, fake_detector):
    labels = tmp_path / "gt.txt"
    labels.write_text("".join(f"{f + 1},7,{4 * f},5,10,35,1,1,1.0\n" for f in range(12)))
    scene = write_scene(tmp_path, tiny_video, f"{{labels: {labels}}}")
    candidates = load_candidates(CANDIDATES)[:2]
    serial = tune(scene, candidates, cache_folder=tmp_path / "cache", jobs=1)
    parallel = tune(scene, candidates, cache_folder=tmp_path / "cache", jobs=2)
    for a, b in zip(serial.trials, parallel.trials, strict=True):
        assert a.score.crossings == b.score.crossings and a.score.counts == b.score.counts
    assert parallel.chosen in parallel.trials


def test_a_second_tune_call_sees_an_edited_truth(tmp_path, tiny_video, fake_detector):
    scene = write_scene(tmp_path, tiny_video, "{exit: 1}")
    assert tune(scene, [], cache_folder=tmp_path / "cache").chosen.score.count_error == 0
    scene.write_text(scene.read_text().replace("{exit: 1}", "{exit: 3}"))
    assert tune(scene, [], cache_folder=tmp_path / "cache").chosen.score.count_error == 2


def test_a_local_weights_file_is_part_of_the_key(tmp_path, tiny_video):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"one")
    config = DetectorConfig(name="fake", weights=str(weights))
    before = detector_key(config, tiny_video)
    weights.write_bytes(b"retrained")
    assert detector_key(config, tiny_video) != before
    with pytest.raises(ConfigError, match="video not found"):
        detector_key(config, tmp_path / "missing.mp4")


def test_an_interrupted_save_leaves_the_old_file(tmp_path, monkeypatch):
    saved = SavedDetections(
        xyxy=np.zeros((0, 4), np.float32),
        confidence=np.zeros(0, np.float32),
        counts=np.array([0]),
        index=np.array([0]),
        timestamp=np.array([0.0]),
        fps=10.0,
        detector="old",
    )
    path = tmp_path / "d.npz"
    saved.save(path)

    def fail(self, out):
        raise KeyboardInterrupt

    monkeypatch.setattr(SavedDetections, "_write", fail)
    with pytest.raises(KeyboardInterrupt):
        saved.save(path)
    assert SavedDetections.load(path).detector == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d.npz"]  # no temporary left
