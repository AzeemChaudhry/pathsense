import pytest
from test_pipeline import WalkingPerson

from pathsense import __version__
from pathsense.cli import Summary, build_parser, main
from pathsense.detect import DETECTORS


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_run_defaults():
    args = build_parser().parse_args(["run"])
    assert args.source == "0" and args.overrides == [] and not args.no_show


def test_summary_line():
    summary = Summary("yolo26n", "bytetrack", 100, 5.0, 8.4, 0.6, 3)
    assert summary.fps == 20.0
    assert str(summary) == (
        "detector=yolo26n tracker=bytetrack frames=100 seconds=5.0 mean_fps=20.0 "
        "detect_ms=8.4 track_ms=0.6 unique_ids=3"
    )


def test_run_on_a_video_with_overrides_and_save(monkeypatch, tiny_video, tmp_path, capsys):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    out = tmp_path / "out.mp4"
    main(["run", "--source", str(tiny_video), "--no-show", "--save", str(out),
          "--set", "detector.name=fake", "--set", "tracker.name=botsort"])  # fmt: skip
    printed = capsys.readouterr().out
    assert "detector=fake tracker=botsort frames=12" in printed and "unique_ids=1" in printed
    assert out.exists() and out.stat().st_size > 0


def test_max_frames_stops_early(monkeypatch, tiny_video, capsys):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    main(["run", "--source", str(tiny_video), "--no-show", "--max-frames", "5",
          "--set", "detector.name=fake"])  # fmt: skip
    assert "frames=5 " in capsys.readouterr().out


def test_bad_config_is_reported_before_the_source_is_opened(tmp_path):
    # The source does not exist either; the config error must win, so a typo is reported
    # before a camera turns on or weights download.
    missing = str(tmp_path / "missing.mp4")
    with pytest.raises(SystemExit, match="Unknown detector 'yolo99'"):
        main(["run", "--no-show", "--set", "detector.name=yolo99", "--source", missing])
    with pytest.raises(SystemExit, match="Unknown tracker 'deepsort'"):
        main(["run", "--no-show", "--set", "tracker.name=deepsort", "--source", missing])


def test_missing_source_exits_with_a_message(tmp_path):
    with pytest.raises(SystemExit, match="Could not open video source"):
        main(["run", "--no-show", "--source", str(tmp_path / "missing.mp4")])


def test_calibrate_needs_a_camera(tiny_video):
    with pytest.raises(SystemExit, match="needs a camera index"):
        main(["calibrate", "--source", str(tiny_video)])


def test_files_are_never_calibrated(tiny_video):
    from pathsense.cli import camera_settings
    from pathsense.config import load_config

    assert camera_settings(load_config(), str(tiny_video), recalibrate=True) is None
    assert build_parser().parse_args(["run", "--recalibrate"]).recalibrate


def test_calibrate_command_measures_the_camera_again(monkeypatch):
    from pathsense.io import camera

    seen = []
    monkeypatch.setattr(
        camera,
        "load_or_calibrate",
        lambda index, request, store, recalibrate: seen.append((index, recalibrate)) or object(),
    )
    main(["calibrate", "--source", "1"])
    assert seen == [(1, True)]


def test_run_passes_recalibrate_to_the_camera(monkeypatch, capsys):
    import pathsense.cli as cli

    seen = []

    def fake_settings(config, source, recalibrate=False):
        seen.append((source, recalibrate))
        raise OSError("stop before opening a camera")

    monkeypatch.setattr(cli, "camera_settings", fake_settings)
    with pytest.raises(SystemExit, match="stop before"):
        main(["run", "--no-show", "--recalibrate", "--source", "0"])
    assert seen == [("0", True)]


def test_recalibrate_on_a_file_says_it_is_ignored(monkeypatch, tiny_video, capsys):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    main(["run", "--no-show", "--recalibrate", "--source", str(tiny_video),
          "--set", "detector.name=fake"])  # fmt: skip
    assert "only applies to cameras" in capsys.readouterr().out


def test_calibrate_on_a_file_does_not_mention_recalibrate(tiny_video, capsys):
    with pytest.raises(SystemExit, match="needs a camera index"):
        main(["calibrate", "--source", str(tiny_video)])
    assert "--recalibrate" not in capsys.readouterr().out


def test_camera_size_check_uses_the_real_frame(capsys):
    import numpy as np

    from pathsense.cli import check_camera_size
    from pathsense.io.camera import CameraSettings

    calibrated = CameraSettings("MJPG", 1280, 960, 30, 30.0)
    check_camera_size(np.zeros((960, 1280, 3), np.uint8), calibrated)
    assert capsys.readouterr().out == ""  # the normal case: no warning
    check_camera_size(np.zeros((480, 640, 3), np.uint8), calibrated)
    assert "delivering 640x480, not the calibrated 1280x960" in capsys.readouterr().out


def test_run_checks_the_camera_size_once_and_never_for_files(monkeypatch, tiny_video, capsys):
    import numpy as np

    import pathsense.cli as cli
    from pathsense.io import source
    from pathsense.io.camera import CameraSettings

    class SmallCamera:  # accepts the calibrated mode but keeps delivering 640x480
        def __init__(self, index):
            self.left = 3

        def isOpened(self):
            return True

        def set(self, prop, value):
            pass

        def get(self, prop):
            return 30.0

        def read(self):
            self.left -= 1
            return self.left >= 0, np.zeros((480, 640, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    main(["run", "--no-show", "--source", str(tiny_video), "--set", "detector.name=fake"])
    assert "delivering" not in capsys.readouterr().out

    monkeypatch.setattr(source.cv2, "VideoCapture", SmallCamera)
    calibrated = CameraSettings("MJPG", 1280, 960, 30, 30.0)
    monkeypatch.setattr(cli, "camera_settings", lambda *a: calibrated)
    main(["run", "--no-show", "--source", "0", "--set", "detector.name=fake"])
    printed = capsys.readouterr().out
    assert "frames=3 " in printed and printed.count("not the calibrated 1280x960") == 1


def test_summary_adds_line_counts_only_when_a_line_is_set():
    plain = Summary("yolo26n", "bytetrack", 100, 5.0, 8.4, 0.6, 3)
    assert " in=" not in str(plain)
    counted = Summary("yolo26n", "bytetrack", 100, 5.0, 8.4, 0.6, 3, line_in=11, line_out=7)
    assert str(counted).endswith("unique_ids=3 in=11 out=7")


def test_run_with_a_line_counts_and_draws(monkeypatch, tiny_video, tmp_path, capsys):
    # The fake walks right across a vertical line at x = 25 drawn bottom to top: one enter.
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    out = tmp_path / "out.mp4"
    line = "analytics.line=[[25, 48], [25, 0]]"
    main(["run", "--source", str(tiny_video), "--no-show", "--save", str(out),
          "--set", "detector.name=fake", "--set", line])  # fmt: skip
    assert capsys.readouterr().out.splitlines()[0].endswith("in=1 out=0")
    assert out.stat().st_size > 0


def test_events_are_written_including_crossings_confirmed_at_the_end(
    monkeypatch, tiny_video, tmp_path, capsys
):
    from pathsense.analytics import Event, load_events

    # Crossing at frame 7; a 5 s hold is never reached, so only finish() confirms it.
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    events = tmp_path / "events.jsonl"
    main(["run", "--source", str(tiny_video), "--no-show", "--events", str(events),
          "--set", "detector.name=fake", "--set", "analytics.line=[[25, 48], [25, 0]]",
          "--set", "analytics.line_confirm=5"])  # fmt: skip
    assert load_events(events) == [Event(0.7, 7, 0, "enter")]
    assert f"saved {events}" in capsys.readouterr().out


def test_events_without_a_line_warn_and_write_an_empty_log(
    monkeypatch, tiny_video, tmp_path, capsys
):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    events = tmp_path / "events.jsonl"
    main(["run", "--source", str(tiny_video), "--no-show", "--events", str(events),
          "--set", "detector.name=fake"])  # fmt: skip
    assert "event log will be empty" in capsys.readouterr().out
    assert events.read_text() == ""


def test_ctrl_c_still_writes_crossings_waiting_for_confirmation(
    monkeypatch, tiny_video, tmp_path, capsys
):
    from pathsense.analytics import Event, load_events

    class StoppedAfterNineFrames(WalkingPerson):
        def __call__(self, frame):
            if self.calls == 9:
                raise KeyboardInterrupt
            return super().__call__(frame)

    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: StoppedAfterNineFrames())
    events = tmp_path / "events.jsonl"
    events.write_text("an older run\n")
    main(["run", "--source", str(tiny_video), "--no-show", "--events", str(events),
          "--set", "detector.name=fake", "--set", "analytics.line=[[25, 48], [25, 0]]",
          "--set", "analytics.line_confirm=5"])  # fmt: skip
    assert load_events(events) == [Event(0.7, 7, 0, "enter")]
    assert "replacing the event log" in capsys.readouterr().out


def test_events_folder_is_created(monkeypatch, tiny_video, tmp_path):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    events = tmp_path / "runs" / "today" / "events.jsonl"
    line = "analytics.line=[[25, 48], [25, 0]]"
    main(["run", "--source", str(tiny_video), "--no-show", "--events", str(events),
          "--set", "detector.name=fake", "--set", line])  # fmt: skip
    assert events.exists()


def test_a_failing_last_write_still_closes_the_log_and_video(monkeypatch, tiny_video, tmp_path):
    from pathsense.analytics import EventLog

    closed = []
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())

    def write(self, events):
        if list(events):  # only the crossing confirmed at the end
            raise OSError("disk full")

    monkeypatch.setattr(EventLog, "write", write)
    real_close = EventLog.close
    monkeypatch.setattr(EventLog, "close", lambda self: (closed.append("log"), real_close(self)))
    video = tmp_path / "out.mp4"
    with pytest.raises(SystemExit, match="disk full"):
        main(["run", "--source", str(tiny_video), "--no-show", "--save", str(video),
              "--events", str(tmp_path / "e.jsonl"), "--set", "detector.name=fake",
              "--set", "analytics.line=[[25, 48], [25, 0]]",
              "--set", "analytics.line_confirm=5"])  # fmt: skip
    assert closed == ["log"] and video.stat().st_size > 0  # the video was finished too


def test_summary_lists_each_zone():
    summary = Summary("yolo26n", "bytetrack", 10, 1.0, 8.0, 0.5, 2, zones=(("door", 3, 1, 2),))
    assert str(summary).endswith("unique_ids=2 door:in=3,out=1,inside=2")


def test_run_with_a_zone_writes_zone_and_dwell_events(monkeypatch, tiny_video, tmp_path, capsys):
    from pathsense.analytics import load_events

    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    events = tmp_path / "events.jsonl"
    right = "analytics.zones=[{name: right, polygon: [[25, 0], [64, 0], [64, 48], [25, 48]]}]"
    main(["run", "--source", str(tiny_video), "--no-show", "--events", str(events),
          "--save", str(tmp_path / "out.mp4"), "--set", "detector.name=fake",
          "--set", right])  # fmt: skip
    assert [e.type for e in load_events(events)] == ["zone_enter", "dwell"]
    assert "right:in=1,out=0,inside=1" in capsys.readouterr().out


def test_run_saves_a_heatmap(monkeypatch, tiny_video, tmp_path, capsys):
    import cv2

    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: WalkingPerson())
    heat = tmp_path / "maps" / "heat.png"
    main(["run", "--source", str(tiny_video), "--no-show", "--heatmap", str(heat),
          "--set", "detector.name=fake"])  # fmt: skip
    image = cv2.imread(str(heat))
    assert image is not None and image.shape[:2] == (48, 64)
    assert f"saved {heat}" in capsys.readouterr().out
