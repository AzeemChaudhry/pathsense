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


def test_bad_config_exits_with_a_message():
    with pytest.raises(SystemExit, match="Unknown detector 'yolo99'"):
        main(["run", "--no-show", "--set", "detector.name=yolo99", "--source", "0"])
