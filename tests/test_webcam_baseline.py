import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "webcam_baseline", Path(__file__).resolve().parents[1] / "scripts" / "webcam_baseline.py"
)
webcam_baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webcam_baseline)


def test_parse_source_camera_index_and_path():
    assert webcam_baseline.parse_source("0") == 0
    assert webcam_baseline.parse_source("2") == 2
    assert webcam_baseline.parse_source("clip.mp4") == "clip.mp4"
    assert webcam_baseline.parse_source("rtsp://cam/1") == "rtsp://cam/1"


def test_fps_meter_starts_at_first_reading_then_smooths():
    meter = webcam_baseline.FpsMeter(smoothing=0.5)
    assert meter.update(0.1) == pytest.approx(10.0)
    assert meter.update(0.05) == pytest.approx(15.0)  # halfway between 10 and 20


def test_fps_meter_ignores_zero_duration_and_rejects_bad_smoothing():
    meter = webcam_baseline.FpsMeter()
    assert meter.update(0.0) == 0.0
    with pytest.raises(ValueError):
        webcam_baseline.FpsMeter(smoothing=1.0)


def test_parser_defaults():
    args = webcam_baseline.build_parser().parse_args([])
    assert args.source == "0" and args.imgsz == 640 and args.save is None


def test_configure_qt_replaces_missing_font_dir_and_picks_xcb_on_wayland(monkeypatch, tmp_path):
    monkeypatch.setattr(webcam_baseline, "SYSTEM_FONT_DIRS", ("/no/such/dir", str(tmp_path)))
    monkeypatch.setenv("QT_QPA_FONTDIR", "/site-packages/cv2/qt/fonts")  # what cv2 sets
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    webcam_baseline.configure_qt()
    assert webcam_baseline.os.environ["QT_QPA_FONTDIR"] == str(tmp_path)
    assert webcam_baseline.os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_configure_qt_keeps_a_valid_user_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_FONTDIR", str(tmp_path))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    webcam_baseline.configure_qt()
    assert webcam_baseline.os.environ["QT_QPA_FONTDIR"] == str(tmp_path)
    assert webcam_baseline.os.environ["QT_QPA_PLATFORM"] == "offscreen"
