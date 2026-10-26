import cv2
import numpy as np
import pytest

from pathsense.io import RealTimeVideoWriter, VideoSource, parse_source
from pathsense.io.display import configure_qt


def test_parse_source_camera_index_and_path():
    assert parse_source("0") == 0
    assert parse_source(2) == 2
    assert parse_source("clip.mp4") == "clip.mp4"
    assert parse_source("rtsp://cam/1") == "rtsp://cam/1"


def test_video_file_frames_have_index_and_file_time(tiny_video):
    with VideoSource(str(tiny_video)) as source:
        assert not source.is_camera and source.fps == pytest.approx(10.0)
        assert source.frame_count == 12
        frames = list(source)
    assert [f.index for f in frames] == list(range(12))
    assert [f.timestamp for f in frames] == pytest.approx([i / 10 for i in range(12)])
    assert frames[0].image.shape == (48, 64, 3)


def test_missing_source_raises(tmp_path):
    with pytest.raises(OSError, match="Could not open"):
        VideoSource(str(tmp_path / "missing.mp4"))


def test_writer_repeats_frames_to_keep_real_time(tmp_path):
    path = tmp_path / "out.mp4"
    image = np.zeros((48, 64, 3), np.uint8)
    with RealTimeVideoWriter(path, fps=30) as writer:
        for timestamp in (0.0, 0.1, 0.2):  # frames arriving at 10 FPS
            writer.write(image, timestamp)
    # 0.2 s at 30 FPS is frame 6, so 7 frames (0 to 6): the video lasts as long as capture.
    assert writer.frames_written == 7
    assert int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT)) == 7


def test_configure_qt_replaces_missing_font_dir_and_picks_xcb_on_wayland(monkeypatch, tmp_path):
    from pathsense.io import display

    monkeypatch.setattr(display, "SYSTEM_FONT_DIRS", ("/no/such/dir", str(tmp_path)))
    monkeypatch.setenv("QT_QPA_FONTDIR", "/site-packages/cv2/qt/fonts")  # what cv2 sets
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    configure_qt()
    assert display.os.environ["QT_QPA_FONTDIR"] == str(tmp_path)
    assert display.os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_configure_qt_keeps_a_valid_user_setting(monkeypatch, tmp_path):
    from pathsense.io import display

    monkeypatch.setenv("QT_QPA_FONTDIR", str(tmp_path))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    configure_qt()
    assert display.os.environ["QT_QPA_FONTDIR"] == str(tmp_path)
    assert display.os.environ["QT_QPA_PLATFORM"] == "offscreen"
