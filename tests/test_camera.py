import pytest

from pathsense.io import camera
from pathsense.io.camera import (
    CalibrationStore,
    CameraIdentity,
    CameraSettings,
    Mode,
    calibrate,
    camera_request,
    candidates,
    identify,
    load_or_calibrate,
    parse_modes,
)

# Trimmed `v4l2-ctl --list-formats-ext` output of the Phase 0 webcam.
V4L2_FORMATS = """\
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.050s (20.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 1280x720
			Interval: Discrete 0.100s (10.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.067s (15.000 fps)
	[2]: 'H264' (H.264, compressed)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
"""

MJPG_720 = Mode("MJPG", 1280, 720, 30.0)
YUYV_480 = Mode("YUYV", 640, 480, 30.0)


@pytest.fixture(autouse=True)
def no_v4l2(monkeypatch):
    # No test may touch a real camera's controls.
    monkeypatch.setattr(camera.shutil, "which", lambda name: None)


def test_parse_modes_keeps_the_best_fps_per_size_and_skips_formats_opencv_cannot_decode():
    assert parse_modes(V4L2_FORMATS) == [
        Mode("MJPG", 1920, 1080, 30.0),
        MJPG_720,
        Mode("MJPG", 640, 480, 30.0),
        Mode("YUYV", 1280, 720, 10.0),
        YUYV_480,
    ]


def test_candidates_are_capped_largest_first_and_prefer_raw_at_equal_size():
    modes = parse_modes(V4L2_FORMATS)
    # 1920 is over max_width; raw 1280x720 cannot claim 30 FPS.
    assert candidates(modes, max_width=1280, fps=30) == [
        MJPG_720,
        YUYV_480,
        Mode("MJPG", 640, 480, 30.0),
    ]


def test_calibrate_keeps_the_largest_mode_at_full_speed_and_stops_there():
    measured = []

    def measure(index, mode):
        measured.append(mode)
        return 29.8

    settings = calibrate(0, modes=[YUYV_480, MJPG_720], measure=measure, log=lambda _: None)
    assert measured == [MJPG_720]  # one measurement, a few seconds
    assert settings == CameraSettings("MJPG", 1280, 720, 30.0, 29.8, {})


def test_calibrate_falls_back_to_the_fastest_mode_when_none_reaches_the_target():
    speeds = {MJPG_720: 15.0, YUYV_480: 20.0}  # e.g. a dim room with dynamic frame rate on
    settings = calibrate(0, modes=list(speeds), measure=lambda i, m: speeds[m], log=lambda _: None)
    assert (settings.width, settings.measured_fps) == (640, 20.0)


def test_calibrate_fails_clearly_when_the_camera_gives_nothing():
    with pytest.raises(OSError, match="no frames"):
        calibrate(0, modes=[MJPG_720], measure=lambda i, m: 0.0, log=lambda _: None)


def test_identify_uses_usb_ids_so_the_key_survives_another_port(tmp_path):
    usb = tmp_path / "devices" / "usb1" / "1-2"
    (usb / "1-2:1.0").mkdir(parents=True)
    for name, value in {"idVendor": "0c45", "idProduct": "6320", "serial": "SN0001"}.items():
        (usb / name).write_text(value + "\n")
    node = tmp_path / "video4linux" / "video0"
    node.mkdir(parents=True)
    (node / "name").write_text("AUKEY PC-W1: AUKEY PC-W1\n")
    (node / "device").symlink_to(usb / "1-2:1.0")

    assert identify(0, tmp_path / "video4linux") == CameraIdentity(
        "usb-0c45:6320-SN0001", "AUKEY PC-W1"
    )


def test_identify_falls_back_to_the_index_without_sysfs(tmp_path):
    assert identify(3, tmp_path).key.endswith("-index-3")


def test_store_round_trip_and_a_changed_request_needs_a_new_calibration(tmp_path):
    store = CalibrationStore(tmp_path / "pathsense" / "cameras.json")
    request = camera_request(1280, 30, dynamic_framerate=False)
    settings = CameraSettings("MJPG", 1280, 720, 30, 29.8, {"exposure_dynamic_framerate": 0})
    store.put(CameraIdentity("usb-a", "Cam A"), request, settings)

    assert store.get("usb-a", request) == settings
    assert store.get("usb-b", request) is None  # another camera
    assert store.get("usb-a", camera_request(640, 30, dynamic_framerate=False)) is None


def test_damaged_store_means_calibrate_again(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text("{not json")
    assert CalibrationStore(path).get("usb-a", camera_request(1280, 30, None)) is None


def test_load_or_calibrate_measures_once_then_reuses(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(camera, "identify", lambda index: CameraIdentity("usb-a", "Cam A"))

    def fake_calibrate(index, max_width, fps, controls, log):
        calls.append(index)
        return CameraSettings("MJPG", 1280, 720, fps, 29.8, {})

    monkeypatch.setattr(camera, "calibrate", fake_calibrate)
    store, request = CalibrationStore(tmp_path / "cameras.json"), camera_request(1280, 30, None)
    lines = []

    first = load_or_calibrate(0, request, store, log=lines.append)
    second = load_or_calibrate(0, request, store, log=lines.append)
    assert first == second and calls == [0]
    assert "saved calibration" in lines[-1]

    load_or_calibrate(0, request, store, recalibrate=True, log=lines.append)
    assert calls == [0, 0]


def test_video_source_applies_camera_settings(monkeypatch):
    from pathsense.io import source

    calls = []

    class FakeCapture:
        def __init__(self, index):
            pass

        def isOpened(self):
            return True

        def set(self, prop, value):
            calls.append(prop)

        def get(self, prop):
            return 30.0

        def release(self):
            pass

    monkeypatch.setattr(source.cv2, "VideoCapture", FakeCapture)
    settings = CameraSettings("MJPG", 1280, 720, 30, 29.8)
    with source.VideoSource("0", camera=settings):
        pass
    cv2 = source.cv2
    # Format before size: many cameras reset the size when the format changes.
    assert calls == [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ]
