import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from pathsense.io import camera
from pathsense.io.camera import (
    MAX_TRIES,
    CalibrationStore,
    CameraIdentity,
    CameraSettings,
    Mode,
    calibrate,
    camera_request,
    candidates,
    identify,
    list_modes,
    load_or_calibrate,
    measure,
    parse_modes,
    set_controls,
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

V4L2_CTRLS = """\
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
     exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
"""

MJPG_720 = Mode("MJPG", 1280, 720, 30.0)
YUYV_480 = Mode("YUYV", 640, 480, 30.0)


def quiet(_):
    pass


@pytest.fixture(autouse=True)
def no_v4l2(monkeypatch):
    # No test may touch a real camera's controls.
    monkeypatch.setattr(camera.shutil, "which", lambda name: None)


@pytest.fixture
def fake_v4l2(monkeypatch):
    """v4l2-ctl present, answering from `replies` (args -> stdout or an exception)."""
    calls, replies = [], {"--list-formats-ext": V4L2_FORMATS, "--list-ctrls": V4L2_CTRLS}

    def run(cmd, **kwargs):
        calls.append(cmd[3:])
        reply = replies.get(cmd[3], "")
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(returncode=1 if reply == "FAIL" else 0, stdout=reply)

    monkeypatch.setattr(camera.shutil, "which", lambda name: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr(camera.subprocess, "run", run)
    return SimpleNamespace(calls=calls, replies=replies)


##### Modes ###############################################################################


def test_parse_modes_keeps_the_best_fps_per_size_and_skips_formats_opencv_cannot_decode():
    assert parse_modes(V4L2_FORMATS) == [
        Mode("MJPG", 1920, 1080, 30.0),
        MJPG_720,
        Mode("MJPG", 640, 480, 30.0),
        Mode("YUYV", 1280, 720, 10.0),
        YUYV_480,
    ]
    assert parse_modes("") == []


def test_list_modes_asks_v4l2_and_falls_back_when_it_lists_nothing(fake_v4l2):
    assert list_modes(0, quiet)[0] == Mode("MJPG", 1920, 1080, 30.0)
    fake_v4l2.replies["--list-formats-ext"] = "FAIL"
    assert list_modes(0, quiet) == [Mode(*m) for m in camera.FALLBACK_MODES]


def test_candidates_are_capped_largest_first_and_prefer_raw_at_equal_size():
    modes = parse_modes(V4L2_FORMATS)
    # 1920 is over max_width; raw 1280x720 cannot claim 30 FPS.
    expected = [MJPG_720, YUYV_480, Mode("MJPG", 640, 480, 30.0)]
    assert candidates(modes, max_width=1280, fps=30) == expected


def test_candidates_aim_at_what_fits_and_accept_29_97():
    # 30 FPS only above max_width: aim for the 15 the fitting sizes claim.
    modes = [Mode("MJPG", 1920, 1080, 30.0), Mode("MJPG", 1280, 720, 15.0)]
    assert candidates(modes, max_width=1280, fps=30) == [Mode("MJPG", 1280, 720, 15.0)]
    ntsc = Mode("YUYV", 640, 480, 29.97)
    assert candidates([ntsc], max_width=1280, fps=30) == [ntsc]
    assert candidates([MJPG_720], max_width=320, fps=30) == []


##### Controls ############################################################################


def test_set_controls_sets_only_the_controls_the_camera_has(fake_v4l2):
    wanted = {"exposure_dynamic_framerate": 0, "no_such_control": 1}
    assert set_controls(0, wanted, quiet) == {"exposure_dynamic_framerate": 0}
    assert fake_v4l2.calls[-1] == ["--set-ctrl", "exposure_dynamic_framerate=0"]


@pytest.mark.parametrize("failure", ["FAIL", subprocess.TimeoutExpired("v4l2-ctl", 10)])
def test_set_controls_reports_nothing_set_when_v4l2_fails(fake_v4l2, failure):
    fake_v4l2.replies["--set-ctrl"] = failure
    assert set_controls(0, {"exposure_dynamic_framerate": 0}, quiet) == {}


def test_without_v4l2_controls_are_left_alone_and_said_so():
    lines = []
    assert set_controls(0, {"exposure_dynamic_framerate": 0}, lines.append) == {}
    assert "v4l2-ctl not found" in lines[0]


##### Measure and calibrate ###############################################################


class FakeCapture:
    """A camera that delivers `shape` frames, failing the first `fail_first` reads."""

    def __init__(self, shape=(720, 1280), fail_first=0):
        self.shape, self.fail_first, self.props = shape, fail_first, {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        if self.fail_first:
            self.fail_first -= 1
            return False, None
        return True, np.zeros((*self.shape, 3), np.uint8)

    def release(self):
        pass


def test_measure_counts_frames_and_requests_the_target_fps(monkeypatch):
    cap = FakeCapture(fail_first=2)  # some cameras need a moment after a mode switch
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: cap)
    assert measure(0, MJPG_720, fps=30, warmup=0.0, seconds=0.05) > 0
    assert cap.props[camera.cv2.CAP_PROP_FPS] == 30


def test_measure_rejects_a_mode_the_camera_silently_replaced(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: FakeCapture((480, 640)))
    assert measure(0, MJPG_720, fps=30, warmup=0.0, seconds=0.05) == 0.0


def test_calibrate_keeps_the_largest_mode_at_full_speed_and_stops_there():
    measured = []

    def fake_measure(index, mode, fps):
        measured.append((mode, fps))
        return 29.8

    settings = calibrate(0, modes=[YUYV_480, MJPG_720], measure=fake_measure, log=quiet)
    assert measured == [(MJPG_720, 30.0)]  # one measurement, a few seconds
    assert settings == CameraSettings("MJPG", 1280, 720, 30.0, 29.8, {})


def test_calibrate_falls_back_to_the_fastest_mode_when_none_reaches_the_target():
    speeds = {MJPG_720: 15.0, YUYV_480: 20.0}  # e.g. a dim room with dynamic frame rate on
    settings = calibrate(0, modes=list(speeds), measure=lambda i, m, f: speeds[m], log=quiet)
    assert (settings.width, settings.measured_fps) == (640, 20.0)


def test_calibrate_measures_at_most_max_tries_modes():
    modes = [Mode("MJPG", 100 + i, 100, 30.0) for i in range(MAX_TRIES + 3)]
    measured = []
    calibrate(0, modes=modes, measure=lambda i, m, f: measured.append(m) or 5.0, log=quiet)
    assert len(measured) == MAX_TRIES


def test_calibrate_fails_clearly():
    with pytest.raises(OSError, match="no frames"):
        calibrate(0, modes=[MJPG_720], measure=lambda i, m, f: 0.0, log=quiet)
    with pytest.raises(OSError, match="no mode up to width 320"):
        calibrate(0, max_width=320, modes=[MJPG_720], measure=lambda i, m, f: 30.0, log=quiet)


##### Identity ############################################################################


def fake_sysfs(tmp_path, serial):
    usb = tmp_path / "devices" / "usb1" / "1-2"
    (usb / "1-2:1.0").mkdir(parents=True)
    ids = {"idVendor": "0c45", "idProduct": "6320"} | ({"serial": serial} if serial else {})
    for name, value in ids.items():
        (usb / name).write_text(value + "\n")
    node = tmp_path / "video4linux" / "video0"
    node.mkdir(parents=True)
    (node / "name").write_text("AUKEY PC-W1: AUKEY PC-W1\n")
    (node / "device").symlink_to(usb / "1-2:1.0")
    return tmp_path / "video4linux"


def test_identify_uses_usb_ids_so_the_key_survives_another_port(tmp_path):
    assert identify(0, fake_sysfs(tmp_path, "SN0001")) == CameraIdentity(
        "usb-0c45:6320-SN0001", "AUKEY PC-W1"
    )


def test_identify_without_serial_or_sysfs(tmp_path):
    assert identify(0, fake_sysfs(tmp_path / "a", None)).key == "usb-0c45:6320-noserial"
    assert identify(3, tmp_path / "b").key.endswith("-index-3")


##### Store ###############################################################################

REQUEST = camera_request(1280, 30, dynamic_framerate=False)
SETTINGS = CameraSettings("MJPG", 1280, 720, 30, 29.8, {"exposure_dynamic_framerate": 0})


def test_store_round_trip_keeps_other_cameras(tmp_path):
    store = CalibrationStore(tmp_path / "pathsense" / "cameras.json")
    store.put(CameraIdentity("usb-a", "Cam A"), REQUEST, SETTINGS)
    store.put(CameraIdentity("usb-b", "Cam B"), REQUEST, SETTINGS)
    assert store.get("usb-a", REQUEST) == SETTINGS
    assert store.get("usb-b", REQUEST) == SETTINGS
    assert store.get("usb-c", REQUEST) is None
    assert list(store.path.parent.iterdir()) == [store.path]  # no temporary file left


@pytest.mark.parametrize(
    "request_",
    [
        camera_request(640, 30, dynamic_framerate=False),
        camera_request(1280, 25, dynamic_framerate=False),
        camera_request(1280, 30, dynamic_framerate=None),
    ],
)
def test_a_changed_request_needs_a_new_calibration(tmp_path, request_):
    store = CalibrationStore(tmp_path / "cameras.json")
    store.put(CameraIdentity("usb-a", "Cam A"), REQUEST, SETTINGS)
    assert store.get("usb-a", request_) is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '{"version": 99, "cameras": {}}',
        '{"version": 1, "cameras": []}',
        '{"version": 1, "cameras": {"usb-a": "x"}}',
        '{"version": 1, "cameras": {"usb-a": {"request": REQ, "settings": {"fourcc": "MJPG"}}}}',
        '{"version": 1, "cameras": {"usb-a": {"request": REQ, "settings": '
        '{"fourcc": "MJPG", "width": "1280", "height": 720, "fps": 30, "measured_fps": 30,'
        ' "controls": {}}}}}',
    ],
)
def test_damaged_store_means_calibrate_again(tmp_path, content):
    import json

    path = tmp_path / "cameras.json"
    path.write_text(content.replace("REQ", json.dumps(REQUEST)))
    store = CalibrationStore(path)
    assert store.get("usb-a", REQUEST) is None
    store.put(CameraIdentity("usb-a", "Cam A"), REQUEST, SETTINGS)  # and can be rewritten
    assert store.get("usb-a", REQUEST) == SETTINGS


##### Load or calibrate ###################################################################


@pytest.fixture
def one_camera(monkeypatch):
    calls = SimpleNamespace(calibrate=[], controls=[])
    monkeypatch.setattr(camera, "identify", lambda index: CameraIdentity("usb-a", "Cam A"))

    def fake_calibrate(index, max_width, fps, controls, log):
        calls.calibrate.append(index)
        return CameraSettings("MJPG", 1280, 720, fps, 29.8, {})

    monkeypatch.setattr(camera, "calibrate", fake_calibrate)
    monkeypatch.setattr(camera, "set_controls", lambda i, c, log: calls.controls.append(c))
    return calls


def test_load_or_calibrate_measures_once_then_reuses_and_reapplies_controls(one_camera, tmp_path):
    store, lines = CalibrationStore(tmp_path / "cameras.json"), []
    first = load_or_calibrate(0, REQUEST, store, log=lines.append)
    second = load_or_calibrate(0, REQUEST, store, log=lines.append)
    assert first == second and one_camera.calibrate == [0]
    assert "saved calibration" in lines[-1]
    # The requested controls, even though the saved entry recorded none were set.
    assert one_camera.controls == [{"exposure_dynamic_framerate": 0}]

    load_or_calibrate(0, REQUEST, store, recalibrate=True, log=lines.append)
    assert one_camera.calibrate == [0, 0]


def test_unwritable_store_still_runs_with_the_measured_settings(one_camera, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    lines = []
    settings = load_or_calibrate(0, REQUEST, CalibrationStore(blocker / "x.json"), log=lines.append)
    assert settings.width == 1280 and "not saved" in lines[-1]


def test_below_target_calibration_says_to_measure_again(tmp_path, monkeypatch):
    monkeypatch.setattr(camera, "identify", lambda index: CameraIdentity("usb-a", "Cam A"))
    monkeypatch.setattr(
        camera, "calibrate", lambda *a, **k: CameraSettings("YUYV", 640, 480, 30, 20.0)
    )
    lines = []
    load_or_calibrate(0, REQUEST, CalibrationStore(tmp_path / "c.json"), log=lines.append)
    assert "pathsense calibrate" in lines[-1]


##### VideoSource #########################################################################


def test_video_source_applies_camera_settings(monkeypatch):
    from pathsense.io import source

    cap = FakeCapture()
    monkeypatch.setattr(source.cv2, "VideoCapture", lambda index: cap)
    with source.VideoSource("0", camera=CameraSettings("MJPG", 1280, 720, 30, 29.8)):
        pass
    cv2 = source.cv2
    # Format before size: many cameras reset the size when the format changes.
    assert list(cap.props) == [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ]


##### Second review: fallbacks, logging and edge cases ####################################


def test_stepwise_only_camera_falls_back_and_says_why(fake_v4l2):
    fake_v4l2.replies["--list-formats-ext"] = (
        "\t[0]: 'YUYV' (YUYV 4:2:2)\n\t\tSize: Stepwise 160x120 - 1920x1080 with step 8/8\n"
    )
    lines = []
    assert list_modes(0, lines.append) == [Mode(*m) for m in camera.FALLBACK_MODES]
    assert "no fixed YUYV or MJPG sizes" in lines[0]


def test_failed_mode_listing_says_so(fake_v4l2):
    fake_v4l2.replies["--list-formats-ext"] = "FAIL"
    lines = []
    list_modes(0, lines.append)
    assert "could not list" in lines[0]


def test_set_controls_says_so_when_the_controls_cannot_be_read(fake_v4l2):
    fake_v4l2.replies["--list-ctrls"] = "FAIL"
    lines = []
    assert set_controls(0, {"exposure_dynamic_framerate": 0}, lines.append) == {}
    assert "could not read camera controls" in lines[0]
    assert not any(call[0] == "--set-ctrl" for call in fake_v4l2.calls)


def test_calibrate_aims_at_what_the_camera_claims_and_saves_that_target():
    measured = []
    slow = Mode("MJPG", 1280, 720, 15.0)
    settings = calibrate(
        0, fps=30, modes=[slow], measure=lambda i, m, f: measured.append(f) or 15.0, log=quiet
    )
    assert measured == [15.0] and settings.fps == 15.0


def test_measure_gives_up_after_a_second_without_frames(monkeypatch):
    cap = FakeCapture(fail_first=10**9)
    released = []
    cap.release = lambda: released.append(True)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: cap)
    clock = iter(range(100))  # each call is one second later
    monkeypatch.setattr(camera.time, "perf_counter", lambda: next(clock))
    assert measure(0, MJPG_720, fps=30) == 0.0 and released == [True]


def test_measure_divides_by_the_elapsed_time(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: FakeCapture())
    ticks = iter([0.0, 0.0, 0.0, 10.0, 10.0, 10.5, 11.0, 12.5, 12.5])
    monkeypatch.setattr(camera.time, "perf_counter", lambda: next(ticks))
    # Start 0, first read ok, warm up ends at 10; frames read at 10.5 and 11, the loop ends
    # at 12.5: 2 frames over 2.5 s, not over the nominal 2 s.
    assert measure(0, MJPG_720, fps=30, warmup=1.0, seconds=2.0) == pytest.approx(0.8)


def test_slow_saved_calibration_warns_on_every_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(camera, "identify", lambda index: CameraIdentity("usb-a", "Cam A"))
    monkeypatch.setattr(camera, "set_controls", lambda i, c, log: {})
    store = CalibrationStore(tmp_path / "c.json")
    store.put(CameraIdentity("usb-a", "Cam A"), REQUEST, CameraSettings("YUYV", 640, 480, 30, 20.0))
    lines = []
    load_or_calibrate(0, REQUEST, store, log=lines.append)
    assert "saved calibration" in lines[0] and "pathsense calibrate" in lines[-1]


def test_failed_save_leaves_no_temporary_file(tmp_path, monkeypatch):
    store = CalibrationStore(tmp_path / "cameras.json")

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(camera.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        store.put(CameraIdentity("usb-a", "Cam A"), REQUEST, SETTINGS)
    assert list(tmp_path.iterdir()) == []


def test_video_source_reports_the_size_the_camera_agreed_to(monkeypatch):
    from pathsense.io import source

    cap = FakeCapture()
    monkeypatch.setattr(source.cv2, "VideoCapture", lambda index: cap)
    with source.VideoSource("0", camera=CameraSettings("MJPG", 1280, 720, 30, 29.8)) as src:
        assert src.size == (1280, 720)
