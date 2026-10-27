"""Capture calibration: find the best capture mode once per camera, then reuse it.

This is about how the camera delivers frames (format, size, frame rate, exposure
controls), not geometric calibration (lens distortion, focal length, camera pose).

Webcams differ, and OpenCV's defaults rarely suit them. This one opens in raw YUYV at
640x480; raw 1280x720 only reaches 10 FPS over USB, while compressed MJPG gives 1280x720
at the full 30 FPS. A camera also ignores requests it cannot meet without saying so.

So the first launch with a new camera measures it:

    1. identify the camera (USB vendor, product and serial on Linux),
    2. list the modes it advertises (`v4l2-ctl --list-formats-ext`),
    3. set controls such as `exposure_dynamic_framerate=0` (steady FPS in dim light),
    4. try modes from the largest picture down and keep the first one that really
       delivers the target FPS, measured after a warm up,
    5. save the result in ~/.config/pathsense/cameras.json.

Later launches find the camera in that file, set its controls again (cameras forget them
on reset) and open it in the saved mode, with no measuring. A change to the request in
the config (`camera.max_width`, `camera.fps`, `camera.dynamic_framerate`) recalibrates.

Without `v4l2-ctl` (macOS, Windows) the camera is keyed by its index, a short list of
common modes is tried, and controls are left as they are.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

SYSFS = Path("/sys/class/video4linux")
STORE_VERSION = 1
FORMATS = ("YUYV", "MJPG")  # what OpenCV decodes; preferred in this order at equal size
# Tried when the camera cannot be asked what it supports.
FALLBACK_MODES = (("MJPG", 1280, 720, 30.0), ("MJPG", 640, 480, 30.0), ("YUYV", 640, 480, 30.0))
MAX_TRIES = 6  # bounds calibration to about 20 s on a camera that reaches no target


@dataclass(frozen=True)
class Mode:
    fourcc: str  # "MJPG" or "YUYV"
    width: int
    height: int
    fps: float  # the most the camera advertises for this format and size


@dataclass(frozen=True)
class CameraSettings:
    fourcc: str
    width: int
    height: int
    fps: float  # requested from the camera
    measured_fps: float  # what it delivered during calibration
    controls: dict[str, int] = field(default_factory=dict)  # v4l2 controls set before opening

    def describe(self) -> str:
        return f"{self.fourcc} {self.width}x{self.height} at {self.measured_fps:.1f} FPS"


@dataclass(frozen=True)
class CameraIdentity:
    key: str  # stable across USB ports and reboots, e.g. "usb-0c45:6320-SN0001"
    name: str  # for people, e.g. "AUKEY PC-W1"


##### Identify the camera ##################################################################


def identify(index: int, sysfs: Path = SYSFS) -> CameraIdentity:
    """Who is camera `index`? USB ids on Linux; the index elsewhere."""
    node = sysfs / f"video{index}"
    try:
        name = (node / "name").read_text().strip().split(":")[0]
        usb = (node / "device").resolve().parent  # the USB device above the video interface
        vendor, product = ((usb / f).read_text().strip() for f in ("idVendor", "idProduct"))
    except OSError:
        return CameraIdentity(f"{sys.platform}-index-{index}", f"camera {index}")
    try:
        serial = (usb / "serial").read_text().strip()
    except OSError:
        serial = "noserial"
    return CameraIdentity(f"usb-{vendor}:{product}-{serial}", name)


##### Ask the camera what it supports ######################################################


def parse_modes(text: str) -> list[Mode]:
    """Modes from `v4l2-ctl --list-formats-ext` output, with the highest FPS of each size."""
    modes, fourcc, size, best = [], None, None, 0.0

    def flush() -> None:
        if fourcc in FORMATS and size and best:
            modes.append(Mode(fourcc, *size, best))

    for line in text.splitlines():
        if m := re.search(r"\[\d+\]: '(\w+)'", line):
            flush()
            fourcc, size, best = m.group(1), None, 0.0
        elif m := re.search(r"Size: Discrete (\d+)x(\d+)", line):
            flush()
            size, best = (int(m.group(1)), int(m.group(2))), 0.0
        elif m := re.search(r"\(([\d.]+) fps\)", line):
            best = max(best, float(m.group(1)))
    flush()
    return modes


def list_modes(index: int, log: Callable[[str], None] = print) -> list[Mode]:
    if shutil.which("v4l2-ctl"):
        if modes := parse_modes(_v4l2(index, "--list-formats-ext") or ""):
            return modes
        log("  v4l2-ctl listed no usable modes; trying common ones")
    return [Mode(*mode) for mode in FALLBACK_MODES]


def target_fps(modes: list[Mode], max_width: int, fps: float) -> float:
    """The FPS to aim for: the request, or less if no mode that fits claims that much."""
    return min(fps, max((m.fps for m in modes if m.width <= max_width), default=fps))


def candidates(modes: list[Mode], max_width: int, fps: float) -> list[Mode]:
    """Modes worth measuring, largest picture first.

    Only sizes up to `max_width` (the detector shrinks frames to 640 anyway, so more is
    wasted decode time) and only modes that claim about the target FPS (29.97 counts as
    30). At equal size, YUYV comes first: it needs no JPEG decoding.
    """
    target = target_fps(modes, max_width, fps)
    fit = [m for m in modes if m.width <= max_width and m.fps >= 0.9 * target]
    return sorted(fit, key=lambda m: (-m.width * m.height, FORMATS.index(m.fourcc)))


##### Camera controls ######################################################################


def set_controls(
    index: int, controls: dict[str, int], log: Callable[[str], None] = print
) -> dict[str, int]:
    """Set the v4l2 controls the camera has; returns the ones that were really set."""
    if not controls:
        return {}
    if not shutil.which("v4l2-ctl"):
        log(f"  v4l2-ctl not found; camera controls left as they are ({controls})")
        return {}
    has = _v4l2(index, "--list-ctrls") or ""
    wanted = {k: v for k, v in controls.items() if re.search(rf"^\s*{k}\s", has, re.M)}
    settings = ",".join(f"{k}={v}" for k, v in wanted.items())
    if wanted and _v4l2(index, "--set-ctrl", settings) is None:
        log(f"  could not set camera controls ({settings})")
        return {}
    return wanted


def _v4l2(index: int, *args: str) -> str | None:
    """stdout of a v4l2-ctl call, or None if it failed or hung."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{index}", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


##### Measure ##############################################################################


def apply_mode(cap: cv2.VideoCapture, fourcc: str, width: int, height: int, fps: float) -> None:
    # Format first: many cameras reset the size when the format changes.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Do not set CAP_PROP_BUFFERSIZE to 1: on this webcam it cut 30 FPS to 16 (NOTES.md).


def measure(index: int, mode: Mode, fps: float, warmup: float = 1.5, seconds: float = 2.0) -> float:
    """Frames per second the camera really delivers in this mode, asked for `fps` (the same
    request VideoSource makes later); 0 if it refuses the mode."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return 0.0
        apply_mode(cap, mode.fourcc, mode.width, mode.height, fps)
        start, ok = time.perf_counter(), False
        while not ok and time.perf_counter() - start < 1.0:  # some need a moment after a switch
            ok, image = cap.read()
        if not ok or image.shape[:2] != (mode.height, mode.width):
            return 0.0  # no frames, or silently another size
        while time.perf_counter() - start < warmup:  # exposure settles in the first second
            cap.read()
        frames, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            frames += cap.read()[0]
        return frames / (time.perf_counter() - start)
    finally:
        cap.release()


##### Calibrate ############################################################################


def calibrate(
    index: int,
    max_width: int = 1280,
    fps: float = 30.0,
    controls: dict[str, int] | None = None,
    measure: Callable[[int, Mode, float], float] = measure,
    modes: list[Mode] | None = None,
    log: Callable[[str], None] = print,
) -> CameraSettings:
    """The largest mode that delivers at least 90 % of the target FPS, else the fastest."""
    applied = set_controls(index, controls or {}, log)
    modes = modes if modes is not None else list_modes(index, log)
    tries = candidates(modes, max_width, fps)
    if not tries:
        found = ", ".join(f"{m.width}x{m.height}" for m in modes)
        raise OSError(f"Camera {index} has no mode up to width {max_width} (found {found})")
    target = target_fps(modes, max_width, fps)
    best: tuple[Mode, float] | None = None
    for mode in tries[:MAX_TRIES]:
        got = measure(index, mode, target)
        log(f"  {mode.fourcc} {mode.width}x{mode.height}: {got:.1f} FPS")
        if best is None or got > best[1]:
            best = (mode, got)
        if got >= 0.9 * target:
            break  # largest picture at full speed; smaller modes cannot beat it
    mode, got = best
    if got == 0:
        raise OSError(f"Camera {index} delivered no frames in any mode")
    return CameraSettings(mode.fourcc, mode.width, mode.height, target, round(got, 1), applied)


##### Saved calibrations ###################################################################


def default_store() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "pathsense" / "cameras.json"


class CalibrationStore:
    """One JSON file, one entry per camera. Kept outside the repo: it describes hardware."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_store()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}  # missing or damaged: calibrate again rather than fail
        if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
            return {}
        cameras = data.get("cameras")
        return cameras if isinstance(cameras, dict) else {}

    def get(self, key: str, request: dict[str, Any]) -> CameraSettings | None:
        """Saved settings for this camera, if they were made for the same request."""
        entry = self._read().get(key)
        if not isinstance(entry, dict) or entry.get("request") != request:
            return None
        try:
            settings = CameraSettings(**entry["settings"])
        except (KeyError, TypeError):
            return None
        valid = isinstance(settings.controls, dict) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in (settings.width, settings.height, settings.fps, settings.measured_fps)
        )
        return settings if valid and settings.fourcc in FORMATS else None

    def put(
        self, camera: CameraIdentity, request: dict[str, Any], settings: CameraSettings
    ) -> None:
        """Save one camera, keeping the others. Raises OSError if the file cannot be written."""
        cameras = self._read()
        cameras[camera.key] = {
            "name": camera.name,
            "calibrated": datetime.now(UTC).isoformat(timespec="seconds"),
            "request": request,
            "settings": asdict(settings),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": STORE_VERSION, "cameras": cameras}
        # Write a temporary file, then swap it in: a crash never leaves half a file.
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, self.path)


##### Load or calibrate ####################################################################


def camera_request(max_width: int, fps: float, dynamic_framerate: bool | None) -> dict[str, Any]:
    """What the config asks for. A saved calibration is reused only for the same request."""
    controls = (
        {} if dynamic_framerate is None else {"exposure_dynamic_framerate": int(dynamic_framerate)}
    )
    return {"max_width": max_width, "fps": fps, "controls": controls}


def load_or_calibrate(
    index: int,
    request: dict[str, Any],
    store: CalibrationStore,
    recalibrate: bool = False,
    log: Callable[[str], None] = print,
) -> CameraSettings:
    camera = identify(index)
    saved = None if recalibrate else store.get(camera.key, request)
    if saved:
        # The requested controls, not only the saved ones: if v4l2-ctl was missing at
        # calibration and is installed now, they get set. Cameras forget them on replug.
        set_controls(index, request["controls"], log)
        log(f"camera: {camera.name}: saved calibration, {saved.describe()}")
        _warn_if_slow(saved, log)
        return saved
    log(f"camera: {camera.name}: calibrating, a few seconds...")
    settings = calibrate(index, request["max_width"], request["fps"], request["controls"], log=log)
    try:
        store.put(camera, request, settings)
        log(f"camera: {camera.name}: {settings.describe()}, saved to {store.path}")
    except OSError as error:  # the settings are still good for this run
        log(f"camera: {camera.name}: {settings.describe()}, not saved ({error})")
    _warn_if_slow(settings, log)
    return settings


def _warn_if_slow(settings: CameraSettings, log: Callable[[str], None]) -> None:
    if settings.measured_fps < 0.9 * settings.fps:
        log(
            f"camera: below the {settings.fps:g} FPS target (dim light, or the camera was "
            "busy?); run `pathsense calibrate` to measure again"
        )
