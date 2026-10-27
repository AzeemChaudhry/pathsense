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


def list_modes(index: int) -> list[Mode]:
    if shutil.which("v4l2-ctl"):
        out = _v4l2(index, "--list-formats-ext")
        if modes := parse_modes(out):
            return modes
    return [Mode(*mode) for mode in FALLBACK_MODES]


def candidates(modes: list[Mode], max_width: int, fps: float) -> list[Mode]:
    """Modes worth measuring, largest picture first.

    Only sizes up to `max_width` (the detector shrinks frames to 640 anyway, so more is
    wasted decode time) and only modes that claim the target FPS. At equal size, YUYV comes
    first: it needs no JPEG decoding.
    """
    target = min(fps, max((m.fps for m in modes), default=fps))
    fit = [m for m in modes if m.width <= max_width and m.fps >= target]
    return sorted(fit, key=lambda m: (-m.width * m.height, FORMATS.index(m.fourcc)))


##### Camera controls ######################################################################


def set_controls(index: int, controls: dict[str, int]) -> dict[str, int]:
    """Set v4l2 controls the camera has; returns the ones that were set."""
    if not controls or not shutil.which("v4l2-ctl"):
        return {}
    has = _v4l2(index, "--list-ctrls")
    applied = {k: v for k, v in controls.items() if re.search(rf"^\s*{k}\s", has, re.M)}
    if applied:
        _v4l2(index, "--set-ctrl", ",".join(f"{k}={v}" for k, v in applied.items()))
    return applied


def _v4l2(index: int, *args: str) -> str:
    result = subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{index}", *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


##### Measure ##############################################################################


def apply_mode(cap: cv2.VideoCapture, fourcc: str, width: int, height: int, fps: float) -> None:
    # Format first: many cameras reset the size when the format changes.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Do not set CAP_PROP_BUFFERSIZE to 1: on this webcam it cut 30 FPS to 16 (NOTES.md).


def measure(index: int, mode: Mode, warmup: float = 1.5, seconds: float = 2.0) -> float:
    """Frames per second the camera really delivers in this mode; 0 if it refuses it."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return 0.0
        apply_mode(cap, mode.fourcc, mode.width, mode.height, mode.fps)
        ok, image = cap.read()
        if not ok or image.shape[:2] != (mode.height, mode.width):
            return 0.0  # silently gave another size
        start = time.perf_counter()
        while time.perf_counter() - start < warmup:  # exposure settles in the first second
            cap.read()
        frames, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            frames += cap.read()[0]
        return frames / seconds
    finally:
        cap.release()


##### Calibrate ############################################################################


def calibrate(
    index: int,
    max_width: int = 1280,
    fps: float = 30.0,
    controls: dict[str, int] | None = None,
    measure: Callable[[int, Mode], float] = measure,
    modes: list[Mode] | None = None,
    log: Callable[[str], None] = print,
) -> CameraSettings:
    """The largest mode that delivers at least 90 % of the target FPS, else the fastest."""
    applied = set_controls(index, controls or {})
    tries = candidates(modes if modes is not None else list_modes(index), max_width, fps)
    if not tries:
        raise OSError(f"Camera {index} advertises no usable mode up to width {max_width}")
    target = min(fps, max(m.fps for m in tries))
    best: tuple[Mode, float] | None = None
    for mode in tries[:MAX_TRIES]:
        got = measure(index, mode)
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
        return data.get("cameras", {})

    def get(self, key: str, request: dict[str, Any]) -> CameraSettings | None:
        """Saved settings for this camera, if they were made for the same request."""
        entry = self._read().get(key)
        if not entry or entry.get("request") != request:
            return None
        try:
            return CameraSettings(**entry["settings"])
        except (KeyError, TypeError):
            return None

    def put(
        self, camera: CameraIdentity, request: dict[str, Any], settings: CameraSettings
    ) -> None:
        cameras = self._read()
        cameras[camera.key] = {
            "name": camera.name,
            "calibrated": datetime.now(UTC).isoformat(timespec="seconds"),
            "request": request,
            "settings": asdict(settings),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": STORE_VERSION, "cameras": cameras}
        self.path.write_text(json.dumps(data, indent=2) + "\n")


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
        set_controls(index, saved.controls)  # cameras forget controls on reset or replug
        log(f"camera: {camera.name}: saved calibration, {saved.describe()}")
        return saved
    log(f"camera: {camera.name}: calibrating, a few seconds...")
    settings = calibrate(index, request["max_width"], request["fps"], request["controls"], log=log)
    store.put(camera, request, settings)
    log(f"camera: {camera.name}: {settings.describe()}, saved to {store.path}")
    return settings
