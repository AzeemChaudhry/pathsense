"""Read frames from a webcam, a video file or a stream URL, each with a timestamp."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


def parse_source(source: str | int) -> int | str:
    """A bare integer means a camera index; anything else is a file path or stream URL."""
    # OpenCV opens /dev/video0 when given the int 0, but treats the string "0" as a filename.
    if isinstance(source, int):
        return source
    return int(source) if source.isdigit() else source


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, as OpenCV reads it
    index: int  # 0, 1, 2... in the order frames were read
    timestamp: float  # seconds since the first frame


class VideoSource:
    """Iterate over frames:

        with VideoSource("0") as source:
            for frame in source:
                ...

    Timestamps come from the real clock for a camera (it may deliver fewer frames than it
    claims, e.g. 20 instead of 30 in dim light) and from the frame position for a file (so
    a file processed faster or slower than real time still has correct times).
    """

    def __init__(self, source: str | int) -> None:
        self.source = parse_source(source)
        self.is_camera = isinstance(self.source, int)
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise OSError(f"Could not open video source {source!r}")
        # What the source claims. Exact for files; for cameras only a nominal value.
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0

    def __iter__(self) -> Iterator[Frame]:
        index, first_tick = 0, None
        while True:
            ok, image = self._cap.read()
            if not ok:
                return  # end of file, or the camera was disconnected
            if self.is_camera:
                now = time.perf_counter()
                first_tick = first_tick if first_tick is not None else now
                timestamp = now - first_tick
            else:
                timestamp = index / self.fps
            yield Frame(image, index, timestamp)
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
