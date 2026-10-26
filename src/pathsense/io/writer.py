"""Write annotated frames to an mp4 that plays back at real speed."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class RealTimeVideoWriter:
    """An mp4 has one fixed frame rate. If frames arrive slower than that (a webcam giving 20
    frames a second into a 30 FPS file), writing each once would make playback too fast. So
    each frame is repeated until the video's clock catches up with its timestamp.

        with RealTimeVideoWriter("out.mp4", fps=30) as writer:
            writer.write(image, timestamp=frame.timestamp)
    """

    def __init__(self, path: str | Path, fps: float) -> None:
        self.path = Path(path)
        self.fps = fps
        self.frames_written = 0
        self._writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray, timestamp: float) -> None:
        if self._writer is None:  # open lazily, once the frame size is known
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
        while self.frames_written <= timestamp * self.fps:
            self._writer.write(image)
            self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()  # without this the mp4 is unfinished and will not play
            self._writer = None

    def __enter__(self) -> RealTimeVideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
