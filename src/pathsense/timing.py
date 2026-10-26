"""Frame rate and latency measurement."""

from __future__ import annotations


class FpsMeter:
    """Exponential moving average of frames per second, so an on screen number does not flicker.

    Raw FPS jumps every frame (one frame takes 45 ms, the next 55 ms, so the number flips
    between 22 and 18). Instead of showing the raw value we blend it into the running value:

        shown = 0.9 * previous_shown + 0.1 * this_frame

    Each new frame only moves the number 10 percent of the way, so it changes smoothly and
    settles on the true average within about a second. `smoothing` is that 0.9: higher is
    steadier but slower to react.
    """

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.fps = 0.0

    def update(self, frame_seconds: float) -> float:
        if frame_seconds <= 0:
            return self.fps
        instant = 1.0 / frame_seconds  # e.g. 0.05 s per frame -> 20 FPS
        if self.fps == 0.0:
            self.fps = instant  # first frame: nothing to blend with yet
        else:
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant
        return self.fps

    @property
    def ms(self) -> float:
        """The same average as milliseconds per frame."""
        return 1000 / self.fps if self.fps else 0.0
