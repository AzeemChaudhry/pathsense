"""Manual test with a real webcam: cross the line within a minute.

    PATHSENSE_CAMERA=1 uv run pytest tests/test_live_camera.py -s

A window shows the camera with a vertical line down the middle. Walk or lean across it
(left to right or right to left) and stay there for the confirmation hold
(`analytics.line_confirm`, 0.5 s). The test passes on the first crossing, and fails if
nobody crosses within 60 s or if you press q or Esc. Skipped in CI and in a plain
`pytest` run, which have no camera.
"""

import os
import time

import pytest

TIMEOUT = float(os.environ.get("PATHSENSE_CAMERA_TIMEOUT", "60"))  # seconds
SOURCE = os.environ.get("PATHSENSE_CAMERA_SOURCE", "0")


@pytest.mark.camera
def test_crossing_the_line_on_the_webcam():
    import cv2

    from pathsense.analytics import LineCounter
    from pathsense.annotate import Annotator
    from pathsense.cli import camera_settings
    from pathsense.config import load_config
    from pathsense.io import VideoSource
    from pathsense.io.display import configure_qt
    from pathsense.pipeline import Pipeline

    config = load_config()
    camera = camera_settings(config, SOURCE)  # the saved capture calibration
    configure_qt()
    annotator = Annotator()

    with VideoSource(SOURCE, camera=camera) as source:
        pipeline = Pipeline.from_config(config, frame_rate=source.fps)
        counter, crossing = None, None
        started = time.perf_counter()
        try:
            for result in pipeline.run(source):
                height, width = result.frame.image.shape[:2]
                if counter is None:  # a vertical line through the middle, top to bottom
                    start, end = (width / 2, 0.0), (width / 2, float(height))
                    analytics = config.analytics  # the same hold as `pathsense run`
                    counter = LineCounter(
                        start, end, margin=analytics.line_margin, confirm=analytics.line_confirm
                    )

                events = counter.update(result.tracked, result.frame.timestamp, result.frame.index)
                left = TIMEOUT - (time.perf_counter() - started)

                image = annotator.draw(
                    result.frame.image,
                    result.tracked,
                    f"cross the line: {max(left, 0):4.1f} s left  people {len(result.tracked)}",
                )
                line_colour = (0, 255, 0) if events else (0, 200, 255)
                p1, p2 = (int(start[0]), int(start[1])), (int(end[0]), int(end[1]))
                cv2.line(image, p1, p2, line_colour, 3)
                cv2.imshow("PathSense line test", image)
                key = cv2.waitKey(1) & 0xFF

                if events:
                    crossing = events[0]
                    cv2.waitKey(500)  # show the green line for a moment
                    break
                if key in (ord("q"), 27):
                    pytest.fail("stopped with q or Esc before anyone crossed")
                if left <= 0:
                    pytest.fail(f"nobody crossed the line within {TIMEOUT:.0f} s")
        finally:
            cv2.destroyAllWindows()

    assert crossing is not None, "the camera stopped before anyone crossed"
    print(
        f"\ncrossed: track #{crossing.track_id} {crossing.type} at {crossing.t:.1f} s "
        f"(frame {crossing.frame})"
    )
