"""`pathsense` command line.

    pathsense run                                   # webcam 0, configs/default.yaml
    pathsense run --source clip.mp4 --save runs/clip.mp4
    pathsense run --set detector.name=rfdetr-nano --set tracker.name=botsort
    pathsense run --source clip.mp4 --no-show --max-frames 300   # quick benchmark
    pathsense calibrate                             # measure webcam 0 again and save it

The first run with a new webcam measures its best capture mode (a few seconds) and saves
it; later runs reuse it (see pathsense.io.camera).

Press "q" or Esc in the window to stop (Ctrl+C without a window). A summary line is
printed at the end for NOTES.md.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pathsense import __version__

if TYPE_CHECKING:
    from pathsense.config import Config


@dataclass(frozen=True)
class Summary:
    detector: str
    tracker: str
    frames: int
    seconds: float
    detect_ms: float  # mean per frame
    track_ms: float
    unique_ids: int

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"detector={self.detector} tracker={self.tracker} frames={self.frames} "
            f"seconds={self.seconds:.1f} mean_fps={self.fps:.1f} "
            f"detect_ms={self.detect_ms:.1f} track_ms={self.track_ms:.1f} "
            f"unique_ids={self.unique_ids}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pathsense", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"pathsense {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="detect and track people in a video or webcam")
    calibrate = commands.add_parser("calibrate", help="measure a webcam's best mode and save it")
    for command in (run, calibrate):
        command.add_argument("--source", default="0", help="camera index, video file or stream URL")
        command.add_argument(
            "--config", type=Path, default=None, help="YAML config (default: configs/default.yaml)"
        )
        command.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="SECTION.KEY=VALUE",
            help="override a config value, e.g. detector.name=rfdetr-nano (repeatable)",
        )
    run.add_argument(
        "--save", type=Path, default=None, help="write the annotated video to this mp4"
    )
    run.add_argument("--no-show", action="store_true", help="do not open a window")
    run.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    run.add_argument("--trail", type=int, default=30, help="trail length in frames")
    run.add_argument(
        "--recalibrate", action="store_true", help="measure the webcam again before running"
    )
    return parser


def camera_settings(config: Config, source: str, recalibrate: bool = False):
    """Saved or freshly measured settings for a webcam source; None for files and streams."""
    from pathsense.io import parse_source
    from pathsense.io.camera import CalibrationStore, camera_request, load_or_calibrate

    index = parse_source(source)
    cam = config.camera
    if recalibrate and not isinstance(index, int):
        print("pathsense: --recalibrate only applies to cameras; ignored for this source")
    if not isinstance(index, int) or not (cam.calibrate or recalibrate):
        return None
    request = camera_request(cam.max_width, cam.fps, cam.dynamic_framerate)
    return load_or_calibrate(index, request, CalibrationStore(cam.store), recalibrate)


def run(args: argparse.Namespace) -> Summary:
    # Heavy imports here, so `pathsense --help` and the tests of this module stay fast.
    import cv2

    from pathsense.annotate import Annotator
    from pathsense.config import load_config
    from pathsense.io import RealTimeVideoWriter, VideoSource
    from pathsense.io.display import configure_qt
    from pathsense.pipeline import Pipeline
    from pathsense.timing import FpsMeter

    config = load_config(args.config, args.overrides)
    Pipeline.check(config)  # before the camera turns on or weights download
    camera = camera_settings(config, args.source, args.recalibrate)
    configure_qt()

    with VideoSource(args.source, camera=camera) as source:
        if camera and source.size != (camera.width, camera.height):
            # Same USB ids, different behaviour (new firmware, another driver): say so.
            print(
                f"camera: delivering {source.size[0]}x{source.size[1]}, not the calibrated "
                f"{camera.width}x{camera.height}; run `pathsense calibrate` to measure again"
            )
        pipeline = Pipeline.from_config(config, frame_rate=source.fps)
        annotator = Annotator(trail_length=args.trail)
        writer = RealTimeVideoWriter(args.save, source.fps) if args.save else None
        loop_fps, detect_fps = FpsMeter(), FpsMeter()
        frames, detect_total, track_total, ids = 0, 0.0, 0.0, set()
        started = last = time.perf_counter()
        try:
            for result in pipeline.run(source):
                frames += 1
                detect_total += result.detect_ms
                track_total += result.track_ms
                ids.update(int(t) for t in result.tracked.tracker_id)

                if writer or not args.no_show:
                    now = time.perf_counter()
                    fps = loop_fps.update(now - last)
                    last = now
                    detect_fps.update(result.detect_ms / 1000)
                    stats = (
                        f"FPS {fps:5.1f}  detect {detect_fps.ms:4.0f} ms  "
                        f"people {len(result.tracked)}"
                    )
                    image = annotator.draw(result.frame.image, result.tracked, stats)
                    if writer:
                        writer.write(image, result.frame.timestamp)
                    if not args.no_show:
                        cv2.imshow("PathSense", image)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or Esc
                            break
                if args.max_frames and frames >= args.max_frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            seconds = time.perf_counter() - started
            if writer:
                writer.close()
            if not args.no_show:
                cv2.destroyAllWindows()

    return Summary(
        detector=pipeline.detector.name,
        tracker=pipeline.tracker.name,
        frames=frames,
        seconds=seconds,
        detect_ms=detect_total / frames if frames else 0.0,
        track_ms=track_total / frames if frames else 0.0,
        unique_ids=len(ids),
    )


def main(argv: list[str] | None = None) -> None:
    from pathsense.config import ConfigError, load_config

    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            config = load_config(args.config, args.overrides)
            if camera_settings(config, args.source, recalibrate=True) is None:
                raise SystemExit("pathsense: calibrate needs a camera index, e.g. --source 0")
            return
        summary = run(args)
    except (ConfigError, OSError) as error:
        raise SystemExit(f"pathsense: {error}") from None
    print(summary)
    if args.save:
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
