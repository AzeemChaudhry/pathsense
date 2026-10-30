"""`pathsense` command line.

    pathsense run                                   # webcam 0, configs/default.yaml
    pathsense run --source clip.mp4 --save runs/clip.mp4
    pathsense run --source clip.mp4 --set "analytics.line=[[0,675],[1920,675]]" \
        --events runs/clip.jsonl                    # count crossings, keep the event log
    pathsense run --source clip.mp4 --heatmap runs/heat.png   # where people spent time
    pathsense run --scene configs/scenes/subway.yaml          # one camera's lines and zones
    pathsense run --set detector.name=rfdetr-nano --set tracker.name=botsort
    pathsense run --source clip.mp4 --no-show --max-frames 300   # quick benchmark
    pathsense calibrate                             # measure webcam 0 again and save it
    pathsense tune configs/scenes/subway.yaml       # choose the scene's tracker settings

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
    line_in: int | None = None  # set when analytics.line is configured
    line_out: int | None = None
    zones: tuple[tuple[str, int, int, int], ...] = ()  # (name, entered, exited, inside at end)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"detector={self.detector} tracker={self.tracker} frames={self.frames} "
            f"seconds={self.seconds:.1f} mean_fps={self.fps:.1f} "
            f"detect_ms={self.detect_ms:.1f} track_ms={self.track_ms:.1f} "
            f"unique_ids={self.unique_ids}"
            + (f" in={self.line_in} out={self.line_out}" if self.line_in is not None else "")
            + "".join(f" {n}:in={i},out={o},inside={c}" for n, i, o, c in self.zones)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pathsense", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"pathsense {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="detect and track people in a video or webcam")
    calibrate = commands.add_parser("calibrate", help="measure a webcam's best mode and save it")
    for command in (run, calibrate):
        command.add_argument(
            "--source", default=None, help="camera index, video file or stream URL (default: 0)"
        )
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
        "--scene",
        type=Path,
        default=None,
        help="a camera's lines, zones and settings over the config (configs/scenes/*.yaml); "
        "its source is used when --source is not given",
    )
    run.add_argument(
        "--save", type=Path, default=None, help="write the annotated video to this mp4"
    )
    run.add_argument(
        "--events", type=Path, default=None, help="write the event log to this .jsonl file"
    )
    run.add_argument(
        "--heatmap", type=Path, default=None, help="save where people spent time to this .png"
    )
    run.add_argument("--no-show", action="store_true", help="do not open a window")
    run.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    run.add_argument("--trail", type=int, default=30, help="trail length in frames")
    run.add_argument(
        "--recalibrate", action="store_true", help="measure the webcam again before running"
    )

    tune = commands.add_parser(
        "tune", help="choose a scene's tracker settings against its truth: block"
    )
    tune.add_argument("scene", type=Path, help="scene file with source: and truth:")
    tune.add_argument(
        "--config", type=Path, default=None, help="YAML config (default: configs/default.yaml)"
    )
    tune.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="setups to try (default: configs/tune/candidates.yaml)",
    )
    tune.add_argument(
        "--out", type=Path, default=Path("runs/tune"), help="folder for <scene>/report.md"
    )
    tune.add_argument(
        "--jobs", type=int, default=None, help="candidates replayed at once (default: cores - 2)"
    )
    tune.add_argument(
        "--refresh", action="store_true", help="run the detector again, even if its boxes are saved"
    )
    return parser


def camera_settings(config: Config, source: str, recalibrate: bool = False):
    """Saved or freshly measured settings for a webcam source; None for files and streams."""
    from pathsense.io import parse_source
    from pathsense.io.camera import CalibrationStore, camera_request, load_or_calibrate

    index = parse_source(source)
    cam = config.camera
    if not isinstance(index, int) or not (cam.calibrate or recalibrate):
        return None
    request = camera_request(cam.max_width, cam.fps, cam.dynamic_framerate)
    return load_or_calibrate(index, request, CalibrationStore(cam.store), recalibrate)


def save_image(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path} (use a .png or .jpg name)")


def check_camera_size(image, camera) -> None:
    """Warn if the camera delivers another size than its calibration (same USB ids, but new
    firmware or another driver). Uses a real frame: some backends report a stale size
    before the first read."""
    height, width = image.shape[:2]
    if (width, height) != (camera.width, camera.height):
        print(
            f"camera: delivering {width}x{height}, not the calibrated "
            f"{camera.width}x{camera.height}; run `pathsense calibrate` to measure again"
        )


def run(args: argparse.Namespace) -> Summary:
    # Heavy imports here, so `pathsense --help` and the tests of this module stay fast.
    import cv2

    from pathsense.analytics import EventLog, Heatmap
    from pathsense.annotate import Annotator, draw_line, draw_zone
    from pathsense.config import load_config, scene_source
    from pathsense.io import RealTimeVideoWriter, VideoSource
    from pathsense.io.display import configure_qt, open_window
    from pathsense.pipeline import Pipeline
    from pathsense.timing import FpsMeter

    config = load_config(args.config, args.overrides, args.scene)
    source_name = args.source or (scene_source(args.scene) if args.scene else None) or "0"
    Pipeline.check(config)  # before the camera turns on or weights download
    camera = camera_settings(config, source_name, args.recalibrate)
    if args.recalibrate and camera is None:
        print("pathsense: --recalibrate only applies to cameras; ignored for this source")
    configure_qt()

    with VideoSource(source_name, camera=camera) as source:
        pipeline = Pipeline.from_config(config, frame_rate=source.fps)
        annotator = Annotator(trail_length=args.trail)
        writer = RealTimeVideoWriter(args.save, source.fps) if args.save else None
        loop_fps, detect_fps = FpsMeter(), FpsMeter()
        frames, detect_total, track_total, ids = 0, 0.0, 0.0, set()
        started = last = time.perf_counter()
        counter = pipeline.line_counter
        flash_until: dict[str | None, float] = {}  # line (None) or zone name -> until when
        if args.events and not pipeline.counters:
            print("pathsense: no analytics.line or zones are set, so the event log will be empty")
        if args.events and args.events.exists():
            print(f"pathsense: replacing the event log {args.events}")
        log = EventLog(args.events) if args.events else None
        heatmap, last_image = None, None
        try:
            for result in pipeline.run(source):
                frames += 1
                if frames == 1 and camera:
                    check_camera_size(result.frame.image, camera)
                detect_total += result.detect_ms
                track_total += result.track_ms
                ids.update(int(t) for t in result.tracked.tracker_id)
                if log:
                    log.write(result.events)
                if args.heatmap:
                    if heatmap is None:
                        height, width = result.frame.image.shape[:2]
                        heatmap = Heatmap(width, height, hidden_legs=config.analytics.hidden_legs)
                    heatmap.add(result.tracked, result.frame.timestamp)
                    last_image = result.frame.image

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
                    now_t = result.frame.timestamp
                    for event in result.events:  # green for half a second after an event
                        flash_until[event.zone] = now_t + 0.5
                    for zone in pipeline.zones:
                        flash = now_t <= flash_until.get(zone.name, -1.0)
                        draw_zone(
                            image, zone.name, zone.polygon, zone.occupancy, zone.entered,
                            zone.exited, flash,
                        )  # fmt: skip
                    if counter:
                        flash = now_t <= flash_until.get(None, -1.0)
                        draw_line(image, counter.points, counter.entered, counter.exited, flash)
                    if writer:
                        writer.write(image, result.frame.timestamp)
                    if not args.no_show:
                        if frames == 1:
                            open_window("PathSense", image.shape[1], image.shape[0])
                        cv2.imshow("PathSense", image)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or Esc
                            break
                if args.max_frames and frames >= args.max_frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            seconds = time.perf_counter() - started
            try:
                late = pipeline.finish()  # crossings to confirm, dwell of people inside
                if log:
                    log.write(late)
            finally:  # close everything even if the last write fails (a full disk)
                if log:
                    log.close()
                if heatmap is not None:
                    save_image(args.heatmap, heatmap.render(last_image))
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
        line_in=counter.entered if counter else None,
        line_out=counter.exited if counter else None,
        zones=tuple((z.name, z.entered, z.exited, z.occupancy) for z in pipeline.zones),
    )


def tune_scene(args: argparse.Namespace) -> Path:
    """Run `pathsense tune`; print the report and save it. Returns the report's path."""
    from pathsense.tune import load_candidates, tune
    from pathsense.tune.run import default_jobs

    candidates = load_candidates(args.candidates) if args.candidates else None
    report = tune(
        args.scene,
        candidates,
        config=args.config,
        refresh=args.refresh,
        jobs=args.jobs or default_jobs(),
    )
    path = args.out / args.scene.stem / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.markdown())
    print(report.markdown())
    return path


def main(argv: list[str] | None = None) -> None:
    from pathsense.config import ConfigError, load_config

    args = build_parser().parse_args(argv)
    try:
        if args.command == "tune":
            print(f"saved {tune_scene(args)}")
            return
        if args.command == "calibrate":
            config = load_config(args.config, args.overrides)
            if camera_settings(config, args.source or "0", recalibrate=True) is None:
                raise SystemExit("pathsense: calibrate needs a camera index, e.g. --source 0")
            return
        summary = run(args)
    except (ConfigError, OSError) as error:
        raise SystemExit(f"pathsense: {error}") from None
    print(summary)
    if args.save:
        print(f"saved {args.save}")
    if args.events:
        print(f"saved {args.events}")
    if args.heatmap:
        print(f"saved {args.heatmap}")


if __name__ == "__main__":
    main()
