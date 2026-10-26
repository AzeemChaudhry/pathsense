"""Phase 0 baseline: YOLO26n + ByteTrack on a webcam or video file.

Draws boxes, "#<id>" labels and trails, overlays a smoothed FPS, and prints a summary on
exit (frames, mean FPS, unique IDs) for NOTES.md. `--save` writes the annotated stream to
an mp4, which is the recording the Phase 0 gate asks for.

    uv run python scripts/webcam_baseline.py                  # webcam 0
    uv run python scripts/webcam_baseline.py --save runs/phase0.mp4
    uv run python scripts/webcam_baseline.py --source clip.mp4 --device cpu

Press "q" or Esc to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

PERSON_CLASS_ID = 0  # COCO "person"


def parse_source(source: str) -> int | str:
    """A bare integer means a camera index; anything else is a file path or stream URL."""
    return int(source) if source.isdigit() else source


class FpsMeter:
    """Exponential moving average of frames per second, so the overlay does not flicker."""

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.fps = 0.0

    def update(self, frame_seconds: float) -> float:
        if frame_seconds <= 0:
            return self.fps
        instant = 1.0 / frame_seconds
        if self.fps == 0.0:
            self.fps = instant
        else:
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant
        return self.fps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="0", help="camera index, video file or stream URL")
    p.add_argument("--weights", default="yolo26n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    # Low on purpose: ByteTrack's second association pass recovers occluded people from
    # boxes below its 0.6 high confidence threshold, so do not filter them out here.
    p.add_argument("--conf", type=float, default=0.1, help="detector confidence threshold")
    p.add_argument("--device", default=None, help="cpu, 0 (first GPU); default: auto")
    p.add_argument("--trail", type=int, default=30, help="trail length in frames")
    p.add_argument("--save", type=Path, default=None, help="write annotated video to this mp4")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Heavy imports live here so the helpers above stay importable without them.
    import cv2
    import numpy as np
    import supervision as sv
    from trackers import ByteTrackTracker
    from ultralytics import YOLO

    source = parse_source(args.source)
    is_camera = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")
    # Webcams often report 30 FPS but deliver less (auto exposure in dim light). This is the
    # nominal rate: the saved video uses it, with camera frames repeated to stay real time.
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = YOLO(args.weights)
    # Warm up once so CUDA initialisation does not inflate the timing of the first frame.
    model(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), device=args.device, verbose=False)
    tracker = ByteTrackTracker(frame_rate=source_fps)  # once, outside the loop
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
    trace_annotator = sv.TraceAnnotator(trace_length=args.trail)
    fps_meter = FpsMeter()
    infer_meter = FpsMeter()

    writer, written, first_tick = None, 0, None

    frames, infer_seconds, seen_ids = 0, 0.0, set()
    started = time.perf_counter()
    try:
        while True:
            tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            first_tick = first_tick or tick
            # Real capture time for a camera, nominal time for a file, so the tracker's
            # lost track budget is in seconds whatever the processing speed.
            timestamp = tick - first_tick if is_camera else frames / source_fps

            infer_start = time.perf_counter()
            result = model(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],
                device=args.device,
                verbose=False,
            )[0]
            detections = tracker.update(
                sv.Detections.from_ultralytics(result), timestamp=timestamp
            )
            infer_elapsed = time.perf_counter() - infer_start
            infer_seconds += infer_elapsed
            # Unconfirmed tracks and unmatched boxes come back with tracker_id -1; hide them.
            detections = detections[detections.tracker_id != -1]
            seen_ids.update(int(t) for t in detections.tracker_id)

            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = trace_annotator.annotate(frame.copy(), detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

            fps = fps_meter.update(time.perf_counter() - tick)
            infer_ms = 1000 / infer_meter.update(infer_elapsed)
            cv2.putText(
                annotated,
                f"FPS {fps:5.1f}  infer {infer_ms:4.0f} ms  people {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            frames += 1

            if args.save:
                if writer is None:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(args.save), fourcc, source_fps, (w, h))
                # Repeat a frame until the video clock catches up with capture time, so a
                # camera slower than its nominal rate still records at real speed.
                while written <= timestamp * source_fps:
                    writer.write(annotated)
                    written += 1

            if not args.no_show:
                cv2.imshow("PathSense webcam baseline", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - started
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    mean_fps = frames / elapsed if elapsed > 0 else 0.0
    mean_infer_ms = 1000 * infer_seconds / frames if frames else 0.0
    print(
        f"frames={frames} seconds={elapsed:.1f} mean_fps={mean_fps:.1f} "
        f"infer_ms={mean_infer_ms:.1f} unique_ids={len(seen_ids)}"
    )
    if args.save:
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
