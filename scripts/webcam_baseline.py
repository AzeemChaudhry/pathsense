"""Phase 0 baseline: YOLO26n + ByteTrack on a webcam or video file.

Draws boxes, "#<id>" labels and trails, overlays a smoothed FPS, and prints a summary on
exit (frames, mean FPS, unique IDs) for NOTES.md. `--save` writes the annotated stream to
an mp4, which is the recording the Phase 0 gate asks for.

    uv run python scripts/webcam_baseline.py                  # webcam 0
    uv run python scripts/webcam_baseline.py --save runs/phase0.mp4
    uv run python scripts/webcam_baseline.py --source clip.mp4 --device cpu

Press "q" or Esc to quit.

How one frame flows through the loop in main():

    camera frame -> YOLO26n (finds people, gives boxes + confidence)
                 -> ByteTrack (links today's boxes to yesterday's, gives each person an ID)
                 -> annotators (draw boxes, "#id" labels, trails)
                 -> window and/or saved mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# This script came first (Phase 0). The helpers it needs now live in the package and are
# tested there: parse_source ("0" means camera 0), configure_qt (fonts and Wayland for the
# window) and FpsMeter (smoothed FPS). `pathsense run` is the configurable successor.
from pathsense.io import parse_source
from pathsense.io.display import configure_qt
from pathsense.timing import FpsMeter

PERSON_CLASS_ID = 0  # YOLO26n is trained on COCO, where class 0 is "person"


##### Command line options #################################################################


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="0", help="camera index, video file or stream URL")
    p.add_argument("--weights", default="yolo26n.pt")
    # The frame is resized to this before YOLO sees it. 640 is what the model was trained on.
    p.add_argument("--imgsz", type=int, default=640)
    # Confidence is how sure YOLO is that a box is a person (0 to 1). We keep it low on
    # purpose: ByteTrack first matches the sure boxes (above 0.6), then uses the unsure ones
    # (0.1 to 0.6) to keep following people who are half hidden or blurred. Filtering at 0.3
    # here would throw most of those away before ByteTrack sees them.
    p.add_argument("--conf", type=float, default=0.1, help="detector confidence threshold")
    p.add_argument("--device", default=None, help="cpu, 0 (first GPU); default: auto")
    p.add_argument("--trail", type=int, default=30, help="trail length in frames")
    p.add_argument("--save", type=Path, default=None, help="write annotated video to this mp4")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Heavy imports live here so the helpers above stay importable without them (CI runs
    # the unit tests without installing torch).
    import cv2
    import numpy as np
    import supervision as sv
    from trackers import ByteTrackTracker
    from ultralytics import YOLO

    configure_qt()

    ##### Open the video source ############################################################
    source = parse_source(args.source)
    is_camera = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")
    # The rate the source *claims*. A video file is exact. A webcam usually says 30, but in
    # dim light many webcams slow down to expose each frame longer (see README), so the real
    # rate can be lower. We only use this number as the frame rate of the saved mp4.
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ##### Build the model, tracker and drawing tools (once, before the loop) ##############
    model = YOLO(args.weights)
    # The very first GPU call is slow (CUDA starts up, memory gets allocated). Run one blank
    # image through now so that cost does not land on the first real frame and skew timing.
    model(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), device=args.device, verbose=False)

    # The tracker remembers every person between frames, so it must be created once. Creating
    # it inside the loop would wipe its memory each frame and give everyone a new ID.
    tracker = ByteTrackTracker(frame_rate=source_fps)

    # Colour by track ID, not by class (everyone is class "person", so all boxes would be
    # the same colour). Now an ID switch shows up as a colour change.
    by_track = sv.ColorLookup.TRACK
    box_annotator = sv.BoxAnnotator(color_lookup=by_track)
    # The label sits above the box. smart_position moves it inside the frame when a
    # close-up box touches the top edge, which would otherwise push the label off screen.
    label_annotator = sv.LabelAnnotator(
        color_lookup=by_track, text_position=sv.Position.TOP_LEFT, smart_position=True
    )
    trace_annotator = sv.TraceAnnotator(color_lookup=by_track, trace_length=args.trail)

    fps_meter = FpsMeter()  # whole loop: camera wait + model + drawing
    infer_meter = FpsMeter()  # model + tracker only, i.e. how fast the AI part is

    ##### Counters for the summary line printed at the end ################################
    writer, written, first_tick = None, 0, None
    frames, infer_seconds, seen_ids = 0, 0.0, set()
    started = time.perf_counter()

    try:
        while True:
            ##### 1. Grab a frame ##########################################################
            tick = time.perf_counter()
            # read() waits until the camera has a new frame. If the camera delivers 20
            # frames a second, this line alone makes the loop wait about 50 ms, so the loop
            # can never run faster than the camera, however fast the model is.
            ok, frame = cap.read()
            if not ok:
                break  # end of a video file, or the camera was unplugged
            first_tick = first_tick or tick

            # Seconds since the first frame. For a camera this is the real clock; for a file
            # it is the frame's position in the video. The tracker uses it to decide how long
            # someone can be missing before their ID is dropped (about 1 second).
            timestamp = tick - first_tick if is_camera else frames / source_fps

            ##### 2. Detect people and track them ##########################################
            infer_start = time.perf_counter()
            result = model(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],  # ignore cars, chairs, cups...
                device=args.device,
                verbose=False,
            )[0]  # the model takes a batch and returns a list; we sent one frame
            # Convert YOLO's output to supervision's Detections (boxes, confidences, class
            # ids), then let ByteTrack match them to the people it saw last frame and fill
            # in detections.tracker_id.
            detections = tracker.update(sv.Detections.from_ultralytics(result), timestamp=timestamp)
            infer_elapsed = time.perf_counter() - infer_start
            infer_seconds += infer_elapsed

            # A brand new person must be seen in 2 frames in a row before ByteTrack trusts
            # them and assigns an ID. Until then (and for boxes it could not match) the ID is
            # -1. We hide those so stray false detections do not flash on screen.
            detections = detections[detections.tracker_id != -1]
            seen_ids.update(int(t) for t in detections.tracker_id)

            ##### 3. Draw boxes, labels, trails and the stats line #########################
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = trace_annotator.annotate(frame.copy(), detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

            # FPS = whole loop (limited by the camera). infer = model + tracker time per
            # frame (limited by the GPU). Comparing the two tells you which one is slow.
            fps = fps_meter.update(time.perf_counter() - tick)
            infer_ms = 1000 / infer_meter.update(infer_elapsed)
            stats = f"FPS {fps:5.1f}  infer {infer_ms:4.0f} ms  people {len(detections)}"
            # Bottom left, so it never covers the ID labels at the top. Drawn twice, thick
            # black then thin green, so it stays readable on light and dark backgrounds.
            origin = (10, annotated.shape[0] - 12)
            for color, thickness in (((0, 0, 0), 4), ((0, 255, 0), 2)):
                cv2.putText(
                    annotated, stats, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness
                )
            frames += 1

            ##### 4. Save to mp4 (optional) ################################################
            if args.save:
                if writer is None:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(args.save), fourcc, source_fps, (w, h))
                # An mp4 has one fixed frame rate (here 30). If the camera only gives 20
                # frames a second and we wrote each once, 10 s of capture would become
                # 200 frames / 30 = 6.7 s of video, playing too fast. So we repeat a frame
                # whenever the video falls behind the real clock, keeping playback real time.
                while written <= timestamp * source_fps:
                    writer.write(annotated)
                    written += 1

            ##### 5. Show the window and check for quit ####################################
            if not args.no_show:
                cv2.imshow("PathSense webcam baseline", annotated)
                # waitKey lets the window redraw and returns the key pressed (27 is Esc).
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass  # Ctrl+C: still fall through to cleanup and the summary
    finally:
        ##### Cleanup: release the camera and finish the mp4 file ##########################
        elapsed = time.perf_counter() - started
        cap.release()
        if writer is not None:
            writer.release()  # without this the mp4 is left unfinished and will not play
        cv2.destroyAllWindows()

    ##### Summary line for NOTES.md ########################################################
    # mean_fps: frames per second over the whole run. infer_ms: average model + tracker time.
    # unique_ids: how many different IDs were handed out. With one person in view this
    # should be 1; anything higher means the tracker lost them and gave them a new ID.
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
