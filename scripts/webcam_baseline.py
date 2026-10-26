"""Phase 0 task: YOLO26n + ByteTrack on your own webcam, IDs and trails on screen, FPS printed.

Write this one yourself; the steps below are the outline.

1. Open the webcam with cv2.VideoCapture(0).
2. Load YOLO26n: `from ultralytics import YOLO; model = YOLO("yolo26n.pt")`.
3. For each frame:
   a. Run the model, keep only class 0 (person).
   b. Convert with `sv.Detections.from_ultralytics(result)`.
   c. Update a ByteTrack tracker from the `trackers` library to get tracker IDs.
   d. Draw boxes, "#<id>" labels and trails (sv.BoxAnnotator, LabelAnnotator, TraceAnnotator).
   e. Compute and overlay FPS.
4. Quit on "q"; release the camera.

Done when: a 15 second recording shows you tracked with a stable ID,
and NOTES.md has your FPS and at least two failures you saw.
"""


def main() -> None:
    raise NotImplementedError("Phase 0: implement the webcam baseline (see module docstring).")


if __name__ == "__main__":
    main()
