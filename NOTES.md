# Experiment log

One entry per thing tried: what, the number it produced, and why it was kept or dropped.
This becomes the "Design decisions" section of the README.

## Template

### YYYY-MM-DD: short title
- **Tried:**
- **Result:** (FPS, HOTA, IDF1, ID switches, what broke)
- **Decision:** kept / dropped, and why

## Phase 0: webcam baseline

### 2026-10-27: YOLO26n + ByteTrack on webcam
- **Tried:** `scripts/webcam_baseline.py` with defaults (YOLO26n COCO weights, imgsz 640,
  conf 0.1, ByteTrack defaults), one person, laptop webcam at 640x480 in a dim room,
  115 s recording (`runs/phase0.mp4`, not committed).
- **Hardware:** GTX 1650 (4 GB), CUDA available: yes (torch with CUDA 13, driver 615).
- **Run summary:** `frames=2364 seconds=115.1 mean_fps=20.5 infer_ms=20.1 unique_ids=5`
- **FPS:**

  | Setup | Loop FPS | Detect + track |
  | --- | --- | --- |
  | Webcam, GPU, camera dynamic frame rate on (default) | 20.5 | 20.1 ms |
  | Webcam, GPU, dynamic frame rate off | 29.3 | 14.4 ms |
  | Webcam, CPU | 19.6 | 32.6 ms |
  | Video file, GPU (no camera limit) | 92 | 8.4 ms |

  The webcam caps the loop: with `exposure_dynamic_framerate=1` it delivers a steady
  20.0 FPS in dim light (30.0 with it off; measured after a 2 s warm up, since the first
  second runs slower while exposure settles). The model is never the bottleneck. Inference
  time drops when the GPU is busier (8.4 ms on a file versus 20 ms waiting on the
  camera), most likely because the GPU clocks down while idle between frames.
- **Failures seen:**
  1. **Re-entry gets a new ID.** One person, five IDs. ID #0 held for the first 76 s,
     including walking to the back of the room and back, turning sideways and motion
     blur. After leaving the frame for about 8 s, every reappearance got a new ID.
     ByteTrack matches on motion and box overlap only (no appearance features) and drops
     a lost track after about 1 s (`lost_track_buffer=30` at 30 FPS).
  2. **Partly visible people are fragmented or missed.** A hand or arm at the frame edge
     became short tracks #1, #2 and #3 (86 to 98 s). A torso cut off at the top and
     sides was not detected at all at 60 s and 94 to 100 s (`people 0`). COCO trained
     YOLO26n is weak on heavily truncated people, and short tracks must survive
     `minimum_consecutive_frames=2` to get an ID.
  3. Display bug (fixed): labels were drawn above the box, so close-ups touching the top
     edge showed no ID. Now `smart_position=True` and colour per track ID.
- **Decision:** kept as the Phase 0 baseline. Re-entry is the failure that matters most
  for counting and dwell time, because one person counted as five inflates every number.
  Phase 3 compares a longer `lost_track_buffer` and an appearance based re-identification
  step (correction, Phase 1: the `trackers` library's BoT-SORT has no appearance
  features, only camera motion compensation). Truncated people at the frame edge are a
  fine tuning target for Phase 4.

## Phase 1: pipeline package

### 2026-10-27: detector speed on the GTX 1650
- **Tried:** YOLO26n and RF-DETR Nano on frames of `people-walking.mp4` (1920x1080,
  supervision sample clip), 30 frames after 5 warm up frames, CUDA synchronised.
- **Result:**

  | Detector | Mode | ms per frame | People in frame 1 (conf 0.3) |
  | --- | --- | --- | --- |
  | YOLO26n | default, imgsz 640 | 16.3 | 21 |
  | RF-DETR Nano | default FP32, 384 | 31.6 | 25 |
  | RF-DETR Nano | compiled FP32 | 30.3 | 25 |
  | RF-DETR Nano | compiled FP16 | 142.7 | 25 |

- **Decision:** RF-DETR runs in plain FP32. FP16 is about 4.5x slower on this card: the
  GTX 1650 (Turing TU117) has no tensor cores, so half precision gains nothing and the
  casts cost time. Compiling saves about 1 ms for a long start up, not worth it. RF-DETR's
  start up advice to use FP16 is silenced in the adapter because it is wrong here.
  RF-DETR reports COCO category ids (person is 1, not 0); the adapter maps it.

### 2026-10-27: Phase 1 gate, swap detector and tracker with one config line
- **Tried:** `pathsense run --source data/samples/people-walking.mp4 --no-show` with one
  `--set detector.name=...` and one `--set tracker.name=...` per run, full clip (341
  frames, 25 FPS, about 20 to 30 people in view). Times are means per frame.
- **Result:**

  | Detector | Tracker | FPS | detect ms | track ms | unique IDs |
  | --- | --- | --- | --- | --- | --- |
  | yolo26n | bytetrack | 95.2 | 8.3 | 0.7 | 32 |
  | yolo26n | botsort | 30.2 | 14.4 | 17.4 | 32 |
  | yolo26n | botsort, `enable_cmc: false` | 94.0 | 8.2 | 1.0 | 32 |
  | yolo26n | ocsort | 96.1 | 8.1 | 0.9 | 38 |
  | rfdetr-nano | bytetrack | 24.7 | 38.5 | 0.6 | 29 |
  | rfdetr-nano | botsort | 19.8 | 38.8 | 10.2 | 30 |
  | rfdetr-nano | ocsort | 24.5 | 38.6 | 0.8 | 33 |

  Webcam through the new CLI: 27.1 FPS, 1 ID for 1 person.
- **What it shows:**
  1. Every combination runs from one config line, and the gate passes.
  2. YOLO26n detects 4.6x faster than RF-DETR Nano on 1080p frames.
  3. BoT-SORT's camera motion compensation (sparse optical flow on every full frame) costs
     10 to 17 ms per frame and even slows detection (CPU contention). With a fixed camera
     it does nothing useful; turned off, BoT-SORT costs 1 ms and gives the same IDs here.
  4. `unique_ids` is not an accuracy measure without ground truth: more IDs can mean
     more ID switches or more people correctly found. OC-SORT's higher count needs
     checking. Phase 3 measures this properly with TrackEval (HOTA, IDF1, ID switches).
- **Decision:** gate passed; default stays YOLO26n + ByteTrack (fastest, fewest IDs).
  Fixed camera configs should set `enable_cmc: false` when using BoT-SORT.

### 2026-10-27: why visible people go untracked
- **Tried:** the documentation frame showed several clearly visible people without an ID.
  Counted detections per frame by confidence on the full sample clip, then re-ran
  ByteTrack with lower `track_activation_threshold` values on the same detections.
- **Result:**

  | Detector | boxes >= 0.3 | >= 0.5 | >= 0.7 | tracked (activation 0.7, default) | tracked (0.5) | tracked (0.4) |
  | --- | --- | --- | --- | --- | --- | --- |
  | yolo26n | 23.3 | 14.8 | 4.8 | 14.4 (32 IDs) | 17.7 (39 IDs) | 17.7 (39 IDs) |
  | rfdetr-nano | 23.8 | 11.8 | 3.9 | 12.5 (29 IDs) | 15.2 (33 IDs) | 15.2 (33 IDs) |

  Values are means per frame. 0.4 changes nothing over 0.5, most likely because new tracks
  also depend on `high_conf_det_threshold` (0.6), which stayed fixed.
- **Decision:** defaults unchanged for now. Small people seen from above rarely score 0.7,
  so the tracker, not the detector, is what drops them. Lowering the threshold tracks more
  people but also creates more IDs, which could be more ID switches. Tune both thresholds
  against MOT17 ground truth in Phase 3 rather than guessing here.

### 2026-10-27: webcam capture modes and capture calibration
- **Tried:** every capture mode of the test webcam through OpenCV, 2 s warm up then 3 to 4 s
  of measuring, with `exposure_dynamic_framerate` on and off.
- **Result:**

  | Mode | Dynamic frame rate on (dim room) | Off |
  | --- | --- | --- |
  | YUYV 640x480 (OpenCV's default) | 20.0 | 30.2 |
  | YUYV 1280x720 | 10.3 | not tried |
  | MJPG 1280x720 | 19.7 | 30.0 |
  | MJPG 1280x960 (calibration's pick) | | 30.5 |

  Raw YUYV above 640x480 does not fit through USB 2 at full speed (the camera advertises
  only 10 FPS for it). MJPG is compressed in the camera, so it reaches 30 FPS at 4 times
  the pixels.
  A first probe that also set `CAP_PROP_BUFFERSIZE=1` measured only 15.7 to 20 FPS with the
  dynamic frame rate off. With one buffer the camera waits for each read before it can
  capture the next frame. Without it, 30 FPS came back.
  The camera returns to dynamic frame rate on by itself, so the setting has to be applied
  at every launch.
- **Decision:** added `pathsense.io.camera`. It runs a capture calibration once per camera, keyed by USB ids:
  advertised modes up to 1280 wide, largest first, keeping the first one measured at 90 %
  or more of 30 FPS. Controls are reapplied at every launch. Capture calibration takes 4 s on this
  camera. Never set the buffer size to 1. A background capture thread is not added yet: it
  only helps when the detector is slower than the camera (RF-DETR Nano at 38 ms against
  33 ms per frame), not with YOLO26n at 8 to 20 ms.
