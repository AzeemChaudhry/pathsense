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

## Phase 2: analytics and event log

### 2026-10-27: line counter against a hand count
- **Tried:** `LineCounter` on `people-walking.mp4` (341 frames, 25 FPS) with a horizontal
  line at y = 675 across the full 1920 width; walking down the image is `enter`, up is
  `exit`. Picked after tracing every track's feet: almost everyone walks up or down, and
  tracks in the top third are sparse (small, far people rarely reach 0.7 confidence). The
  hand count was done on a copy of the clip with only the line and a timer drawn, then
  checked against a review video with each counted crossing labelled.
- **Result:**

  | | Hand count | Counter, activation 0.7 | Counter, activation 0.6 |
  | --- | --- | --- | --- |
  | enter | 14 (11 crossing the line) | 10 | **11** |
  | exit | 7 | 7 | **7** |
  | false counts | | 0 | 0 |

  Every track seen on both sides of the line was counted, in the right direction, once.
  The 4 missing enters were tracks that started below the line:
  1. Three people walked in from the left edge below the line (tracks start at x = 37
     to 42). They never cross the line; they enter the area. That is a zone question,
     not a line question.
  2. One woman was detected at 0.61 to 0.62 confidence while crossing, below ByteTrack's
     0.7 `track_activation_threshold`, so her track started after the line. At 0.6
     exactly this one enter is added (t = 6.64 s) and nothing else changes; 0.5 adds
     nothing more. Unique IDs on the clip go from 32 to 39.
- **Decision:**
  1. A line counts crossings only. Entering an area from any side is a zone event
     (`zone_enter`), the next Phase 2 piece.
  2. `configs/default.yaml` sets `track_activation_threshold: 0.6`. OC-SORT has no such
     setting (it starts tracks at `high_conf_det_threshold`, already 0.6), so the tracker
     wrapper skips settings that only other trackers take, with a warning, and still
     rejects names no tracker knows. Phase 3 checks the extra IDs against MOT17 ground
     truth.

### 2026-10-28: confirmation hold against arm waving
- **Tried:** sitting at arm's length from the webcam with a vertical line through the
  middle, waving a hand across it. One track (the person), box about 0.76 of the frame
  wide; when an arm stretches out the box widens and its bottom centre jumps across the
  line. Added a confirmation hold: a flip only counts once the person stays on the new
  side for `line_confirm` seconds, and is cancelled if they come back sooner. Someone who
  vanishes on the new side is counted after the same time, or at the end of the video.
  Then swept the hold on `people-walking.mp4`.
- **Result:**

  | Hold | Webcam waving, false crossings | Sample clip in / out |
  | --- | --- | --- |
  | none | 53 | 11 / 7 |
  | 0.5 s | about 11 (replayed from the first run) | 11 / 7 |
  | 1.0 s | 14 (7 in, 7 out) | 11 / 7 |
  | 1.5 s | 4 (2 in, 2 out) | 11 / 7 |
  | 2.0 and 3.0 s | | 11 / 7 |

  On the clip every real walker stayed on the new side for 1.48 to 11.64 s and nobody
  stepped back; those who left the view sooner were counted by the "vanished" rule.
- **Decision:** default `line_confirm: 0.5`. It removes most waving and jitter, costs
  nothing on real footage, and the totals update only half a second late; 1.0 s already
  felt slow on the webcam. Close-up use is a README note (raise it to 1 to 1.5 s), not the
  default's job: no hold removes waving completely. The clip has nobody hesitating at the
  line, so a busier clip (a doorway where people pause) is still needed to show the hold
  helping on real footage.

### 2026-10-28: three more public clips, and why one line position fits none of them
- **Tried:** `grocery-store.mp4` (4K, 33 s), `subway.mp4` (portrait 4K, 43 s) and
  `market-square.mp4` (portrait 4K, 8 s, 60 FPS), from the same supervision sample assets
  as `people-walking.mp4`. First with a horizontal line mid frame on each, then with a line
  placed from each scene's use case after looking at it over time: subway, boarding and
  getting off along the train's doors; grocery-store, a shopper entering the aisle; market,
  people walking into the square from the street.
- **Result:**
  1. A line not placed for the scene's question gives numbers that mean nothing; all
     three first runs were discarded.
  2. Subway: at `imgsz=640` the 4K portrait frame is shrunk 6 times and only about 10 of
     the 30 or more people at the doors are detected (24 at `imgsz=1280`). Boarding and
     getting off were 2 and 5 with a straight line, 4 and 4 with a polyline, far below
     what the video shows. People leaving the train in a cluster are mostly never
     detected; that is a detector limit for Phases 3 and 4, not something counting fixes.
  3. Subway: the train's lower edge is curved by the wide lens (x 345 at y 3120 where a
     straight line through its top half gives 550), so a two point line ran through the
     waiting crowd. Lines now take more points.
  4. Grocery: one shopper walking past a line once was counted IN, OUT, IN, OUT, IN. His
     cart hides his legs, so his box shrinks (709 px to 429 px at 16.3 s) and the box
     bottom jumps to the cart handle. `analytics.hidden_legs` now places the feet from the
     head when a box is more than 25 % shorter than the person's full height.
  5. Market: at `imgsz=640` almost nobody is detected (8 boxes for about 60 people); at
     1920, 109. The early crossings that looked suspicious were real people walking across
     the border; each counted once in the right direction.
  6. A waiting person shuffling near a line crosses it slowly and is counted each time;
     a 5 px margin is under 1 % of a person 680 px tall. `line_margin_height` is
     available to test during tuning; off by default until hand counts show it helps.
- **Decision:** Tune each scene by placing its lines and zones from the use case (and
  `detector.imgsz` for large frames), not by changing the model. Build all of Phase 2
  first (zones, dwell, occupancy, polylines, hidden legs, heatmap), then tune and hand
  count each clip. `people-walking.mp4` is unchanged: line 11 in, 7 out; a floor zone below
  the line gives 28 enters, 20 exits and 8 people inside at the end.

### 2026-10-28: tuning the four scenes (detection, ID stitching, partial boxes, margins)
- **Tried:** one scene file per clip (`configs/scenes/`), placed from its use case; then,
  per clip, fixing what its trace showed: the detector input size or tiles, a stitcher
  that gives a person lost for up to 2 s their ID back, dropping a partial box of a person
  also boxed whole, and a margin that grows with the person's height.
- **Result:** people found in a single frame (confidence 0.6 or more, enough to start a
  track), and the scene counts:

  | Frame | Default 640 | imgsz 1920 | Tiles 960 at 960 |
  | --- | --- | --- | --- |
  | subway doors, 26 s | 5 | 17 (40 ms) | 14 (265 ms) |
  | market square, 2 s | 0 | 57 (41 ms) | 78 (266 ms) |
  | grocery shopper far away, 0.5 to 3 s | 0.33 to 0.68 | 0.61 to 0.83 | |

  | Scene | First scene run | Now | Settings |
  | --- | --- | --- | --- |
  | people-walking | 11 IN, 7 OUT | 11 IN, 7 OUT (hand count 11, 7) | defaults |
  | grocery-store | 3 IN, 1 OUT; 2 aisle visits | **1 IN**; one 31.7 s aisle visit | imgsz 1920, drop_partial, line_margin_height 0.1 |
  | subway | 2 boarded, 3 got off | 5 boarded, 10 got off; platform 63 in, 61 out | imgsz 1920, 11 point line to the far end |
  | market-square | nothing detected | 1 IN, 20 OUT; fountain 5 people | tiles 960 at 960 |

  1. Tiles hurt the subway: people near the camera are taller than a tile's overlap, so
     every tile cuts them. Tiles suit many small people seen from above.
  2. Grocery: with the cart hiding his legs, the detector boxed the shopper's upper body
     and whole body (same sides and top within 2 px); the tracker followed the second box
     as a phantom person across the line. Dropping such boxes fixed it, but the first rule
     (80 % shared width, tops within 10 %) dropped 23 % of boxes at the subway doors, most
     of them people standing behind others; a tight rule still dropped 4 %, some real. So
     `detector.drop_partial` is per scene, on only for the grocery camera.
  3. Grocery: the shopper browses with the cart at the line from 10.5 to 13.5 s and his
     feet flip across it; a margin of 10 % of his height gives the single correct IN.
  4. Market: of the tracks that cross the kerb, 16 walk down into the street and 1 up, so
     1 IN and 20 OUT follows the tracks; to be checked by eye.
  5. The stitcher changes nothing on people-walking (39 IDs either way).
- **Decision:** settings belong to the scene; Phase 3 builds scene tuning (`pathsense
  tune`) to choose them from a profile of each video, and reference counts per scene to
  score them. Ideas beyond that are in docs/BACKLOG.md. Subway and market counts still
  need a hand count; subway is mostly limited by people leaving the train in a cluster.

## 2026-10-28: half precision, and true counts from MOT

- **Tried:** FP16 inference (`quantize="fp16"`, formerly `half=True`) with yolo26n on the
  GTX 1650, 100 subway frames.
- **Result:** slower, with the same boxes. 1920: 97.7 ms against 40.7 ms per frame; 640:
  16.2 ms against 7.1 ms. Boxes at 0.5 or more: 1167 against 1161.
- **Decision:** stay in FP32 on this GPU. Speed for the product comes from the browser
  export (Phase 5), not from PyTorch settings.
- **Next:** four sample clips are too few to say the counting works. MOT17 (02, 04, 09)
  and MOT20 (01, 02, 03, 05) are fixed cameras with every person labelled by hand;
  `scripts/mot_counts.py` counts those tracks through the same lines and zones for the true
  counts. Evaluation only (CC BY-NC-SA): no frames in public docs.

## 2026-10-28: first true counts (MOT17), and the PR 10 review

- **MOT17, lines and zones placed per scene** (configs/scenes/mot/), truth from the hand
  labelled tracks through the same counters:

  | Scene, imgsz | People truth / ours | Line in, out truth / ours | Zone in, out truth / ours |
  | --- | --- | --- | --- |
  | 02 square, 640 | 74 / 24 | 1, 10 / 2, 9 | tree 48, 21 / 6, 4 |
  | 02 square, 1920 | 74 / 78 | 1, 10 / 0, 10 | tree 48, 21 / 39, 31 |
  | 04 street, 640 | 88 / 24 | 8, 5 / 6, 4 | shop front 34, 16 / 3, 3 |
  | 04 street, 1920 | 88 / 83 | 8, 5 / 8, 4 | shop front 34, 16 / 40, 24 |
  | 09 shop fronts, 640 | 27 / 28 | 1, 7 / 3, 6 | doorway 3, 3 / 5, 4 |

- **Result:** lines are within 2 everywhere. Zones far from the camera fail at 640 (people
  too small), and at 1920 overcount exits: of 31 tree exits, 20 are tracks that vanished
  inside the zone (truth: 12), people lost in the group under the tree for longer than the
  stitcher waits. A stitch window of 8 s only brings exits from 31 to 23.
- **Decision:** imgsz 1920 in the two far scenes. The extra zone exits are a detection limit
  in groups (with the cluster problem in docs/BACKLOG.md), not a zone bug.
- **Review fixes:** stitcher could give two people one ID (order of detections); feet kept
  an old height forever (now it shrinks with the box's width, so walking away is not hidden
  legs); drop_partial depended on box order; zone finish twice repeated dwell.
- **Sample scenes after the fixes:** people-walking and grocery unchanged (correct); subway
  5 to 7 boarded and market 20 to 16 OUT, from the feet change in crowds seen from above.
  Neither has a true count; MOT20 (crowds from above) will show which way it moved.
- **Speed:** headless runs reach 22 FPS on the 4K clips against 10 with the window, so
  drawing and display cost as much as detection.

## 2026-10-28: truth, seen and ours (second PR 10 review)

- **Change:** the scorer gives truth without the hidden legs estimate (it moved MOT17-09's
  true line out from 7 to 10), adds a `seen` truth (boxes at least 25 % visible), and
  scores dwell and mean occupancy, not only enters and exits.
- **Result** (imgsz 1920): MOT17-04 ours matches seen (shop front dwell 450 s against 448,
  13 inside against 13.1, line within 1); the gap to truth (565 s, 16 inside) is people
  hidden behind others. MOT17-02's tree group: truth 26.5 inside, seen 13.7, ours 8.4, so
  half is hidden and we find 60 % of the rest.
- **Decision:** the street scene is solved within what a camera sees. Groups at a distance
  are our real error: detection in clusters, the next target after Phase 2.
