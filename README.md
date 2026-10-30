# PathSense

**Private, in browser people analytics you can question in plain language.**

PathSense detects and tracks people in your webcam or an uploaded security camera clip, turns
their movement into counts, occupancy, dwell time, trails, a heatmap and an event log, and
answers questions like *"how many people came in after 2 pm?"* or *"who waited longest at
the counter?"*. The finished app runs entirely in your browser: **no video is uploaded**.

> Status: **Phase 3, evaluation and scene tuning.** Phases 0 to 2 done. See the [roadmap](#roadmap) and [docs/PLAN.md](docs/PLAN.md).

<!-- Add once they exist:
[![Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/MuhammadAzeem/pathsense)
[![Model](https://img.shields.io/badge/model-Hugging%20Face-blue)](https://huggingface.co/MuhammadAzeem/pathsense-detector)
-->
![CI](https://github.com/AzeemChaudhry/pathsense/actions/workflows/ci.yml/badge.svg)

## What it does

1. **Detects** people in each frame (YOLO26n or RF-DETR Nano, exported to ONNX).
2. **Tracks** each person with a stable ID across frames (ByteTrack; BoT-SORT and OC-SORT compared).
3. **Analyses** movement: in and out counts over a line, occupancy, dwell per zone, trails, heatmap.
4. **Logs events** (`enter`, `exit`, `zone_enter`, `zone_exit`, `dwell`) with track IDs and times.
5. **Answers questions** about the event log with a small LLM running in the browser. The numbers
   come from deterministic functions over the log; the model only picks the function and words the answer.

## Why it is different

People counting with YOLO is a solved tutorial. PathSense focuses on what is not common yet:
processing that never leaves the device (so you can try it on real cameras safely), and
plain language questions over tracked events. The roadmap is in [docs/PLAN.md](docs/PLAN.md).

## Architecture

```
Browser (Static Hugging Face Space, nothing leaves the device)
  webcam / video file
    -> detector ONNX   (onnxruntime-web; WebGPU, WASM fallback)
    -> tracker         (ByteTrack in TypeScript, parity tested against Python)
    -> analytics       (line counts, zones, dwell, heatmap) -> event log
    -> Q&A             (query tools over the event log + small in browser LLM)

Python (this repo)
  research pipeline -> evaluation (TrackEval) -> training on free cloud GPUs (Kaggle)
  -> ONNX export -> Hugging Face model repo
```

## Quickstart

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is optional.

```bash
git clone https://github.com/AzeemChaudhry/pathsense.git
cd pathsense
uv sync --extra dev                 # add --extra rfdetr for the RF-DETR detector
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
uv run pathsense run                # webcam 0, YOLO26n + ByteTrack
```

## Usage

`pathsense run` detects and tracks people in a webcam, a video file or a stream. The window
shows boxes, `#<id>` labels, trails, the loop FPS and the detection time in ms. Press `q` or
`Esc` to stop (Ctrl+C with `--no-show`). A summary line is printed at the end:

```
detector=yolo26n tracker=bytetrack frames=341 seconds=3.6 mean_fps=95.2 detect_ms=8.3 track_ms=0.7 unique_ids=32
```

```bash
uv run pathsense run --source clip.mp4 --save runs/clip.mp4       # record the annotated video
uv run pathsense run --set detector.name=rfdetr-nano              # swap the detector
uv run pathsense run --set tracker.name=botsort                   # swap the tracker
uv run pathsense run --config my.yaml --set detector.device=cpu   # own config file
uv run pathsense run --source clip.mp4 --no-show --max-frames 300 # quick benchmark
uv run pathsense calibrate                                        # re-measure webcam 0
uv run pathsense run --source clip.mp4 --set "analytics.line=[[0,675],[1920,675]]"  # count crossings
uv run pathsense run --source clip.mp4 --set "analytics.line=[[0,675],[1920,675]]" --events runs/clip.jsonl  # and keep the event log
uv run pathsense run --source clip.mp4 --set "analytics.zones=[{name: floor, polygon: [[0,675],[1920,675],[1920,1080],[0,1080]]}]"  # who is in an area, and for how long
uv run pathsense run --source clip.mp4 --heatmap runs/heat.png      # where people spent time
uv run pathsense run --scene configs/scenes/subway.yaml --events runs/subway.jsonl  # one camera's setup
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source` | `0` | camera index, video file or stream URL |
| `--scene` | none | one camera's lines, zones and settings, laid over the config (`configs/scenes/*.yaml`); its `source` is used when `--source` is not given |
| `--config` | `configs/default.yaml` | YAML config |
| `--set SECTION.KEY=VALUE` | none | override one config value; repeatable |
| `--save` | off | write the annotated video to an mp4 that plays at real speed |
| `--events` | off | write the event log (one JSON line per event) to this file |
| `--heatmap` | off | save a heatmap of where people spent time to this `.png` |
| `--no-show` | off | run without a window (for example over SSH or for benchmarks) |
| `--max-frames` | all | stop after this many frames |
| `--trail` | `30` | trail length in frames |
| `--recalibrate` | off | measure the webcam again before running |

### Configuration

Everything that changes results lives in [configs/default.yaml](configs/default.yaml), so
swapping a model is one line, never a code change. Typos are errors, not silent defaults.

| Key | Default | Options |
| --- | --- | --- |
| `detector.name` | `yolo26n` | `yolo26n`, `yolo26s`, `rfdetr-nano`, `rfdetr-small` |
| `detector.weights` | model's own | path or name of other weights (fine tuned models in Phase 4) |
| `detector.imgsz` | model default | YOLO 640, RF-DETR Nano 384 |
| `detector.conf` | `0.1` | kept low so ByteTrack can use low confidence boxes |
| `detector.device` | `auto` | `auto`, `cpu`, `cuda`, `0` |
| `detector.drop_partial` | `false` | drop a box that is a partial view of a person also boxed whole (upper body behind a cart); for eye level cameras, not crowds seen from above |
| `detector.slice` | `null` | detect on overlapping tiles of this many pixels as well as the whole frame: finds many small people in large frames (a square seen from above), several times slower |
| `tracker.name` | `bytetrack` | `sort`, `bytetrack`, `botsort`, `ocsort`, `cbiou` |
| `tracker.params` | `{track_activation_threshold: 0.6}` | passed to the tracker, e.g. `{lost_track_buffer: 60}`; a setting the chosen tracker does not take is skipped with a warning |
| `tracker.stitch_seconds` | `2` | a person the tracker lost for up to this long gets their old ID back when they reappear where they were heading; `0`: off |
| `tracker.stitch_distance` | `1` | how far from where they were heading, in box heights |
| `tracker.stitch_size` | `0.5` | how different their box height may be (from 1/1.5 to 1.5 times) |
| `analytics.line` | `null` | counting line `[[x1, y1], [x2, y2], ...]` in pixels, two or more points (more follow a curved edge); crossing from left to right of start to end is IN |
| `analytics.line_margin` | `5` | feet closer than this many pixels to the line do not change side |
| `analytics.line_margin_height` | `0` | if above 0, the margin is at least this share of the person's height (0.15: 100 px for someone 680 px tall) |
| `analytics.line_confirm` | `0.5` | seconds a person must stay on the new side before a crossing counts |
| `analytics.line_bounce` | `12` | if above 0, a crossing that reverses the ID's last one waits this many seconds and is dropped if the ID crosses back: a tracker ID swap in a crowd ([tracker study](docs/TRACKER_STUDY.md)). Live IN and OUT totals show such a crossing up to this long (plus `line_confirm`) late; someone who crosses, crosses back and crosses again within it loses the crossing back |
| `analytics.zones` | `[]` | areas, `[{name: door, polygon: [[x1, y1], [x2, y2], [x3, y3], ...]}]`: enter, exit and dwell events, and how many are inside |
| `analytics.zone_confirm` | `0.5` | seconds inside (or outside) before a zone enter (or exit) counts |
| `analytics.zone_count_present` | `true` | people already inside when the video starts count as entered; `false`: inside, but not counted as entering |
| `analytics.hidden_legs` | `0.25` | a box this much shorter than the person's full height (a cart or another person hides the legs) gets its feet placed from the head; `0`: off |
| `camera.calibrate` | `true` | capture calibration: measure a new webcam once and reuse the result |
| `camera.max_width` | `1280` | largest picture to try |
| `camera.fps` | `30` | target frame rate |
| `camera.dynamic_framerate` | `false` | `false`: steady FPS in dim light; `null`: leave the camera's setting |
| `camera.store` | `null` | capture calibration file; `null` is `~/.config/pathsense/cameras.json` |

For a fixed camera, run BoT-SORT with `tracker.params={enable_cmc: false}`: its camera motion
compensation costs 10 to 17 ms per 1080p frame and only helps when the camera moves.

### Using it from Python

```python
from pathsense.config import load_config
from pathsense.io import VideoSource
from pathsense.pipeline import Pipeline

config = load_config(overrides=["detector.name=rfdetr-nano"])
with VideoSource("clip.mp4") as source:
    pipeline = Pipeline.from_config(config, frame_rate=source.fps)
    for result in pipeline.run(source):
        print(result.frame.timestamp, result.tracked.tracker_id)  # supervision.Detections
```

### Webcam notes

On a webcam the loop FPS is usually capped by the camera, so `detect_ms` is the number that
measures the model.

The first run with a new webcam runs a **capture calibration**, which takes about 4 seconds:

```
camera: AUKEY PC-W1: calibrating, a few seconds...
  MJPG 1280x960: 30.5 FPS
camera: AUKEY PC-W1: MJPG 1280x960 at 30.5 FPS, saved to ~/.config/pathsense/cameras.json
```

It lists the modes the camera advertises, then measures them from the largest picture down
and keeps the first that really delivers the target FPS. Webcams default to raw 640x480,
and raw frames at higher sizes do not fit through USB at full speed, so the winner is
usually compressed MJPG. The test webcam goes from 640x480 to 1280x960 at the same 30 FPS.
The capture calibration also turns off `exposure_dynamic_framerate`, which otherwise drops many webcams
to 20 FPS in dim light, at the cost of a darker picture.

Later runs recognise the camera by its USB vendor, product and serial number, reapply the
saved settings (cameras forget their controls on replug), and skip the measuring. Changing
a `camera.*` value recalibrates. Two identical webcams without a USB serial number share one
entry, which is harmless because they support the same modes. Without `v4l2-ctl` (macOS, Windows, or Linux without
`v4l-utils`) a few common modes are tried and the camera is recognised by its index only.

Capture calibration picks how the camera delivers frames (format, size, frame rate). It
is not geometric calibration (lens distortion, focal length, camera position), which
PathSense does not need while it measures in pixels.

Line counting is made for a camera that sees people from a few metres away. Sitting at arm's
length, the box covers most of the frame and widens when an arm stretches out, so its
bottom centre jumps across a line and waving looks like crossing. A crossing only counts
once the person stays on the new side for `analytics.line_confirm` seconds (0.5 by default),
which removes most of this. For close-up use raise it to 1 or 1.5
(`--set analytics.line_confirm=1.5`); it never removes all of it, and the IN and OUT totals
update that much later.

### Lines, zones and the event log

Place lines and zones from what the camera is for, not at a convenient spot: a subway
camera counts boarding with a line along the train's doors, an aisle camera measures
browsing with a zone over the aisle floor. Each person is reduced to their feet, so draw on
the floor. A wide lens bends straight edges near the corners; give the line more points to
follow them. Small, far people in a 4K or portrait video may need a larger
`detector.imgsz` to be detected at all.

Getting people detected comes first: no line or zone counts someone the detector misses.
In a 4K or portrait frame people shrink 3 to 6 times at the default input size; set
`detector.imgsz: 1920` (or `detector.slice` for many small people seen from above). Before
tracking, `detector.drop_partial` can drop a box that is a partial view of a person also
boxed whole (upper body behind a shopping cart); leave it off for crowds seen from above,
where it also drops people standing behind others. After tracking, a person lost for up to `tracker.stitch_seconds`
keeps their ID when they reappear, using position, motion and box size only.

`configs/scenes/` has one file per public sample clip, each starting with the scene's use
case and what IN and OUT mean there: `people-walking` (concourse footfall), `subway`
(boarding and getting off along the train's doors, waiting time on the platform),
`grocery-store` (shoppers in an aisle and how long they browse) and `market-square`
(people entering the square from the street, who gathers at the fountain). A scene changes
only the keys it names; everything else comes from the config.

With `--events`, each event is one JSON line:

```json
{"t": 6.64, "frame": 166, "track_id": 21, "type": "enter"}
{"t": 7.08, "frame": 177, "track_id": 30, "type": "zone_enter", "zone": "floor"}
{"t": 9.52, "frame": 238, "track_id": 30, "type": "zone_exit", "zone": "floor"}
{"t": 9.52, "frame": 238, "track_id": 30, "type": "dwell", "zone": "floor", "duration": 2.44}
```

`enter` and `exit` are line crossings. `dwell` comes with every zone exit, and at the end of
the video for people still inside (their time so far, with no exit). `load_events` and
`count_events` in `pathsense.analytics` read a log back.

Each track ID gets its own colour, so an ID switch shows up as a colour change. The tracker
gets real capture timestamps, so it forgets a lost person after the same time whatever the
frame rate. `scripts/webcam_baseline.py` is the single file Phase 0 version, kept as a
commented walkthrough of the same loop. Measured numbers and failure cases are in
[NOTES.md](NOTES.md).

### Scoring counts against ground truth

MOT17 and MOT20 label every person in every frame by hand. `scripts/mot_counts.py` feeds
those tracks through the same lines and zones as the pipeline, so each scene in
`configs/scenes/mot/` is scored without hand counting:

```bash
# MOT16.zip, MOT17Labels.zip and MOT20.zip from https://motchallenge.net/data/ in data/mot/
uv run python scripts/mot_counts.py prepare
uv run python scripts/mot_counts.py score configs/scenes/mot/*.yaml
uv run python scripts/mot_counts.py score configs/scenes/mot/MOT17-02.yaml --set detector.imgsz=1280
```

For each line and zone it prints the enters, exits, total dwell and mean occupancy under:
- `truth`: every labelled person, hidden ones included;
- `seen`: only people at least a quarter visible;
- `ours`: the pipeline.

Truth minus seen is what the scene hides; seen minus ours is the pipeline's error. MOT data
is CC BY-NC-SA: evaluation only, never redistributed, no frames in these docs.

### Comparing trackers

`scripts/tracker_study` runs the detector once per MOT scene, then replays those same
boxes through every tracker and every layer of ours, so only the tracking differs. The
study is defined in `configs/study/trackers.yaml`; the results are in
[docs/TRACKER_STUDY.md](docs/TRACKER_STUDY.md).

```bash
uv run python scripts/tracker_study cache   # detector once per scene into data/cache (GPU, about 10 min)
uv run python scripts/tracker_study all     # sweep, run, render, report, check (CPU, 2 to 3 h)
uv run python scripts/tracker_study check   # where the misses come from; matcher rescoring (2 min)
```

Each run is saved in `runs/tracker-study/runs/<setup>/<scene>/`: the tracks (MOT format),
the events, every crossing marked right, wrong or missed, the metrics and a video with
those marks drawn on it.

### Tuning a scene

Every scene has its own problems (crowds, small people, carts hiding legs), and the
tracker study showed that no one setup suits them all. `pathsense tune` chooses a scene's
tracker settings against what really happened there:

```bash
uv run pathsense tune configs/scenes/mot/MOT20-03.yaml   # report in runs/tune/MOT20-03/
```

It runs the detector over the scene's video once (saved in `data/cache/`, made again if
the video, the weights file or a detector setting changes), then replays those boxes
through the scene as it is and through each setup in
[configs/tune/candidates.yaml](configs/tune/candidates.yaml), and scores them: by crossing
F1, then by count error. A setup replaces the scene's own only if it comes out strictly
ahead in that order. Candidates may change the tracker and the counting rules, not the
detector or the line and zones. The report describes the scene, compares every setup,
shows the block to paste into the scene file, and puts every missed crossing in one group:
the detector's (never detected), the tracker's (detected, but the track was lost or
split), a swap (an ID crossed but on someone else), both (detected only at times), or
unclear; it also says what the counting rules alone get wrong with perfect tracks. It
never edits the scene file itself.

The saved boxes are made again when the video file changes or moves (moving the repo
counts), or when a local weights file changes; pretrained weights the model library
downloads are known by name only. `--refresh` makes them again in any case.

The scene file says what really happened in a `truth:` block, either hand labels or hand
counts:

```yaml
truth:
  labels: data/mot/gt/MOT20-03.txt   # every person boxed in every frame (MOT format)
# or
truth:
  enter: 11                          # hand counts: crossings each way,
  exit: 7
  zones: {aisle: 1}                  # and people entering each zone
```

Labels score every crossing one by one; hand counts only score totals, where a false
crossing and a missed one cancel out. A count left out of `truth:` is not scored, and the
report marks it so (grocery-store gives no exits, for example).

### Checks

```bash
uv run pre-commit run --all-files
uv run pytest                                    # fast, no GPU or downloads needed
PATHSENSE_SLOW=1 uv run pytest -m slow           # real YOLO26n and RF-DETR on the sample clip
```

## Repository layout

```
pathsense/
  src/pathsense/
    config.py     YAML config and section.key=value overrides
    detect/       Detector protocol, registry and adapters (YOLO26, RF-DETR)
    track/        Tracker wrapper (SORT, ByteTrack, BoT-SORT, OC-SORT, C-BIoU), stitcher
    io/           Frame sources with timestamps, real time video writer, window set up
    pipeline.py   Detector -> tracker -> stitcher -> counters, one frame at a time
    annotate.py   Boxes, ID labels, trails and the stats line
    cli.py        The `pathsense` command
    analytics/    Line counts, zones, dwell time, heatmap (Phase 2)
    eval/         Scoring against hand labels: tracking metrics, crossings one by one
    tune/         Saved detections, replays, scene profiles and `pathsense tune`
  web/            Browser app for the Static Space (Phase 5 and 6)
  app/            Placeholder for an optional Gradio fallback demo (not built yet)
  configs/        YAML configs: default, scenes, tune candidates, tracker study
  train/          Data download and fine tuning
  scripts/        Phase 0 webcam baseline; MOT count scoring; tracker study
  tests/          pytest suite
  NOTES.md        Experiment log: what was tried, the number, the decision
```

## Benchmarks

Filled in during Phase 3 and Phase 4.

| Detector | Tracker | Data | HOTA | IDF1 | MOTA | FPS (GTX 1650) | FPS (CPU) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26n (COCO) | ByteTrack | MOT17 val | | | | | |
| YOLO26n (fine tuned) | ByteTrack | held out clips | | | | | |
| RF-DETR Nano (fine tuned) | ByteTrack | held out clips | | | | | |

## Roadmap

- [x] **Phase 0** Webcam baseline (YOLO26n + ByteTrack)
- [x] **Phase 1** Pipeline package (detector interface, tracker wrapper, tests)
- [x] **Phase 2** Analytics and event log
- [ ] **Phase 3** Evaluation and scene tuning (tracker comparison on crowds, `pathsense tune`, recovering missed crossings)
- [ ] **Phase 4** Train on Kaggle (Open Images), export ONNX, weights on the Hub
- [ ] **Phase 5** Browser app: in browser detection, tracking and analytics
- [ ] **Phase 6** Ask your footage: plain language Q&A over the event log
- [ ] **Phase 7** Hardening and showcase

## Data

Deployable models train on Open Images V7 (CC BY 4.0 annotations); CrowdHuman is used only
for benchmarks. Research and license notes: [docs/DATASETS.md](docs/DATASETS.md).
Download with `train/get_data.py` ([train/README.md](train/README.md)); train on free cloud
GPUs as described in [docs/COMPUTE.md](docs/COMPUTE.md).

## Design decisions

Plan: [docs/PLAN.md](docs/PLAN.md). Before and after each phase, with images and numbers:
[docs/PROGRESS.md](docs/PROGRESS.md). Experiments are logged in [NOTES.md](NOTES.md).
Libraries and methods for the browser app and appearance tracking:
[docs/RESEARCH.md](docs/RESEARCH.md).

## Development

Branches, signed commits, CI, `@claude` reviews, versioning and releases are described in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Changes per version: [CHANGELOG.md](CHANGELOG.md).

## Privacy and limitations

- PathSense counts and tracks anonymous people. It does **not** identify who anyone is: no face recognition, demographics or emotion analysis.
- The browser app processes video on your device; nothing is uploaded.
- An optional, off by default research setting compares clothing vectors to follow people. On a video they stay in memory for that video and are never tied to a name; only the tuning and study tools save them to disk, for research clips ([details](docs/TRACKER_STUDY.md#appearance-step-5-does-telling-people-apart-by-looks-recover-crossings)).
- Only footage recorded with the consent of the people in it is used for training or demos.
- Public camera feeds are out of scope.

## License

AGPL 3.0, because the project uses Ultralytics YOLO, which is AGPL 3.0. See [LICENSE](LICENSE).
