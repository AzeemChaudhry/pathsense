# PathSense

**Private, in browser people analytics you can question in plain language.**

PathSense detects and tracks people in your webcam or an uploaded security camera clip, turns
their movement into counts, occupancy, dwell time, trails, a heatmap and an event log, and
answers questions like *"how many people came in after 2 pm?"* or *"who waited longest at
the counter?"*. The finished app runs entirely in your browser: **no video is uploaded**.

> Status: **Phase 2, analytics and event log.** Phases 0 and 1 done. See the [roadmap](#roadmap) and [docs/PLAN.md](docs/PLAN.md).

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
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source` | `0` | camera index, video file or stream URL |
| `--config` | `configs/default.yaml` | YAML config |
| `--set SECTION.KEY=VALUE` | none | override one config value; repeatable |
| `--save` | off | write the annotated video to an mp4 that plays at real speed |
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
| `tracker.name` | `bytetrack` | `bytetrack`, `botsort`, `ocsort` |
| `tracker.params` | `{}` | passed to the tracker, e.g. `{lost_track_buffer: 60}` |
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
a `camera.*` value recalibrates. Without `v4l2-ctl` (macOS, Windows, or Linux without
`v4l-utils`) a few common modes are tried and the camera is recognised by its index only.

Capture calibration picks how the camera delivers frames (format, size, frame rate). It
is not geometric calibration (lens distortion, focal length, camera position), which
PathSense does not need while it measures in pixels.

Each track ID gets its own colour, so an ID switch shows up as a colour change. The tracker
gets real capture timestamps, so it forgets a lost person after the same time whatever the
frame rate. `scripts/webcam_baseline.py` is the single file Phase 0 version, kept as a
commented walkthrough of the same loop. Measured numbers and failure cases are in
[NOTES.md](NOTES.md).

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
    track/        Tracker wrapper (ByteTrack, BoT-SORT, OC-SORT)
    io/           Frame sources with timestamps, real time video writer, window set up
    pipeline.py   Detector -> tracker, one frame at a time, with timings
    annotate.py   Boxes, ID labels, trails and the stats line
    cli.py        The `pathsense` command
    analytics/    Line counts, zones, dwell time, heatmap (Phase 2)
  web/            Browser app for the Static Space (Phase 5 and 6)
  app/            Placeholder for an optional Gradio fallback demo (not built yet)
  configs/        YAML configs (model, tracker, zones)
  eval/           TrackEval scripts and MOT format export
  train/          Data download and fine tuning
  scripts/        Phase 0 webcam baseline
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
- [ ] **Phase 2** Analytics and event log
- [ ] **Phase 3** Evaluation harness (TrackEval on MOT17, tracker comparison)
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

## Development

Branches, signed commits, CI, `@claude` reviews, versioning and releases are described in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Changes per version: [CHANGELOG.md](CHANGELOG.md).

## Privacy and limitations

- PathSense counts and tracks anonymous people. It does **not** identify who anyone is: no face recognition, demographics or emotion analysis.
- The browser app processes video on your device; nothing is uploaded.
- Only footage recorded with the consent of the people in it is used for training or demos.
- Public camera feeds are out of scope.

## License

AGPL 3.0, because the project uses Ultralytics YOLO, which is AGPL 3.0. See [LICENSE](LICENSE).
