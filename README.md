# PathSense

**Private, in browser people analytics you can question in plain language.**

PathSense detects and tracks people in your webcam or an uploaded security camera clip, turns
their movement into counts, occupancy, dwell time, trails, a heatmap and an event log, and
answers questions like *"how many people came in after 2 pm?"* or *"who waited longest at
the counter?"*. The finished app runs entirely in your browser: **no video is uploaded**.

> Status: **Phase 1, pipeline package.** Phase 0 baseline done. See the [roadmap](#roadmap) and [docs/PLAN.md](docs/PLAN.md).

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
plain language questions over tracked events. Research behind this: [docs/MARKET.md](docs/MARKET.md).

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
uv sync --extra dev
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
uv run python scripts/webcam_baseline.py
```

### Webcam baseline (Phase 0)

YOLO26n detects people and ByteTrack gives each one a stable ID. The window shows
boxes, `#<id>` labels, trails, the loop FPS and the detect plus track time in ms. Press `q`
or `Esc` to quit (or Ctrl+C with `--no-show`); a summary line is printed on exit:

```
frames=370 seconds=18.9 mean_fps=19.5 infer_ms=20.0 unique_ids=1
```

On a webcam the loop FPS is usually capped by the camera, so `infer_ms` is the number that
measures the model. Many webcams lower their frame rate in dim light to expose longer (the
test laptop drops from 30 to 20 FPS). On Linux you can check and turn this off:

```bash
v4l2-ctl -d /dev/video0 --get-ctrl=exposure_dynamic_framerate
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0   # fixed 30 FPS, darker image
```

Each track ID gets its own colour, so an ID switch shows up as a colour change. The tracker
gets real capture timestamps, and `--save` repeats frames as needed so the recording plays
back at real speed. Measured numbers and failure cases are in [NOTES.md](NOTES.md).

```bash
uv run python scripts/webcam_baseline.py                          # webcam 0
uv run python scripts/webcam_baseline.py --save runs/phase0.mp4   # also record the output
uv run python scripts/webcam_baseline.py --source clip.mp4 --device cpu
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source` | `0` | camera index, video file or stream URL |
| `--weights` | `yolo26n.pt` | Ultralytics weights; downloaded on first run |
| `--imgsz` | `640` | inference size |
| `--conf` | `0.1` | detector threshold; kept low so ByteTrack can use low confidence boxes |
| `--device` | auto | `cpu` or `0` for the first GPU |
| `--trail` | `30` | trail length in frames |
| `--save` | off | write the annotated video to an mp4 |
| `--no-show` | off | run without a window (for example over SSH) |

Run the checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Repository layout

```
pathsense/
  src/pathsense/
    detect/       Detector interface and model adapters (YOLO26, RF-DETR)
    track/        Tracker wrapper (ByteTrack, BoT-SORT, OC-SORT)
    analytics/    Line counts, zones, dwell time, heatmap
    io/           Video, webcam and RTSP readers and writers
    pipeline.py   Glues the stages together, one frame at a time
  web/            Browser app for the Static Space (Phase 5 and 6)
  app/            Gradio fallback demo for browsers without WebGPU or WASM speed
  configs/        YAML configs (model, tracker, zones)
  eval/           TrackEval scripts and MOT format export
  train/          Fine tuning configs and scripts
  scripts/        One off scripts (webcam baseline, benchmarks)
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
- [ ] **Phase 1** Pipeline package (detector interface, tracker wrapper, tests)
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

Plan: [docs/PLAN.md](docs/PLAN.md). Experiments are logged in [NOTES.md](NOTES.md).

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
