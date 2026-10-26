# PathSense

Real time people detection, tracking and movement analytics: line crossing counts, live occupancy, zone dwell time, trajectories and heatmaps, from a webcam or a video file.

> Status: **Phase 0, setup and webcam baseline.** See the [roadmap](#roadmap).

<!-- Add once they exist:
[![Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/<user>/pathsense)
[![Model](https://img.shields.io/badge/model-Hugging%20Face-blue)](https://huggingface.co/<user>/pathsense-detector)
![CI](https://github.com/AzeemChaudhry/pathsense/actions/workflows/ci.yml/badge.svg)
-->

## What it does

1. **Detects** people in each frame (YOLO26n by default, RF-DETR as a challenger).
2. **Tracks** each person with a stable ID across frames (ByteTrack first, then BoT-SORT and OC-SORT).
3. **Analyses** movement: in and out counts over a line, occupancy, dwell time per zone, trails and a heatmap.
4. **Exports** an annotated video, an events CSV and charts.

## Architecture

```
Input (webcam / video / RTSP)
   -> Detector   (YOLO26n | RF-DETR, ONNX Runtime, person class only)
   -> Tracker    (ByteTrack | BoT-SORT | OC-SORT, stable IDs)
   -> Analytics  (LineZone, PolygonZone, dwell, heatmap, trails)
   -> Outputs    (annotated video, events CSV, MOT file for TrackEval)
```

Every detector returns the same `supervision.Detections` object, so swapping models is one config line, not a rewrite.

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is optional.

```bash
git clone https://github.com/AzeemChaudhry/pathsense.git
cd pathsense
uv sync --extra dev
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
uv run python scripts/webcam_baseline.py
```

Run the checks:

```bash
uv run ruff check .
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
  app/            Gradio app for the Hugging Face Space
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

- [ ] **Phase 0** Setup and webcam baseline
- [ ] **Phase 1** Pipeline package (detector interface, tracker wrapper, tests)
- [ ] **Phase 2** Analytics (counts, zones, dwell, heatmap, CSV)
- [ ] **Phase 3** Evaluation harness (TrackEval on MOT17, tracker comparison)
- [ ] **Phase 4** Fine tuning on a named failure case, before and after benchmark
- [ ] **Phase 5** Gradio app on a ZeroGPU Space, weights on the Hub
- [ ] **Phase 6** Production hardening (CI deploy, Docker, validation)
- [ ] **Phase 7** Showcase (GIF, model card, write up)

## Design decisions

Logged in [NOTES.md](NOTES.md) as they are made.

## Privacy and limitations

- PathSense counts and tracks anonymous people. It does **not** identify who anyone is and does no face recognition.
- Only footage recorded with the consent of the people in it is used for training or demos.
- Public camera feeds are out of scope.

## License

AGPL 3.0, because the project uses Ultralytics YOLO, which is AGPL 3.0. See [LICENSE](LICENSE).
