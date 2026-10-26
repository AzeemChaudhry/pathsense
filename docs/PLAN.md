# Project plan (condensed)

Full plan with research and diagrams: the "People Tracking CV App: Project Plan" doc on claude.ai.

## Goal

Detect and track people in video, turn movement into analytics (line counts, occupancy,
dwell time, trails, heatmap), deploy publicly on Hugging Face so anyone can test it with a
webcam or their own camera footage, and show a measured improvement from fine tuning.

## Constraints

- Local GPU: GTX 1650, 4 GB VRAM, 16 GB RAM, Ubuntu. Nano and small models, 640 input, AMP.
- Hosting starts free: Hugging Face ZeroGPU Gradio Space (free accounts: up to 2, account
  must be over 30 days old) and a Static Space for in browser inference.

## Stack

YOLO26n (AGPL) and RF-DETR Nano (Apache 2.0) detectors; `trackers` (ByteTrack, BoT-SORT,
OC-SORT); `supervision` for zones, counts, heatmaps; ONNX Runtime; Gradio 6 + FastRTC;
TrackEval for HOTA / IDF1 / MOTA; Hugging Face Hub for weights and data; GitHub Actions CI.

## Phases and gates

| Phase | Work | Gate before moving on |
| --- | --- | --- |
| 0 | Webcam baseline: YOLO26n + ByteTrack, IDs and trails | Stable ID on screen, FPS logged in NOTES.md |
| 1 | Pipeline package: Detector protocol, adapters, tracker wrapper, tests | Swap detector with one config line |
| 2 | Analytics: line counts, zones, dwell, heatmap, CSV | Counts match a hand count on a clip |
| 3 | Evaluation: TrackEval on MOT17 val, compare 3 trackers | Baseline HOTA, IDF1, MOTA logged |
| 4 | Fine tune on Open Images (docs/DATASETS.md); YOLO26n and RF-DETR | Beats baseline on held out data |
| 5 | Gradio app on ZeroGPU, weights on the Hub | Public Space link works |
| 6 | Hardening: CI deploy on tag, Docker, input validation | A git tag redeploys the Space |
| 7 | Showcase: README GIF, model card, write up | Linked from resume and website |

After v1: live webcam demo in the browser (WebGPU, Static Space), walking speed via
homography, pose based alerts, MEVA auto labelling for security camera views.

## Current status

- Repo scaffolded, CI green, data download script ready (Open Images, CrowdHuman, Roboflow).
- **Next task: Phase 0 webcam baseline** in `scripts/webcam_baseline.py`.
