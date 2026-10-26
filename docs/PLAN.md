# Project plan

Direction set 2026-10-27, after the market research in `docs/MARKET.md`.

## Pitch

**Private, in browser people analytics you can question in plain language.**

PathSense detects and tracks people in a webcam stream or an uploaded security camera clip,
turns movement into analytics (line counts, occupancy, dwell time, trails, heatmap) and an
event log, and lets you ask questions about it ("how many people entered between 2 and 3
pm?", "who waited longest at the counter?"). Everything runs **inside the visitor's
browser**: no video is uploaded, so anyone can safely test it on their own cameras.

## Why this direction

Plain people counting is commoditized: Ultralytics ships counting, heatmaps and queues as
built in "solutions", several Hugging Face Spaces already do it, and Upwork sells it as a
package. Three things are not common, and PathSense combines them:

1. **Private by design (A):** in browser inference with WebGPU, WASM fallback. Matches the
   EU AI Act push toward anonymous, on device processing (transparency rules from Aug 2026).
2. **Ask your footage (B):** a tracker built event log queried by a small LLM, the same
   hybrid pattern as NVIDIA's video search and summarization blueprint.
3. **Measured model work (C, optional):** fine tuning, and later distilling a large teacher
   (SAM 3 or RF-DETR large) into a nano model, on free cloud GPUs.

## Architecture

```
Browser (Static Hugging Face Space, nothing leaves the device)
  webcam / video file
    -> detector ONNX (onnxruntime-web or transformers.js; WebGPU, WASM fallback)
    -> tracker (ByteTrack ported to TypeScript, parity tested against Python)
    -> analytics (line counts, zones, dwell, heatmap) -> event log (JSON)
    -> Q&A: tools compute the numbers from the event log; a small LLM in the browser
            (WebLLM or transformers.js, Qwen3 0.6B to 2B) only chooses tools and phrases answers

Python (this repo)
  research pipeline, evaluation (TrackEval), training on free cloud GPUs,
  ONNX export, weights pushed to a Hugging Face model repo
```

Design rule for Q&A: **the LLM never does arithmetic.** Counts, durations and times come
from deterministic functions over the event log; the model selects the function and
writes the sentence. This is what keeps answers correct with a 1B model.

## Constraints

- Local GPU: GTX 1650, 4 GB, Ubuntu. Use it for inference and quick experiments only.
- Training runs on free cloud GPUs (see `docs/COMPUTE.md`): Kaggle first, Colab second.
- Hosting: Static Spaces are free for everyone and have no GPU quota. A ZeroGPU Gradio
  Space (free accounts: up to 2, account over 30 days old) is only a fallback demo.
- WebGPU support is uneven (Chrome on Linux needs recent Intel GPUs; Firefox on Linux has
  none). The browser app must fall back to WASM on CPU and say which backend it is using.

## Model and license tracks

| Track | Detector | Data | Use |
| --- | --- | --- | --- |
| Portfolio | YOLO26n (AGPL) or RF-DETR Nano | Open Images, CrowdHuman for benchmarks | This public repo and demo |
| Commercial | RF-DETR Nano (Apache 2.0) | Open Images (+ client footage) | Paid client work |

The browser demo should ship the RF-DETR Nano (Apache 2.0) model if its accuracy and speed
are close, so the same artifact works on both tracks.

## Phases and gates

Each phase has one gate. Do not start the next phase until the gate passes.

| Phase | Work | Gate |
| --- | --- | --- |
| 0 | **Webcam baseline** in Python: YOLO26n + ByteTrack, IDs and trails on screen | Stable ID on a 15 s recording; FPS and two failures logged in NOTES.md |
| 1 | Pipeline package: `Detector` protocol, YOLO26 and RF-DETR adapters, tracker wrapper, frame sources, tests | Swap detector or tracker with one config line |
| 2 | Analytics + **event log**: line crossings, zones, dwell, heatmap; events as JSON lines (`enter`, `exit`, `zone_enter`, `zone_exit`, `dwell`) with track ID and timestamp | Counts match a hand count on one clip; event log reproduces them |
| 3 | Evaluation: TrackEval on MOT17 val half; compare ByteTrack, BoT-SORT, OC-SORT; FPS on GTX 1650 and CPU | Baseline HOTA, IDF1, MOTA in README table |
| 4 | **Train on Kaggle**: fine tune YOLO26n and RF-DETR Nano on Open Images people; export ONNX; push weights + model card to the Hub | Fine tuned model beats COCO baseline on held out data; ONNX runs in onnxruntime (Python) |
| 5 | **Browser app (A)**: Static Space; ONNX detector in the browser; TypeScript ByteTrack; line counter, heatmap, event log; webcam and clip upload; WebGPU with WASM fallback | Same detections in Python and browser give the same track IDs and counts on a test clip |
| 6 | **Ask your footage (B)**: query tools over the event log; small in browser LLM picks tools and answers; optional "bring your own API key" mode | 20 question test set answered with correct numbers |
| 7 | Hardening and showcase: CI deploy of the Space on tag, README GIF, model card, write up, resume bullets | Linked from resume and website |

**Stretch (C):** on a cloud GPU, auto label real security camera frames from MEVA
(CC BY 4.0) with SAM 3 or a large RF-DETR, train the nano model on them, publish before and
after numbers on CCTV style video.

## Current status

- Repo scaffolded, CI green.
- `train/get_data.py` downloads Open Images (deployable) and CrowdHuman (benchmarks).
- **Phase 0 gate passed (2026-10-27).** `scripts/webcam_baseline.py`: ID held for 76 s;
  20.5 FPS camera bound (20 ms detect + track on the GTX 1650). Failures logged in
  NOTES.md: re-entry gets a new ID, and partly visible people are fragmented or missed.
- **Next task: Phase 1 pipeline package** (`Detector` adapters, tracker wrapper, frame
  sources, config driven swap).
