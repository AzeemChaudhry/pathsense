# Roadmap

Direction set 2026-10-27. Progress per phase, with numbers and images: [PROGRESS.md](PROGRESS.md).

## Pitch

**Private, in browser people analytics you can question in plain language.**

PathSense detects and tracks people in a webcam stream or an uploaded security camera clip,
turns movement into analytics (line counts, occupancy, dwell time, trails, heatmap) and an
event log, and lets you ask questions about it ("how many people entered between 2 and 3
pm?", "who waited longest at the counter?"). Everything runs **inside the visitor's
browser**: no video is uploaded, so anyone can safely test it on their own cameras.

## Why this direction

People counting on its own is well covered: Ultralytics ships counting, heatmaps and queue
monitoring as built in "solutions", and several Hugging Face Spaces already demo it. Three
things are not common, and PathSense combines them:

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

- Development GPU: GTX 1650 (4 GB). Used for inference and quick experiments only.
- Training runs on free cloud GPUs (see `docs/COMPUTE.md`): Kaggle first, Colab second.
- Hosting: Static Spaces are free for everyone and have no GPU quota. A ZeroGPU Gradio
  Space (free accounts: up to 2, account over 30 days old) is only a fallback demo.
- WebGPU support is uneven (Chrome on Linux needs recent Intel GPUs; Firefox on Linux has
  none). The browser app must fall back to WASM on CPU and say which backend it is using.

## Models and licenses

| Detector | License | Data | Notes |
| --- | --- | --- | --- |
| YOLO26n | AGPL 3.0 | Open Images; CrowdHuman for benchmarks only | Fastest on the development GPU |
| RF-DETR Nano | Apache 2.0 | Open Images | Permissive license, usable in closed products |

The browser demo should ship RF-DETR Nano if its accuracy and speed are close, so the
deployed model carries the permissive license. Data licenses: [DATASETS.md](DATASETS.md).

## Phases and gates

Each phase has one gate. Do not start the next phase until the gate passes.

| Phase | Work | Gate |
| --- | --- | --- |
| 0 | **Webcam baseline** in Python: YOLO26n + ByteTrack, IDs and trails on screen | Stable ID on a 15 s recording; FPS and two failures logged in NOTES.md |
| 1 | Pipeline package: `Detector` protocol, YOLO26 and RF-DETR adapters, tracker wrapper, frame sources, tests | Swap detector or tracker with one config line |
| 2 | Analytics + **event log**: line crossings, zones, dwell, heatmap; events as JSON lines (`enter`, `exit`, `zone_enter`, `zone_exit`, `dwell`) with track ID and timestamp | Counts match a hand count on one clip; event log reproduces them |
| 3 | Evaluation and **scene tuning**. (1) Find why the false crossings happen: clips around each one, detector merges and misses against ID swaps. (2) Tracker comparison on MOT20-03 and 05 with the detector, line and scoring fixed: ByteTrack, ByteTrack with a longer buffer, BoT-SORT with ReID on and off, OC-SORT. (3) An event level safeguard against crossings made by an ID change, with tests for genuine repeat crossings. (4) `pathsense tune`: profile each scene (people size, density, overlap, motion, view), run 5 to 10 candidate configs from cached detections, score them, and say whether an error is a scene limit, a setting or a bug. Tune on MOT20-03 and 05, validate on the other MOT scenes and the sample clips. (5) Recover missed crossings the tracker loses (tracks lost, started late or split at the line), including appearance embeddings for re-linking; each change is kept only if it raises recall without adding wrong crossings on the check scenes. TrackEval HOTA, IDF1, MOTA; FPS on GTX 1650 and CPU | False crossings per scene, crossing precision and recall per direction, count error, ID switches, fragmentation and FPS in a README table for all eleven scenes. The chosen tracker cuts MOT20-03 false crossings without losing real ones (recall no worse than ByteTrack), and holds on the validation scenes |
| 4 | **Train on Kaggle**: fine tune YOLO26n and RF-DETR Nano on Open Images people; export ONNX; push weights + model card to the Hub | Fine tuned model beats COCO baseline on held out data; ONNX runs in onnxruntime (Python) |
| 5 | **Browser app (A)**: Static Space; ONNX detector in the browser; TypeScript ByteTrack; line counter, heatmap, event log; webcam and clip upload; WebGPU with WASM fallback | Same detections in Python and browser give the same track IDs and counts on a test clip |
| 6 | **Ask your footage (B)**: query tools over the event log; small in browser LLM picks tools and answers; optional "bring your own API key" mode | 20 question test set answered with correct numbers |
| 7 | Hardening and showcase: CI deploy of the Space on tag, README GIF, model card, write up | Public demo, model card and write up linked from the README |

Ideas for after the current phase, kept apart so they do not pull work forward:
[BACKLOG.md](BACKLOG.md).

**Plan change 2026-10-28:** testing on four public clips showed that every scene has its
own problems (small far people, crowds at train doors, a cart hiding the legs, curved
edges from a wide lens). Phase 3 now also builds scene tuning, so each scene is measured
and tuned on its own, and a wrong count can be traced to the scene, a setting or the code.

**Plan change 2026-10-28 (2):** scoring against MOT17 and MOT20 hand labelled tracks
(`scripts/mot_counts.py`) showed that the remaining errors in crowds are ID swaps, not
counting logic. Phase 3 starts with a controlled tracker comparison on the two worst
scenes, and judges every change by false crossings and missed crossings separately, so a
fix that removes false crossings by also losing real ones does not pass. ReID compares
clothing, not faces; it stays in memory for one video, and must earn its cost in the
browser to be kept.

**Plan change 2026-10-30:** the tracker study found that most missed crossings in crowds
are not detection failures: 156 of 437 had the person detected but the track lost,
started late or split at the line. Phase 3 gains step 5 to recover them before the
counting stack is frozen for Phase 4, tested on the study harness and kept only if it
passes on the check scenes. Appearance embeddings (clothing, in memory for one video) are
tried there for re-linking lost people ([RESEARCH.md](RESEARCH.md)). Which trackers the
browser ships is decided at Phase 5, from the results after Phase 4's detector.

**Stretch (C):** on a cloud GPU, auto label real security camera frames from MEVA
(CC BY 4.0) with SAM 3 or a large RF-DETR, train the nano model on them, publish before and
after numbers on CCTV style video.

## Current status

- Repo scaffolded, CI green.
- `train/get_data.py` downloads Open Images (deployable) and CrowdHuman (benchmarks).
- **Phase 0 gate passed (2026-10-27).** `scripts/webcam_baseline.py`: ID held for 76 s;
  20.5 FPS camera bound (20 ms detect + track on the GTX 1650). Failures logged in
  NOTES.md: re-entry gets a new ID, and partly visible people are fragmented or missed.
- **Phase 1 gate passed (2026-10-27), v0.2.0.** `pathsense run` swaps YOLO26n, RF-DETR
  Nano, ByteTrack, BoT-SORT and OC-SORT with one config line; all six combinations
  measured in NOTES.md.
- **Phase 2 gate passed (2026-10-28), v0.3.0.** Lines, zones, dwell, occupancy, heatmap
  and the JSON lines event log. people-walking: 11 IN, 7 OUT, matching the hand count;
  MOT17-04 against hand labelled truth: line 8 IN (8), 4 OUT (5), zone dwell 450 s against
  448 s visible; the event log alone reproduces every count. **Limitation:** in dense
  crowds (MOT20) ByteTrack swaps IDs and the line counts false crossings the wrong way
  (MOT20-03: 28 against 4); far groups in zones are undercounted. Both go to Phase 3.
- **Phase 3 in progress.** Steps 1 and 2 done (2026-10-30, [TRACKER_STUDY.md](TRACKER_STUDY.md)):
  the wrong-way crossings are tracker ID swaps. Tuned ByteTrack with the new bounce rule
  cuts wrong crossings from 392 to 129 on the two crowds it was tuned on and from 79 to 62
  on the five check scenes, missing 2 fewer on the crowds and 4 more on the check
  scenes. The check scenes favour OC-SORT and BoT-SORT, so the tracker is a per scene
  choice. `line_bounce` is on by default; the crowd tracker settings live in the crowd
  scene files, since they split tracks on the close-up grocery clip. Step 3's crossing
  safeguard is the bounce rule. Step 4, `pathsense tune`, built (NOTES.md): it reproduces
  the study and picks a setup per scene against the scene's `truth:` block; subway has no
  truth yet. Step 5 done (2026-10-31, [TRACKER_STUDY.md](TRACKER_STUDY.md#appearance-step-5-does-telling-people-apart-by-looks-recover-crossings)):
  no change passed clearly, so nothing new is on by default. Position only stitching at
  the line and appearance embeddings both failed or passed too narrowly to trust;
  embeddings stay as a toggle, off, and re-entry remains untested. Next: the gate run.
