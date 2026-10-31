# Research: browser stack and appearance tracking

Notes from 2026-10-30, written before Phase 3 step 5 and Phase 5, to choose libraries and
methods with the evidence in one place. Each section ends with what PathSense will do and
what is still to be measured. Nothing here changes the plan by itself; changes go through
[PLAN.md](PLAN.md).

The constraint behind every choice: **the video never leaves the visitor's device.** So
everything below runs in the browser on the visitor's own CPU or GPU, and anything we
learn about a person (a box, a track, an appearance embedding) stays in memory for that
one video.

## 1. Running models on the visitor's hardware

| Option | What it is | Fit for PathSense |
| --- | --- | --- |
| [ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web) | The ONNX runtime compiled for the browser: WASM (CPU) and WebGPU backends, and a WebNN backend | **Main choice.** The same ONNX file runs in Python (`onnxruntime`) and the browser, which is what the Phase 5 parity test needs. WebGPU is still labelled experimental; WASM is the universal fallback |
| [Transformers.js](https://huggingface.co/spaces/webml-community/RF-DETR-WebGPU) | Hugging Face's pipelines on top of ONNX Runtime Web | Working demos of [RF-DETR Nano](https://huggingface.co/spaces/webml-community/RF-DETR-WebGPU/blob/main/README.md) and [YOLO26](https://huggingface.co/spaces/ipepe/YOLO26-WebGPU/blob/main/index.html) on WebGPU. Convenient, but it hides pre and post processing we must keep identical to Python. Useful as a reference, and for the Phase 6 LLM |
| WebNN | A browser API that can reach the NPU | [Chromium only, partly behind a flag](https://blog.openreplay.com/on-device-ai-browser/); GPU and NPU support [still limited](https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98/m/9mwAcwz3AAAJ). Reachable later through ONNX Runtime Web's WebNN provider with no code change; not a target now |

Practical points from working YOLO browser projects
([1](https://github.com/nomi30701/yolo-object-detection-onnxruntime-web),
[2](https://github.com/nomi30701/yolo-onnx-benchmark-web)):
- Export ONNX at **opset 12** for WebGPU compatibility; keep nano models (about 10 MB)
  for memory.
- Detect WebGPU, fall back to WASM, and **show which backend is running** (already a
  Phase 5 rule, since Firefox and most Linux Chrome have no WebGPU).
- No published browser benchmark answers our question, so we measure our own model with
  a benchmark page, on WebGPU and WASM, before choosing between YOLO26n and RF-DETR Nano.

**Video frames.** For uploaded clips, [WebCodecs](https://developer.mozilla.org/Web/API/WebCodecs_API)
decodes frame by frame with hardware acceleration and works in a dedicated worker;
[Chrome's guidance](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs)
is to do per frame work in a worker so the page stays responsive. The plan for Phase 5:
decode with WebCodecs in a worker, run the detector and tracker in the same worker, draw
on an `OffscreenCanvas`, and send only results (boxes, IDs, events) to the page. Every
frame is processed, so the event log does not depend on playback speed. Webcam frames come
from `MediaStreamTrackProcessor` where available.

## 2. Drawing lines and polygons on an uploaded video

Lines and zones are placed per scene from its real use case, so the browser needs an
editor: click to add points, drag a corner, insert a point on an edge, delete one, name
the zone, for lines and for polygons with any number of points.

| Option | Notes |
| --- | --- |
| [Konva](https://konvajs.org/docs/overview.html) | Canvas 2D framework with `Line`, polygon and drag and drop built in. A stage laid over the `<video>`, with one draggable circle per corner, is the [usual pattern](https://www.github.com/definite2/kanva-draggable-polygon). Mature and framework free |
| Plain SVG over the video | No dependency. Polygons and handles are DOM elements, so hit testing, focus and keyboard access come free. Enough for a few shapes |
| Annotation kits ([polygon-annotation](https://github.com/definite2/polygon-annotation), [@pixano/graphics-2d](https://www.npmjs.com/package/@pixano/graphics-2d)) | Built for labelling datasets; video support partial or planned; more than we need |

**Choice: plain SVG over the video,** with Konva as the fallback if the editor grows. A
handful of shapes does not need a canvas framework, and SVG keeps the editor accessible
and easy to test. Two rules carry over from Python:
- Shapes are stored in **video pixel coordinates**, not screen coordinates, and
  converted on every resize, so a zone drawn in the browser is exactly the zone the
  counters use, and can be exported as the same YAML a scene file holds.
- The editor shows the current frame while drawing, so the user places the line where
  people actually walk (the per scene rule from Phase 2).

## 3. Appearance embeddings for tracking

Our trackers match people by position and motion only. The study showed what that costs
in a crowd: IDs slide between overlapping people, and 156 missed crossings had the person
detected but the track lost, started late or changed ID at the line
([TRACKER_STUDY.md](TRACKER_STUDY.md#where-the-misses-come-from)). An appearance
embedding (a short vector describing what someone's clothes look like) can tell two
overlapping people apart and link a track that comes back after being hidden.

**Privacy.** An embedding compares clothing and build, not faces. It is computed in the
browser, kept in memory for one video and never stored or sent. It is not used to
recognise anyone across videos. The UI and README must say so.

| Method | Cost | Notes |
| --- | --- | --- |
| [OSNet x0.25](https://arxiv.org/pdf/1910.06827) (separate ReID model) | About 0.2 M parameters, under 1 MB. One browser project [reports about 9 ms per person](https://github.com/Basiliskenzahn/Laser-Tag/pull/9) with ONNX Runtime Web (anecdotal, desktop only) | The standard light ReID network; [int8 versions](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/re_identification/osnet/README.md) keep about 70% mAP. At 9 ms a person, 50 people in a crowd would cost 450 ms a frame, so it can only run on the few people that need it |
| [LITE](https://kinetik.umm.ac.id/index.php/kinetik/article/view/2478) (embeddings from the detector's own feature maps) | Nearly free: the detector has already computed the features | No second network in the browser. Weaker than a trained ReID model, but evaluated on MOT20. Needs the ONNX export to also output a feature map |
| Colour histogram per box | Free | The classic baseline; weak in crowds of similar dark clothes at night (MOT20) |

Where embeddings would earn their cost, in order:
1. **Re-linking**: the stitcher already links a lost person to a new box by position. In
   crowds it picks the wrong person (it added 13 to 63% wrong crossings). Checking that
   the two look alike before linking is exactly the missing test. Runs only when a track
   is lost or new: a few people a second, not everyone every frame.
2. **Near the line**: matching with appearance only for boxes in a band around the
   counting line, where a swap becomes a wrong crossing.
3. Everywhere (DeepSORT style): the most accurate and the most expensive; unlikely to fit
   the browser budget in crowds.

**What PathSense will do.** Phase 3 step 5 tests embeddings for uses 1 and 2, with
OSNet x0.25 as ONNX run through `onnxruntime` in Python. The same file then runs in the
browser through ONNX Runtime Web, so a result measured here is the result shipped. An
embedding is kept only if it reduces missed or wrong crossings on the check scenes, and
its cost per frame is reported next to the gain. LITE comes after, if OSNet earns its
place and the cost needs cutting.

**Result (2026-10-31).** Tested in and around the tracker, including Deep OC-SORT's
confidence weighted memory and adaptive weighting: not kept for live tracking, off by
default ([TRACKER_STUDY.md](TRACKER_STUDY.md#appearance-step-5-does-telling-people-apart-by-looks-recover-crossings)).
Two corrections to the notes above. `trackers` has no appearance tracker in any release
from 2.1.0, when it dropped DeepSORT and ReIDModel. And the OSNet weights cannot ship: they
are trained on MSMT17, whose agreement allows academic use only. Re-entry, someone coming
back minutes later, is untested and remains the open use.

**Input size (2026-10-31).** These notes missed it, and it matters more than any library
here: on MOT20-02 a crossing F1 of 0.31 at 640 against 0.71 at 1600
([DETECTOR_STUDY.md](DETECTOR_STUDY.md)). The browser must measure what it can afford.

## 4. Libraries for YOLO with ByteTrack

| Library | Language | Has | Fit |
| --- | --- | --- | --- |
| [Roboflow trackers](https://trackers.roboflow.com/develop/trackers/core/deepsort/tracker) (we use 2.6.1) | Python, Apache 2.0 | SORT, ByteTrack, BoT-SORT, OC-SORT, C-BIoU, McByte; TrackEval compatible evaluation; an Optuna tuner | **Stays.** Our study and evaluation run on it. Its tuner optimises one tracker's MOTA or HOTA on MOT files; `pathsense tune` needs crossing F1, our layers and scene profiles, so it is built on our own replay instead. Its docs describe a DeepSORT with a ReID model, not in 2.6.1 |
| [BoxMOT](https://pypi.org/project/boxmot/20.0.0/) | Python, AGPL 3.0 | ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, BoostTrack; ReID models (OSNet, LightMBN, CLIP-ReID) with ONNX export | The best source of **ReID weights and ONNX export** for step 5, and of appearance trackers to compare against. Not a runtime dependency |
| Ultralytics | Python, AGPL 3.0 | YOLO26 training and export; ByteTrack and BoT-SORT built in | Training and ONNX export (Phase 4). Its tracker is not used; ours is chosen in config |
| supervision | Python, MIT | Detections, zones, annotators | Stays, as the shared detections type |
| ByteTrack in JavaScript | | [No maintained npm library](https://www.npmjs.com/package/bytetrack); one [demo app](https://github.com/topics/bytetrack?l=javascript) | **We port it** to TypeScript in Phase 5 (already the plan), from the `trackers` implementation, with a parity test against Python |

Licences: everything we ship in the browser must be compatible with the repo's AGPL 3.0,
and Apache 2.0 parts are preferred where they are close (PLAN.md). BoxMOT is AGPL, so it
is used for research and weights, and any code we ship is our own.

## What to measure next

1. Phase 3 step 5: OSNet x0.25 embeddings for re-linking and near the line, against the
   study's chosen setup, on the tune and check scenes. Gain in crossings, cost in ms.
2. Phase 5 start: a benchmark page for the detector and the ReID model on WebGPU and
   WASM, on the development laptop (WASM only on Linux Firefox) and one other machine.
