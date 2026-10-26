# CLAUDE.md

Context for Claude Code working in this repo. Read this first, then `docs/PLAN.md`.

## Project in one paragraph

PathSense is **private, in browser people analytics you can question in plain language**.
It detects and tracks people in a webcam stream or an uploaded security camera clip, turns
movement into counts, occupancy, dwell time, trails, a heatmap and an **event log**, and
answers questions about that log with a small LLM. The finished product runs entirely in
the visitor's browser on a Static Hugging Face Space, so no video ever leaves their
device. This repo also holds the Python research, evaluation and training pipeline.

## Where things are decided

| File | What it holds |
| --- | --- |
| `docs/PLAN.md` | Pitch, architecture, phases 0 to 7 with one gate each, current status |
| `docs/MARKET.md` | Why plain counting is not enough; what is new in 2026 |
| `docs/DATASETS.md` | Data license review; Open Images for deployable models |
| `docs/COMPUTE.md` | Free GPUs; how to train on Kaggle and push weights to the Hub |
| `NOTES.md` | Experiment log: what was tried, the number, the decision |

Do not change the direction in `docs/PLAN.md` without Azeem agreeing; if something makes
a phase unworkable, say so and propose the change.

## How Azeem wants to work

- **One task at a time.** Work out the current phase from `docs/PLAN.md` (first phase whose
  gate has not passed) and give a single next task. Keep the backlog visible, but do not
  ask him to do anything beyond the current task.
- **Coach on learning tasks.** For Phases 0 to 3 especially, explain, review and point at
  the next step rather than writing the whole solution, unless he asks for the code.
- **Gates are real.** Do not move to the next phase until the gate passes; update the
  "Current status" section of `docs/PLAN.md` and tick the README roadmap when it does.
- Never use em dashes or dashes as punctuation in replies. Keep wording plain and direct.

## Commits

- Author as **Azeem Chaudhry <chaudhry.azeem0308@gmail.com>** (GitHub: AzeemChaudhry).
- Run `uv run ruff check .` and `uv run pytest` before every commit; CI runs both.

## Environment and accounts

- Laptop: Ubuntu, GTX 1650 (4 GB VRAM), 16 GB RAM. Use it for inference, debugging and tiny
  runs only (`imgsz=640`, `batch=8`, AMP).
- **Training runs on free cloud GPUs, Kaggle first** (`docs/COMPUTE.md`).
- Python 3.11 with uv: `uv sync --extra dev` (`--extra data` for downloads, `--extra app`
  for the Gradio fallback).
- Hugging Face user: `MuhammadAzeem`. Weights go to `MuhammadAzeem/pathsense-detector`;
  the browser app will be a Static Space under the same account.
- WebGPU is limited on Linux (Chrome needs recent Intel GPUs; Firefox has none), so the
  browser app must work on the WASM backend too. Test both.

## Technical rules

- Every Python detector returns `supervision.Detections`; keep that contract.
- The browser tracker must match the Python tracker: keep a parity test that feeds the
  same detections to both and compares track IDs and counts.
- The event log is the interface between vision and language. Q&A answers must come from
  deterministic functions over the event log; the LLM only picks functions and phrases the
  answer. Never let the model compute counts or times itself.
- Privacy: no face recognition, identification, demographics or emotion, ever. No video
  upload in the browser app. Say this in the UI and README.
- Never commit data, videos or weights (`data/`, `*.pt`, `*.onnx`, `*.mp4` are ignored).
  CrowdHuman images must never be redistributed; deployable models train on Open Images.
- Licenses: repo is AGPL 3.0 because of Ultralytics. Prefer shipping RF-DETR Nano
  (Apache 2.0) in the browser demo if it is close in accuracy and speed.
