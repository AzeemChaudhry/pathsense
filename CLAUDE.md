# CLAUDE.md

Context for Claude Code working in this repo.

## Project

PathSense: people detection, tracking and movement analytics, deployed on Hugging Face.
Read `docs/PLAN.md` for phases and gates, `docs/DATASETS.md` for data and license
decisions, and `NOTES.md` for the experiment log.

## How Azeem wants to work

- **One task at a time.** Assign a single next task from the current phase; keep the rest
  of the backlog visible in `docs/PLAN.md` but don't ask for it until it's next.
- **Coach, don't do it for him** on learning tasks (for example `scripts/webcam_baseline.py`):
  explain, review and point at the next step rather than writing the whole solution, unless
  he asks for the code.
- Never use em dashes or dashes as punctuation in replies. Keep wording plain and direct.

## Commits

- Author commits as **Azeem Chaudhry <chaudhry.azeem0308@gmail.com>** (GitHub: AzeemChaudhry).
- Run `uv run ruff check .` and `uv run pytest` before every commit; CI runs both.

## Environment

- Ubuntu, GTX 1650 (4 GB VRAM), 16 GB RAM. Use small models, `imgsz=640`, `batch=8`, AMP.
- Python 3.11 with uv: `uv sync --extra dev` (add `--extra data` for downloads).

## Rules

- Never commit data, videos or weights; `data/`, `*.pt`, `*.onnx`, `*.mp4` are git ignored.
  Weights go to a Hugging Face model repo, data to a dataset repo (Open Images only:
  CrowdHuman images must not be redistributed).
- Deployable models train on Open Images, not CrowdHuman (non commercial).
- License is AGPL 3.0 because of Ultralytics.
- Every detector returns `supervision.Detections`; keep that contract.
