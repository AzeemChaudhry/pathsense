# Changelog

All notable changes to PathSense are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) as described in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#versioning): each passed phase gate is a new minor
version, fixes are patch versions, and 1.0.0 is the finished showcase (Phase 7).

## [Unreleased]

### Added

- Line counting: `analytics.line` in the config counts people crossing it in each
  direction. The window shows the line and IN and OUT totals, and the summary line ends
  with `in=… out=…`. Each crossing is an event with time, frame and track ID, the start
  of the Phase 2 event log. On the sample clip it matches a hand count (11 in, 7 out).
  A crossing counts once the person stays on the new side for `analytics.line_confirm`
  seconds (0.5 by default), so an arm waving across the line on a close-up webcam is not
  counted again and again. Standing on the line does not confirm a crossing, and someone
  hidden behind another person for under a second is not counted on the wrong side.
- `pathsense run --events runs/clip.jsonl` writes the event log, one JSON line per event
  (time, frame, track ID, type), as events happen. `load_events` and `count_events` read
  it back: the sample clip's log alone gives 11 in and 7 out. A run stopped with Ctrl+C
  keeps its log; a last line cut short by a crash is skipped when reading.
- `PATHSENSE_CAMERA=1 uv run pytest tests/test_live_camera.py -s`: a manual webcam test
  that passes when someone crosses a line within a minute.
- Webcam capture calibration: the first run with a new camera measures its best capture mode in
  about 4 seconds and saves it in `~/.config/pathsense/cameras.json`; later runs recognise
  the camera by its USB ids and reuse it. The test webcam goes from 640x480 to 1280x960 at
  the same 30 FPS. `pathsense calibrate` and `pathsense run --recalibrate` measure again,
  and a `camera` config section sets the target and turns capture calibration off.

### Security

- Pillow upgraded from 12.3.0 to 13.0.0, which fixes 39 Dependabot alerts (5 critical,
  23 high). The unused `app` extra (Gradio, which requires Pillow below 13) is removed
  until the fallback demo is built.
- Remaining alerts: pymongo 4.9.2 and strawberry-graphql 0.316.0 come only from FiftyOne
  in the `data` extra, whose latest release pins them. They are used only by the local
  dataset downloader, never by the pipeline or a server.
- `main` is protected by a ruleset: pull requests only, passing CI and signed commits
  required, no force pushes or deletion. Release tags can only be created, moved or
  deleted by the maintainer.
- Workflows from fork pull requests need maintainer approval; only GitHub's own actions
  and the third party actions in use may run (`claude-code-action`, `setup-uv`, and
  `setup-bun` pinned to the commit the Claude action needs).
- The `@claude` action is pinned to a commit, and a test fails if any third party action
  that receives a secret is not.
- `SECURITY.md` explains how to report a vulnerability privately and what is protected.

### Changed

- Tracks now start at 0.6 detector confidence instead of ByteTrack's 0.7
  (`tracker.params.track_activation_threshold`), which missed a person crossing the line
  at 0.61. Tracker settings that only another tracker takes are skipped with a warning,
  so `tracker.name=ocsort` still works; unknown names are still errors, and
  `tracker.params` cannot set `frame_rate`, which comes from the video.
- The `analytics` config section is checked like the others: unknown keys, malformed or
  non finite line points, and a margin or hold that is not a number from 0 are errors.
- `CLAUDE.md` is no longer part of the repository (it holds local assistant
  instructions). The project rules it contained now live in `docs/DEVELOPMENT.md`.
- `docs/PLAN.md` is now a public roadmap: pitch, architecture, phases, gates and status.
  Planning notes outside the engineering scope, and the market research
  (`docs/MARKET.md`), moved to a private notes repository.
- `docs/DEVELOPMENT.md` setup uses placeholders instead of a personal email address.

## [0.2.0] - 2026-10-27

Phase 1: pipeline package. Gate passed: the detector and the tracker each swap with one
config line, measured for all six combinations in NOTES.md.

### Added

- `pathsense run` command: webcam, video file or stream; `--set section.key=value`
  overrides, `--save` real time recording, `--no-show` and `--max-frames` for benchmarks,
  and a summary line with FPS, detect and track ms and unique IDs.
- `pathsense.config`: YAML config with defaults, `--set` overrides parsed as YAML, and
  errors for unknown sections, keys, detectors or trackers instead of silent defaults.
- Detector registry and adapters: `yolo26n`, `yolo26s` (Ultralytics) and `rfdetr-nano`,
  `rfdetr-small` (RF-DETR, `--extra rfdetr`). Every adapter returns people only as
  `supervision.Detections` with class 0, whatever numbering the model uses.
- Tracker wrapper for ByteTrack, BoT-SORT and OC-SORT: real timestamps, confirmed tracks
  only, frames passed only to BoT-SORT, extra parameters from config.
- `Pipeline` (detector then tracker with per stage timing), `VideoSource` (frames with
  timestamps), `RealTimeVideoWriter` and `Annotator` as reusable modules.
- Tests for every module with fake models (no GPU or downloads), plus opt in slow tests
  with the real models (`PATHSENSE_SLOW=1 uv run pytest -m slow`).
- `commit-msg` hook and a CI check that reject tool trailers in commit messages.
- Tests that keep the ruff version, the setup-uv pin and the changelog links consistent.
- Pull request template with the review checklist.
- `docs/PROGRESS.md`: before and after each phase, with images from the sample clip, the
  measurements and the open issues carried forward.

### Changed

- `scripts/webcam_baseline.py` imports its helpers from the package instead of keeping
  copies.
- `@claude` is review only: it runs with a read only token and no file editing tools, so it
  can comment but never push. Replies come from `github-actions[bot]`.
- Releases are published only from signed, verified tags on `main` whose commit passed CI.
- CI runs the pre-commit hooks (including YAML, TOML and merge conflict checks) instead of
  separate ruff steps, caches the hook environments, never cancels runs on `main`, and
  has a timeout on every job.
- `docs/DEVELOPMENT.md` lists every rule with what enforces it; `CLAUDE.md` points there.

### Fixed

- NOTES.md no longer claims the `trackers` BoT-SORT has appearance re-identification; it
  only does camera motion compensation.

## [0.1.0] - 2026-10-27

Phase 0: webcam baseline. Gate passed: one person kept the same ID for 76 s.

### Added

- `scripts/webcam_baseline.py`: YOLO26n detection and ByteTrack tracking on a webcam, video
  file or stream, with `#id` labels, trails coloured per track, loop FPS and detect plus
  track time on screen, a real time `--save` recording and a summary line on exit.
- `train/get_data.py`: downloads Open Images V7 people (deployable) and CrowdHuman
  (benchmarks only) and converts both to YOLO labels; Roboflow Universe exports too.
- Project plan, dataset license review, market research and free GPU guide in `docs/`.
- Experiment log in `NOTES.md` with the Phase 0 measurements: 20.5 FPS capped by the
  camera, 20 ms detect plus track on a GTX 1650, and two tracking failures (new ID on
  re-entry, missed or fragmented people at the frame edge).
- `uv.lock` so every install resolves the same versions.
- CI runs ruff lint, ruff format and pytest on Python 3.11 and 3.12 with the locked
  dependencies.
- `@claude` mentions on issues and PRs run Claude Code for reviews (owner and
  collaborators only).
- Pushing a `v*.*.*` tag creates a GitHub release from this changelog.
- `docs/DEVELOPMENT.md`: branches, commits, signing, versioning and release steps.

### Changed

- Direction: private, in browser people analytics with plain language questions over an
  event log, instead of a server side counting demo.
- The package version is defined once, in `src/pathsense/__init__.py`.

[Unreleased]: https://github.com/AzeemChaudhry/pathsense/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AzeemChaudhry/pathsense/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AzeemChaudhry/pathsense/releases/tag/v0.1.0
