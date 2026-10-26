# Development workflow

The single source for how changes get into PathSense and the rules the code follows. The
same rules apply to people and to AI assistants. Most rules are enforced by a hook, CI or a GitHub
ruleset; the table says which, so nothing depends on memory.

## The rules

| Rule | Enforced by |
| --- | --- |
| Every change goes through a PR into `main`; no direct pushes, no force pushes | `main` ruleset |
| CI (`lint`, `test (3.11)`, `test (3.12)`) must pass before merging | `main` ruleset |
| Commits are signed | `main` ruleset (rejects unsigned commits) |
| Merge with a merge commit, never squash or rebase (keeps each signature) | repo settings |
| Commit messages carry no `Co-Authored-By`, `Claude-Session` or "Generated with Claude" lines | `commit-msg` hook, CI `lint` |
| Code is formatted and lint clean; no stray whitespace, broken YAML/TOML or files over 1 MB | `pre-commit` hooks, CI `lint` |
| `uv.lock` matches `pyproject.toml` | CI `test` (`uv sync --locked`) |
| ruff version agrees between pre-commit and `uv.lock` | `tests/test_repo_consistency.py` |
| Version, changelog section and changelog links agree | `tests/test_changelog.py` |
| Releases come only from signed tags on `main` whose commit passed CI | `release.yml`, tag ruleset |
| `@claude` reviews only and cannot push | `claude.yml` (read only token) |
| Docs and `CHANGELOG.md` change in the same PR as the code | PR template checklist, review |

## Project rules

These hold for every change, whoever writes it:

- **Detector contract.** Every detector returns `supervision.Detections` with people only and
  `class_id` 0 (`pathsense.detect.base`). New models are added to the `DETECTORS`
  registry and never imported directly elsewhere.
- **Config, not code.** Stages and thresholds are chosen in `configs/default.yaml` or with
  `--set`, never hard coded.
- **Numbers come from code.** In the question answering phase, answers are computed by
  deterministic functions over the event log; the language model only picks the function
  and words the answer. It never computes counts or times itself.
- **Browser parity.** The browser tracker must match the Python tracker; a parity test feeds
  the same detections to both and compares IDs and counts.
- **Privacy.** No face recognition, identification, demographics or emotion analysis, ever.
  The browser app never uploads video. Docs and images use public sample footage only,
  never recordings of identifiable people taken for this project.
- **No data, video or weights in git** (`data/`, `*.mp4`, `*.pt`, `*.onnx` are ignored).
  CrowdHuman images are never redistributed, and deployable models train on Open Images
  (`docs/DATASETS.md`).
- **Licenses.** The repo is AGPL 3.0 because of Ultralytics. RF-DETR (Apache 2.0) is the
  model for commercial work.

## Setup (once per clone)

```bash
uv sync --extra dev          # add --extra data for dataset downloads
uv run pre-commit install    # installs the pre-commit and commit-msg hooks
```

Signing, with the SSH key registered on GitHub as a signing key:

```bash
git config user.name "Azeem Chaudhry"
git config user.email "chaudhry.azeem0308@gmail.com"   # must be verified on AzeemChaudhry
git config gpg.format ssh
git config user.signingkey ~/.ssh/github_signing.pub
git config commit.gpgsign true
git config tag.gpgsign true
```

Python 3.11 or 3.12. After changing `pyproject.toml`, run `uv lock` and commit the lock file
with it.

## Making a change

1. Branch from an up to date `main`, named by type: `phase1/detector-adapters`,
   `fix/qt-fonts`, `docs/kaggle-guide`, `chore/ci-cache`.
2. Commit in small, focused steps. Formatting only changes, dependency bumps and behaviour
   changes go in separate commits.
3. Message: imperative subject under about 70 characters with no trailing period ("Add
   ByteTrack wrapper"), a blank line, then a body that explains why.
4. Before pushing: `uv run pre-commit run --all-files` and `uv run pytest`.
5. Open a PR. The template lists what to fill in; update the docs and `CHANGELOG.md` in
   the same PR (see [Documentation](#documentation)).
6. Optionally ask `@claude` for a review (below).
7. When CI is green, merge (merge commit is the only option) and delete the branch.

## CI

`.github/workflows/ci.yml` runs on every PR and every push to `main`:

| Job | Steps | Time |
| --- | --- | --- |
| `lint` | all pre-commit hooks on every file; on PRs, the commit message check | about 15 s |
| `test (3.11)`, `test (3.12)` | `uv sync --locked --extra dev`, then `pytest` | about 40 s |

A new push to a PR cancels that PR's older run. Runs on `main` are never cancelled, because
releases need a finished CI result for the merge commit. Every job has a timeout.

Tests must not need a GPU, a webcam or downloaded data: use the fakes in `tests/` (a fake
detector, synthetic boxes, a tiny generated video) and keep model imports inside the
`load()` methods so modules import without weights.

Tests that load real weights are marked `@pytest.mark.slow` and skipped unless asked for.
Run them locally before a PR that touches a detector or the pipeline:

```bash
uv sync --extra dev --extra rfdetr
mkdir -p data/samples && cd data/samples && uv run python -c \
  "from supervision.assets import download_assets, VideoAssets; download_assets(VideoAssets.PEOPLE_WALKING)"
cd ../.. && PATHSENSE_SLOW=1 uv run pytest -m slow
```

`data/samples/people-walking.mp4` (1080p, 25 FPS, about 25 people) is also the clip used for
the speed tables in NOTES.md. It is not committed (`data/` is ignored).

## Review with Claude

Comment `@claude` on a PR, in a PR review, or in a new issue, with what you want, for
example `@claude review this PR for tracking bugs and missing tests`.

- The `Claude review` workflow runs Claude Code with this file as context,
  and Claude replies in the thread as `github-actions[bot]`.
- It is **review only**. Claude acts with the workflow's own token, which has read only
  access to the code, and its file editing tools are disabled. It can read code and CI
  logs and write comments; it cannot commit, push or open PRs. Apply suggested changes
  yourself (or with Claude Code locally) on your branch.
- Only the repo owner and collaborators can trigger it, and bot comments never do.
- Needs the repo secret `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`, uses the Claude
  subscription). No GitHub app is needed.

## Versioning

[Semantic Versioning](https://semver.org/), mapped to the phase plan in `docs/PLAN.md`:

| Change | Bump | Example |
| --- | --- | --- |
| A phase gate passes | minor | Phase 1 done: `0.1.0` to `0.2.0` |
| Fix after a release | patch | `0.2.0` to `0.2.1` |
| Phase 7 showcase complete | major | `1.0.0` |

Phase N is `0.(N+1).0`. The version lives only in `__version__` in
`src/pathsense/__init__.py`; `pyproject.toml` reads it from there.

## Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Any PR that
changes something a user or contributor would notice adds a line under `## [Unreleased]`,
grouped as Added, Changed, Fixed or Removed. Write it for a reader of the project, not as
a list of commits.

## Releasing

1. On a branch, bump `__version__` in `src/pathsense/__init__.py`.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new empty
   `## [Unreleased]` above it, and update the links at the bottom:
   ```
   [Unreleased]: https://github.com/AzeemChaudhry/pathsense/compare/vX.Y.Z...HEAD
   [X.Y.Z]: https://github.com/AzeemChaudhry/pathsense/compare/vPREVIOUS...vX.Y.Z
   ```
   `tests/test_changelog.py` fails until all of this lines up.
3. PR, CI green, merge.
4. Tag the merge commit on `main` with a signed tag and push it:
   ```bash
   git switch main && git pull
   git tag -s vX.Y.Z -m "vX.Y.Z: <phase or summary>"
   git push origin vX.Y.Z
   ```
5. `release.yml` checks that the tag is on `main`, is signed and verified, matches
   `__version__`, and that CI passed on that commit (it waits up to 20 minutes if CI is
   still running). Then it publishes the GitHub release with that version's changelog
   section as the notes. If any check fails, nothing is published: fix the cause, delete
   the tag (`git push origin :refs/tags/vX.Y.Z`, allowed for the owner only) and tag again.

## Documentation

Docs ship in the same PR as the change:

| When you... | Update |
| --- | --- |
| add or change a script, flag or module | `README.md` usage, and the module docstring |
| run an experiment or measure something | `NOTES.md` (what, the number, the decision) |
| pass a phase gate | `docs/PLAN.md` "Current status", the README roadmap, and a before and after section in `docs/PROGRESS.md` (images from public sample footage only) |
| change data sources or licenses | `docs/DATASETS.md` |
| change how work is done | this file |
| change anything users notice | `CHANGELOG.md` under Unreleased |
