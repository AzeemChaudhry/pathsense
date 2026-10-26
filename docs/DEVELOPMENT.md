# Development workflow

How changes get into PathSense: branches, commits, checks, review, versions and releases.
`CLAUDE.md` points Claude Code here, so the same rules apply to people and to Claude.

## Setup

```bash
uv sync --extra dev          # add --extra data for dataset downloads
uv run pre-commit install    # runs ruff and file hygiene checks on every commit
```

Python 3.11 or 3.12 (CI tests both). `uv.lock` pins every dependency; after changing
`pyproject.toml` run `uv lock` and commit the lock file with it. CI fails if they disagree.

## Branches and pull requests

- `main` is always green and releasable. Nothing is committed to it directly.
- One branch per task, named by type: `phase1/detector-adapters`, `fix/qt-fonts`,
  `docs/kaggle-guide`, `chore/ci-matrix`.
- Open a PR into `main`. Merge with a **merge commit** (not squash or rebase) so each signed
  commit keeps its signature, then delete the branch.
- A PR is ready when CI is green and the docs in the same PR describe the change (see
  [Documentation](#documentation)).

## Commits

- **Signed.** Every commit and tag is signed, so GitHub shows it as Verified. Set up once
  per clone with the SSH key registered on GitHub as a signing key:

  ```bash
  git config user.name "Azeem Chaudhry"
  git config user.email "chaudhry.azeem0308@gmail.com"
  git config gpg.format ssh
  git config user.signingkey ~/.ssh/github_signing.pub
  git config commit.gpgsign true
  git config tag.gpgsign true
  ```

  The email must be a verified email on the GitHub account, or commits show as Unverified
  and are attributed to whichever account owns that address.
- **Messages.** Subject in the imperative, under about 70 characters, no trailing period
  ("Add ByteTrack wrapper", not "Added ByteTrack wrapper."). Blank line, then a body that
  explains why the change was made and anything a reviewer would not see from the diff.
- **No tool trailers.** No `Co-Authored-By`, `Claude-Session` or "Generated with" lines.
- **Small and focused.** Formatting only changes, dependency bumps and behaviour changes go
  in separate commits.
- **Checks first:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`.

## Continuous integration

`.github/workflows/ci.yml` runs on every PR and every push to `main`:

| Job | What it does |
| --- | --- |
| `lint` | `ruff check` and `ruff format --check` |
| `test (3.11)`, `test (3.12)` | `uv sync --locked --extra dev`, then `pytest` |

Tests must not need a GPU, a webcam or downloaded data. Keep heavy imports inside functions
(see `scripts/webcam_baseline.py`) so helpers can be tested on their own.

## Review with Claude

Comment `@claude` on a PR (or in a review, or an issue) with what you want, for example
`@claude review this PR for tracking bugs and missing tests`. The `Claude` workflow runs
Claude Code with `CLAUDE.md` as its instructions and replies in the thread. Only the repo
owner and collaborators can trigger it.

One time setup (either option):

1. Install the Claude GitHub app on this repo: <https://github.com/apps/claude>.
2. Add **one** repo secret under Settings > Secrets and variables > Actions:
   - `CLAUDE_CODE_OAUTH_TOKEN` to use a Claude Pro or Max subscription: run
     `claude setup-token` locally and paste the token.
   - `ANTHROPIC_API_KEY` to pay per use: create a key at console.anthropic.com.

Running `/install-github-app` inside Claude Code does both steps interactively.

## Versioning

[Semantic Versioning](https://semver.org/), adapted to the phase plan in `docs/PLAN.md`:

| Change | Version bump | Example |
| --- | --- | --- |
| A phase gate passes | minor | Phase 1 done: `0.1.0` to `0.2.0` |
| Bug fix or doc fix after a release | patch | `0.2.0` to `0.2.1` |
| Phase 7 showcase complete | major | `1.0.0` |

Phase N maps to `0.(N+1).0`, so the version tells you how far the project is. The version
lives in one place, `__version__` in `src/pathsense/__init__.py`; `pyproject.toml` reads it
from there.

## Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Every PR
that changes behaviour, adds a feature or changes the docs in a way users notice adds a
line under `## [Unreleased]`, grouped as Added, Changed, Fixed or Removed. Write it for
someone using the project, not as a list of commits.

## Releasing

When a phase gate passes (or a fix needs shipping):

1. On a branch, bump `__version__` in `src/pathsense/__init__.py`.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new
   empty `## [Unreleased]` above it, and update the compare links at the bottom.
   `tests/test_changelog.py` fails if these do not line up.
3. PR, CI green, merge.
4. Tag the merge commit on `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -s vX.Y.Z -m "vX.Y.Z: <phase or summary>"
   git push origin vX.Y.Z
   ```

5. `.github/workflows/release.yml` checks that the tag matches `__version__` and creates the
   GitHub release with that version's changelog section as the notes.

## Documentation

Docs are part of the change, in the same PR, not a follow up:

| When you... | Update |
| --- | --- |
| add or change a script, flag or module | `README.md` usage, and the module docstring |
| run an experiment or measure something | `NOTES.md` (what, the number, the decision) |
| pass a phase gate | `docs/PLAN.md` "Current status" and the README roadmap |
| change data sources or licenses | `docs/DATASETS.md` |
| change how work is done | this file and `CLAUDE.md` |
| change anything users notice | `CHANGELOG.md` under Unreleased |
