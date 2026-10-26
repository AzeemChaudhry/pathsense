"""Guards for values that live in more than one file and must agree."""

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def locked_version(package: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return next(p["version"] for p in lock["package"] if p["name"] == package)


def test_pre_commit_ruff_matches_uv_lock():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    rev = next(r["rev"] for r in config["repos"] if r["repo"].endswith("/ruff-pre-commit"))
    assert rev == f"v{locked_version('ruff')}", "bump the ruff-pre-commit rev with uv.lock"


def test_workflows_pin_setup_uv_to_one_release():
    # setup-uv has no floating major tag, so each use names an exact release; keep them equal.
    pins = set()
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        pins.update(re.findall(r"astral-sh/setup-uv@(\S+)", workflow.read_text()))
    assert len(pins) == 1, pins


def test_actions_that_receive_secrets_are_pinned_to_a_commit():
    # A third party action that sees a secret must not change under us when its tag moves.
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = workflow.read_text()
        if "secrets." not in text:
            continue
        for action in re.findall(r"uses:\s*(\S+)", text):
            if action.startswith("actions/"):
                continue  # GitHub's own actions
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), (workflow.name, action)
