import re
from pathlib import Path

import pathsense

CHANGELOG = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pathsense.__version__)


def test_changelog_has_a_section_for_the_current_version():
    # The release workflow builds the GitHub release notes from this section.
    assert re.search(
        rf"^## \[{re.escape(pathsense.__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
        CHANGELOG,
        re.MULTILINE,
    )


def test_changelog_keeps_an_unreleased_section_on_top():
    sections = re.findall(r"^## \[(.+?)\]", CHANGELOG, re.MULTILINE)
    assert sections[0] == "Unreleased"
    assert sections[1] == pathsense.__version__
