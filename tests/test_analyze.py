"""The scene analyzer: a scan, a description, and the first rule that fits."""

import numpy as np
import pytest
from conftest import detections

from pathsense.config import ConfigError, DetectorConfig
from pathsense.tune.analyze import Analysis, Rule, Rules, analyze, choose, load_rules, scan


def rule(name, below=None, at_least=None, overrides=()):
    return Rule(name, "because", tuple(overrides), below or {}, at_least or {})


RULES = Rules(
    DetectorConfig(),
    4,
    (
        rule("small people", below={"height_small": 100}, overrides=["detector.imgsz=1920"]),
        rule("a crowd", at_least={"people": 2}),
        rule("default"),
    ),
)


class Seen:
    """Counts frames and finds `people` boxes, each `height` tall."""

    def __init__(self, people=3, height=200):
        self.people, self.height, self.frames = people, height, 0

    def __call__(self, image):
        self.frames += 1
        boxes = [[10 + 60 * k, 5, 50 + 60 * k, 5 + self.height] for k in range(self.people)]
        return detections(boxes)


def test_a_scan_looks_at_the_frames_it_was_asked_for(tiny_video):
    seen = Seen()
    saved = scan(tiny_video, seen, 4)
    assert seen.frames == 4 and len(saved) == 4 and len(saved.xyxy) == 12
    assert saved.size == (64, 48)


def test_the_first_rule_that_fits_wins(tiny_video):
    assert analyze(tiny_video, RULES, lambda c: Seen(height=50)).rule.name == "small people"
    assert analyze(tiny_video, RULES, lambda c: Seen()).rule.name == "a crowd"
    assert analyze(tiny_video, RULES, lambda c: Seen(people=1)).rule.name == "default"


def test_nobody_in_view_falls_through_to_the_last_rule(tiny_video):
    found = analyze(tiny_video, RULES, lambda c: Seen(people=0))
    assert found.rule.name == "default"
    assert choose(found.profile, RULES) is found.rule


def test_the_report_gives_the_settings(tiny_video):
    found = analyze(tiny_video, RULES, lambda c: Seen(height=50))
    assert isinstance(found, Analysis) and "--set detector.imgsz=1920" in found.report()


def test_the_shipped_rules_load_and_always_fit():
    rules = load_rules()
    assert rules.frames > 0 and not (rules.rules[-1].below or rules.rules[-1].at_least)


@pytest.mark.parametrize(
    "text",
    [
        "scan: {name: yolo26n}\nframes: 4\nrules: []\n",
        "scan: {name: yolo26n}\nframes: 4\nrules:\n  - {name: a, why: b, below: {people: 3}}\n",
        "scan: {name: yolo26n}\nframes: 4\nrules:\n"
        "  - {name: a, why: b, below: {nonsense: 3}}\n  - {name: c, why: d}\n",
    ],
)
def test_bad_rules_are_errors(tmp_path, text):
    path = tmp_path / "rules.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_rules(path)


def test_scan_handles_an_empty_detector_result_shape(tiny_video):
    saved = scan(tiny_video, Seen(people=0), 3)
    assert np.array_equal(saved.counts, [0, 0, 0])
