import pytest

from pathsense.config import (
    DEFAULT_CONFIG,
    Config,
    ConfigError,
    apply_override,
    load_config,
)


def test_default_file_loads_and_matches_code_defaults():
    config = load_config()
    assert DEFAULT_CONFIG.exists()
    assert config.detector == Config().detector  # "auto" device becomes None
    assert config.tracker.name == "bytetrack"
    assert config.tracker.params == {"track_activation_threshold": 0.6}


def test_one_override_swaps_the_detector_or_tracker():
    # The Phase 1 gate: one config line swaps a stage.
    assert load_config(overrides=["detector.name=rfdetr-nano"]).detector.name == "rfdetr-nano"
    assert load_config(overrides=["tracker.name=botsort"]).tracker.name == "botsort"


def test_override_values_are_parsed_as_yaml():
    config = load_config(
        overrides=[
            "detector.conf=0.25",
            "detector.imgsz=480",
            "tracker.params={lost_track_buffer: 60}",
        ]
    )
    assert config.detector.conf == 0.25 and config.detector.imgsz == 480
    assert config.tracker.params == {"lost_track_buffer": 60}


def test_file_values_are_read(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("detector:\n  name: yolo26s\n  device: cpu\ntracker:\n  name: ocsort\n")
    config = load_config(path)
    assert config.detector.name == "yolo26s" and config.detector.device == "cpu"
    assert config.tracker.name == "ocsort"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("detector.nmae=x", "Unknown keys in [detector]"),
        ("detectr.name=x", "Unknown config sections"),
        ("detector.name", "section.key=value"),
        ("name=x", "section.key=value"),
    ],
)
def test_typos_are_errors_not_silent_defaults(override, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_config(overrides=[override])


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_apply_override_creates_nested_keys():
    raw = {}
    apply_override(raw, "analytics.line=[[0, 0], [10, 10]]")
    assert raw == {"analytics": {"line": [[0, 0], [10, 10]]}}


def test_camera_section_defaults_and_overrides():
    config = load_config()
    assert config.camera.calibrate and config.camera.dynamic_framerate is False
    config = load_config(overrides=["camera.max_width=640", "camera.dynamic_framerate=null"])
    assert config.camera.max_width == 640 and config.camera.dynamic_framerate is None


def test_analytics_line_is_read_and_checked():
    assert load_config().analytics.line is None  # off by default
    config = load_config(overrides=["analytics.line=[[0, 675], [1920, 675]]"])
    assert config.analytics.line == ((0.0, 675.0), (1920.0, 675.0))
    assert config.analytics.line_margin == 5


@pytest.mark.parametrize(
    "value", ["[[0, 675]]", "[[0, 675], [1920]]", "[[a, 1], [2, 3]]", "[[5, 5], [5, 5]]", "7"]
)
def test_bad_lines_are_config_errors(value):
    with pytest.raises(ConfigError, match="analytics.line"):
        load_config(overrides=[f"analytics.line={value}"])


def test_analytics_typos_are_errors_now():
    with pytest.raises(ConfigError, match=r"Unknown keys in \[analytics\]"):
        load_config(overrides=["analytics.lien=[[0, 1], [2, 3]]"])
