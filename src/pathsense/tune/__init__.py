"""Scene tuning: choose a scene's tracker settings from a clip whose truth is known.

    pathsense tune configs/scenes/subway.yaml

    from pathsense.tune import tune
    report = tune("configs/scenes/subway.yaml")
    print(report.markdown())

The detector runs over the video once (cache); every candidate setup then replays the
saved detections (replay), is scored against the scene's `truth:` block (truth), and the
report describes the scene (profile) and what is left wrong (run).
"""

from pathsense.tune.cache import SavedDetections, cached_detections, detect_video
from pathsense.tune.profile import SceneProfile, scene_profile
from pathsense.tune.replay import NO_DETECTOR, Replay, replay
from pathsense.tune.run import CANDIDATES, Candidate, TuneReport, load_candidates, tune
from pathsense.tune.truth import Truth, scene_truth

__all__ = [
    "CANDIDATES",
    "NO_DETECTOR",
    "Candidate",
    "Replay",
    "SavedDetections",
    "SceneProfile",
    "Truth",
    "TuneReport",
    "cached_detections",
    "detect_video",
    "load_candidates",
    "replay",
    "scene_profile",
    "scene_truth",
    "tune",
]
