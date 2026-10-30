"""Choose a scene's tracker settings from a clip whose truth is known.

    report = tune("configs/scenes/subway.yaml")
    print(report.markdown())

1. Run the detector over the scene's video once (pathsense.tune.cache).
2. Describe the scene: people, size, crowding, speed (pathsense.tune.profile).
3. Replay every candidate in configs/tune/candidates.yaml over the same detections, and
   the scene as it is ("current").
4. Score each against the scene's `truth:` block: every crossing one by one with labels,
   the totals with hand counts.
5. Keep the current setup unless a candidate scores strictly better.
6. Say what is left, and whose it is: the detector's (people never detected), the
   tracker's (detected, but the track was lost or split), or the counting rules' (errors
   even with perfect tracks).

Nothing is written to the scene file: the report shows the block to paste, so the file
keeps its comments and a person decides what changes.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from pathsense.analytics import Event
from pathsense.config import Config, ConfigError, apply_override, load_config, read_yaml
from pathsense.config import scene_source as video_of
from pathsense.eval import (
    CrossingScore,
    Identity,
    explain_misses,
    ground_truth,
    label_events,
    match_crossings,
    misses_table,
)
from pathsense.eval.crossings import DIRECTIONS
from pathsense.eval.misses import DETECTED
from pathsense.pipeline import Pipeline
from pathsense.tune.cache import CACHE, SavedDetections, cached_detections
from pathsense.tune.profile import SceneProfile, scene_profile
from pathsense.tune.replay import NO_DETECTOR, Replay, replay
from pathsense.tune.truth import Truth, count_key, scene_truth

CANDIDATES = Path(__file__).resolve().parents[3] / "configs" / "tune" / "candidates.yaml"


@dataclass(frozen=True)
class Candidate:
    name: str
    why: str
    overrides: tuple[str, ...]  # `section.key=value`, laid over the scene file


CURRENT = Candidate("current", "the scene file as it is", ())


def load_candidates(path: Path | str = CANDIDATES) -> list[Candidate]:
    raw = read_yaml(Path(path)).get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: expected a non empty list under candidates")
    candidates = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"name", "why", "set"}:
            raise ConfigError(f"{path}: candidates[{i}] must have exactly name, why and set")
        overrides = item["set"]
        if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
            raise ConfigError(f"{path}: candidates[{i}].set must be a list of section.key=value")
        candidates.append(Candidate(str(item["name"]), str(item["why"]), tuple(overrides)))
    names = [c.name for c in candidates]
    if len(set(names)) != len(names) or CURRENT.name in names:
        raise ConfigError(f"{path}: candidate names must be unique and not {CURRENT.name!r}")
    return candidates


@dataclass(frozen=True)
class Score:
    counts: dict[str, int]  # ours, by Truth.counts key
    truth: dict[str, int]  # the true counts, from the labels or by hand
    crossings: CrossingScore | None  # with labels: every crossing matched one by one
    ids: int
    track_ms: float

    @property
    def count_error(self) -> int:
        return sum(abs(self.counts.get(key, 0) - n) for key, n in self.truth.items())

    @property
    def objective(self) -> tuple[float, ...]:
        """Higher is better: crossing F1 then count error with labels, count error alone
        with hand counts."""
        if self.crossings is not None:
            return (round(self.crossings.f1(), 6), -self.count_error)
        return (-self.count_error,)


@dataclass(frozen=True)
class Trial:
    candidate: Candidate
    score: Score


def counts_of(events: Iterable[Event]) -> dict[str, int]:
    keys = (count_key(e.type, e.zone) for e in events)
    return dict(Counter(key for key in keys if key))


@dataclass(frozen=True)
class _Scene:
    """Everything a trial needs, loaded once per process."""

    path: Path
    config: Path | None
    saved: SavedDetections
    labels: dict | None
    true_events: list[Event] | None
    true_counts: dict[str, int]  # by Truth.counts key

    def candidate_config(self, candidate: Candidate) -> Config:
        return load_config(self.config, list(candidate.overrides), self.path)

    def run(self, candidate: Candidate) -> tuple[Score, Replay]:
        config = self.candidate_config(candidate)
        pipeline = Pipeline.from_config(config, self.saved.fps, detector=NO_DETECTOR)
        result = replay(pipeline, self.saved)
        crossings = None
        if self.labels is not None and self.true_events is not None:
            identity = Identity()
            for index, people in result.tracks.items():
                identity.add(index, people, self.labels.get(index))
            ours = [e for e in result.events if e.type in DIRECTIONS]
            truth = [e for e in self.true_events if e.type in DIRECTIONS]
            crossings = match_crossings(ours, truth, identity, self.saved.fps)
        score = Score(
            counts_of(result.events), self.true_counts, crossings, result.ids, result.track_ms
        )
        return score, result


def _truth(scene: Path) -> Truth:
    truth = scene_truth(scene)
    if truth is None:
        raise ConfigError(
            f"{scene} has no truth: block; add hand counts or labels (pathsense.tune.truth)"
        )
    return truth


@cache  # per process and per file: tune() makes the saved detections before any loading
def _load(scene: Path, config: Path | None, saved: Path) -> _Scene:
    truth = _truth(scene)
    detections = SavedDetections.load(saved)
    labels = true_events = None
    counts = truth.counts
    if truth.labels is not None:
        labels = ground_truth(truth.labels)
        analytics = load_config(config, scene=scene).analytics
        frames = zip(detections.index.tolist(), detections.timestamp.tolist(), strict=True)
        true_events = label_events(analytics, labels, frames, detections.fps)
        zones = [f"zone:{z.name}" for z in analytics.zones]
        keys = [*(DIRECTIONS if analytics.line else ()), *zones]
        found = counts_of(true_events)
        counts = {key: found.get(key, 0) for key in keys}
    return _Scene(scene, config, detections, labels, true_events, counts)


def _trial(job: tuple[Path, Path | None, Path, Candidate]) -> Trial:
    scene, config, saved, candidate = job
    return Trial(candidate, _load(scene, config, saved).run(candidate)[0])


@dataclass(frozen=True)
class Diagnosis:
    """What the chosen setup still gets wrong, and whose it is."""

    misses: Counter  # pathsense.eval.explain_misses table
    rules_wrong: int  # crossings counted wrong even with perfect (label) tracks
    rules_missed: int


@dataclass(frozen=True)
class TuneReport:
    scene: Path
    profile: SceneProfile
    trials: list[Trial]  # current first, then the candidates in file order
    chosen: Trial
    diagnosis: Diagnosis | None  # with labels only

    def chosen_yaml(self) -> str:
        """The chosen candidate's overrides as a scene file block ('' for current)."""
        raw: dict = {}
        for override in self.chosen.candidate.overrides:
            apply_override(raw, override)
        return yaml.safe_dump(raw, sort_keys=False, default_flow_style=None) if raw else ""

    def markdown(self) -> str:
        keys = list(self.chosen.score.truth)
        labelled = self.chosen.score.crossings is not None
        head = ["candidate"] + (["F1", "wrong", "missed"] if labelled else [])
        head += [f"{k} (true {self.chosen.score.truth[k]})" for k in keys]
        head += ["count error", "IDs", "track ms"]
        lines = [
            f"# Tuning {self.scene.stem}",
            "",
            "## The scene",
            "",
            *(f"- {line}" for line in self.profile.lines()),
            "",
            "## Candidates",
            "",
            "| " + " | ".join(head) + " |",
            "| --- |" + " --- |" * (len(head) - 1),
        ]
        for trial in self.trials:
            s, mark = trial.score, " **chosen**" if trial is self.chosen else ""
            row = [trial.candidate.name + mark]
            if s.crossings is not None:
                wrong, missed = sum(s.crossings.fp.values()), sum(s.crossings.fn.values())
                row += [f"{s.crossings.f1():.3f}", str(wrong), str(missed)]
            row += [str(s.counts.get(k, 0)) for k in keys]
            row += [str(s.count_error), str(s.ids), f"{s.track_ms:.1f}"]
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", *(f"- **{t.candidate.name}**: {t.candidate.why}" for t in self.trials)]
        lines += ["", "## Chosen", ""]
        block = self.chosen_yaml()
        if block:
            lines += [
                f"**{self.chosen.candidate.name}** scores better than the scene as it is. "
                "To keep it, put this in the scene file:",
                "",
                "```yaml",
                block.rstrip(),
                "```",
            ]
        else:
            lines.append("The scene file as it is: no candidate scores strictly better.")
        lines += ["", "## What is left", "", *self._left()]
        return "\n".join(lines) + "\n"

    def _left(self) -> list[str]:
        s = self.chosen.score
        if self.diagnosis is None:
            off = [
                f"{key} {s.counts.get(key, 0)} against {n}"
                for key, n in s.truth.items()
                if s.counts.get(key, 0) != n
            ]
            return [
                ("Off: " + "; ".join(off) + "." if off else "Every count matches."),
                "",
                "These are totals: a false crossing and a missed one cancel out. Labels "
                "(truth.labels) score every crossing and say why one was missed.",
            ]
        d, c = self.diagnosis, s.crossings
        assert c is not None
        missed = sum(c.fn.values())
        lost = {"no track", "starts or ends", "ID changes"}
        tracking = sum(n for (t, det), n in d.misses.items() if det == DETECTED[0] and t in lost)
        never = sum(n for (_, det), n in d.misses.items() if det == DETECTED[2])
        return [
            f"{sum(c.fp.values())} wrong crossings and {missed} missed, out of "
            f"{sum(s.truth.get(k, 0) for k in DIRECTIONS)} true.",
            "",
            f"- **Detector (scene limit for tuning):** {never} missed crossings by people "
            "never detected around the crossing.",
            f"- **Tracker:** {tracking} by people detected in at least half the frames, but "
            "with no track, a track starting or ending at the line, or an ID change there.",
            f"- **Counting rules:** with perfect tracks (the labels), the rules alone give "
            f"{d.rules_wrong} wrong and {d.rules_missed} missed.",
            "",
            "Every missed crossing, by what the tracks did and whether the person was detected:",
            "",
            misses_table(d.misses).rstrip(),
        ]


def _diagnose(scene: _Scene, result: Replay, crossings: CrossingScore) -> Diagnosis:
    assert scene.labels is not None and scene.true_events is not None
    detections = {frame.index: boxes for frame, boxes in scene.saved}
    misses = explain_misses(
        crossings.missed, crossings.wrong, scene.labels, detections, result.tracks, scene.saved.fps
    )
    analytics = load_config(scene.config, scene=scene.path).analytics
    frames = list(zip(scene.saved.index.tolist(), scene.saved.timestamp.tolist(), strict=True))
    perfect = label_events(analytics, scene.labels, frames, scene.saved.fps, rules=True)
    identity = Identity()
    for index, people in scene.labels.items():
        identity.add(index, people, people)
    rules = match_crossings(
        [e for e in perfect if e.type in DIRECTIONS],
        [e for e in scene.true_events if e.type in DIRECTIONS],
        identity,
        scene.saved.fps,
    )
    return Diagnosis(misses, sum(rules.fp.values()), sum(rules.fn.values()))


def tune(
    scene: Path | str,
    candidates: list[Candidate] | None = None,
    config: Path | str | None = None,
    cache_folder: Path | str = CACHE,
    refresh: bool = False,
    jobs: int = 1,
) -> TuneReport:
    """Try the candidates on `scene` and report; see the module docstring."""
    scene, config = Path(scene), Path(config) if config else None
    video = video_of(scene)
    if video is None:
        raise ConfigError(f"{scene} names no source: video to tune on")
    _truth(scene)  # before the detector runs, so a missing block fails straight away
    detector = load_config(config, scene=scene).detector
    cached_detections(video, detector, cache_folder, refresh)
    saved_path = Path(cache_folder) / f"{Path(video).stem}.npz"
    everyone = [CURRENT, *(load_candidates() if candidates is None else candidates)]
    jobs_list = [(scene, config, saved_path, c) for c in everyone]
    if jobs > 1:
        with ProcessPoolExecutor(min(jobs, len(everyone))) as pool:
            trials = list(pool.map(_trial, jobs_list))
    else:
        trials = [_trial(job) for job in jobs_list]
    chosen = max(trials, key=lambda t: t.score.objective)  # the first best: current on ties

    loaded = _load(scene, config, saved_path)
    current = loaded.run(CURRENT)[1]
    diagnosis = None
    if loaded.labels is not None:
        crossings = chosen.score.crossings
        assert crossings is not None
        best = current if chosen.candidate == CURRENT else loaded.run(chosen.candidate)[1]
        diagnosis = _diagnose(loaded, best, crossings)
    return TuneReport(scene, scene_profile(loaded.saved, current.tracks), trials, chosen, diagnosis)


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


__all__ = ["CANDIDATES", "CURRENT", "Candidate", "Score", "TuneReport", "load_candidates", "tune"]
