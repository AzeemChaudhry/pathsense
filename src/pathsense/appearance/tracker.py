"""ByteTrack whose first matching pass can also use how people look.

Each mode changes the score that pairs a track with a confident box; the pairing itself,
the second pass and everything else stay ByteTrack's. `cos` below is the similarity of the
box to the track's remembered look, from -1 to 1.

| mode    | score of a track and a box                                               |
| ------- | ------------------------------------------------------------------------ |
| off     | overlap, exactly ByteTrack (tests/test_appearance.py checks)             |
| botsort | overlap, raised to the look's score if they overlap and look alike       |
| veto    | overlap, or 0 if cos is below `gate`                                     |
| blend   | (1 - blend) x overlap + blend x cos, 0 without overlap; `gate` as veto   |
| adaptive| overlap + (weight + boost) x cos, among pairs that pass the overlap test |

`adaptive` is Deep OC-SORT's adaptive weighting (Maggiolino et al., 2023, Eq 4 to 6): look
counts for more when a track's or box's best match is well ahead of its second best, so
two similar people lean on overlap. Unlike the paper, a pair must still overlap as much
as `minimum_iou_threshold` asks; look only ranks those pairs.

What a track remembers can be protected from bad boxes, which the paper and others find
matter most in occlusion:

- `dynamic` (the paper's Eq 3): a box's share of the memory falls with its confidence, to
  none at the first pass threshold. The paper gives no value for that threshold; here it
  is `high_conf_det_threshold`, since only first pass boxes reach the memory.
- `skip_overlap`: a box overlapping another this much shows two people, so it is not kept.

The library has no hook for this (`iou=` changes overlap only, and it dropped DeepSORT and
ReIDModel in 2.1.0), so `update` follows its ByteTrack.update step by step (trackers 2.6).
The test against the library catches it if a new release moves apart.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from trackers.core.bytetrack.tracklet import ByteTrackTracklet
from trackers.core.bytetrack.utils import _get_alive_tracklets
from trackers.utils.detections import default_confidences


class AppearanceByteTrack(ByteTrackTracker):
    search_space = {}  # the library's tuner reads this from every subclass

    def __init__(
        self,
        *,
        mode: str = "off",
        closeness: float = 0.25,
        proximity: float = 0.5,
        blend: float = 0.5,
        gate: float = 0.0,
        memory: float = 0.9,
        adaptive_weight: float = 0.75,
        adaptive_cap: float = 0.5,
        dynamic: bool = False,
        skip_overlap: float = 0.0,
        **params,
    ) -> None:
        super().__init__(**params)
        self.mode, self.closeness, self.proximity = mode, closeness, proximity
        self.blend, self.gate, self.memory = blend, gate, memory
        self.adaptive_weight, self.adaptive_cap = adaptive_weight, adaptive_cap
        self.dynamic, self.skip_overlap = dynamic, skip_overlap

    def _score(self, overlap: np.ndarray, tracks: list, vectors: np.ndarray) -> np.ndarray:
        """The overlap of tracks (rows) and boxes (columns), changed by this mode."""
        known = [getattr(t, "embedding", None) is not None for t in tracks]
        if self.mode == "off" or not any(known) or len(vectors) == 0:
            return overlap
        has = np.array(known)
        size = vectors.shape[1]
        looks = [t.embedding if k else np.zeros(size) for t, k in zip(tracks, known, strict=True)]
        cos = np.stack(looks).astype(np.float32) @ vectors.T  # both have length 1
        cos[~has] = np.nan  # a track with no look yet is judged by overlap alone
        if self.mode == "botsort":
            distance = (1 - cos) / 2
            distance[np.isnan(distance) | (distance > self.closeness)] = 1
            distance[(1 - overlap) > self.proximity] = 1
            return np.maximum(overlap, 1 - distance)
        if self.mode == "adaptive":
            alike = np.clip(np.nan_to_num(cos), 0, 1)
            weight = self.adaptive_weight + _boost(alike, self.adaptive_cap)
            score = overlap + weight * alike
            score[overlap < self.minimum_iou_threshold] = 0
            return score
        score = overlap.copy()
        if self.mode == "blend":
            score = (1 - self.blend) * overlap + self.blend * np.clip(np.nan_to_num(cos), 0, 1)
            score[~has] = overlap[~has]
            score[overlap <= 0] = 0
        if self.gate > 0:
            score[cos < self.gate] = 0  # NaN compares False: no look, no veto
        return score

    def _share(self, confidence: float | None) -> float:
        """How much of the memory stays when a box of this confidence joins it."""
        if not self.dynamic or confidence is None:
            return self.memory
        floor = self.high_conf_det_threshold
        trust = np.clip((confidence - floor) / (1 - floor), 0, 1)
        return self.memory + (1 - self.memory) * (1 - float(trust))

    def _remember(self, track, vector: np.ndarray | None, confidence: float | None = None) -> None:
        if vector is None:
            return
        old = getattr(track, "embedding", None)
        if old is None:
            track.embedding = vector.astype(np.float32)
            return
        keep = self._share(confidence)
        mixed = keep * old + (1 - keep) * vector
        track.embedding = (mixed / (np.linalg.norm(mixed) + 1e-12)).astype(np.float32)

    def update(self, detections: sv.Detections, frame=None, timestamp=None) -> sv.Detections:
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._detections_for_skipped_update(detections)
        if len(self.tracks) == 0 and len(detections) == 0:
            return _empty()
        vectors = detections.data.get("embedding") if len(detections) else None

        crowded = _overlapped(detections.xyxy, self.skip_overlap)

        def look(row: int) -> np.ndarray | None:
            if vectors is None or crowded[row]:
                return None
            return np.asarray(vectors[row], np.float32)

        kept: list[int] = []  # detection rows, in the order the library returns them
        ids: list[int] = []
        self._predict_tracklets(self.tracks, timing)
        budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        self._prune_lost_tracks(timing)
        boxes = detections.xyxy
        confidence = default_confidences(detections)
        high = np.where(confidence >= self.high_conf_det_threshold)[0]
        low = np.where(confidence < self.high_conf_det_threshold)[0]
        predicted = (
            np.array([t.get_state_bbox() for t in self.tracks]) if self.tracks else np.empty((0, 4))
        )

        score = self.iou.compute(predicted, boxes[high])
        if vectors is not None and len(self.tracks) and len(high):
            score = self._score(score, self.tracks, np.asarray(vectors[high], np.float32))
        matched, unmatched, unmatched_high = self._get_associated_indices(
            score, self.minimum_iou_threshold
        )
        for row, column in matched:
            self._confirm(self.tracks[row], boxes[high][column])
            self._remember(self.tracks[row], look(high[column]), float(confidence[high[column]]))
            kept.append(int(high[column]))
            ids.append(self.tracks[row].tracker_id)

        remaining = [self.tracks[i] for i in unmatched]
        rest = predicted[unmatched] if unmatched else np.empty((0, 4))
        matched, _, unmatched_low = self._get_associated_indices(
            self.iou.compute(rest, boxes[low]), self.minimum_iou_threshold
        )
        for row, column in matched:
            self._confirm(remaining[row], boxes[low][column])
            kept.append(int(low[column]))
            ids.append(remaining[row].tracker_id)

        for column in unmatched_low:
            kept.append(int(low[column]))
            ids.append(-1)
        for column in unmatched_high:
            detection = int(high[column])
            kept.append(detection)
            ids.append(-1)
            if float(confidence[detection]) >= self.track_activation_threshold:
                track = ByteTrackTracklet(
                    initial_bbox=boxes[detection], state_estimator_class=self.state_estimator_class
                )
                self._remember(track, look(detection))
                self.tracks.append(track)

        self.tracks = _get_alive_tracklets(
            tracklets=self.tracks,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_frames_without_update=self.maximum_frames_without_update,
            maximum_time_without_update=budget,
        )
        if not kept:
            return _empty()
        result = detections[np.array(kept)]
        result.tracker_id = np.array(ids, dtype=int)
        return result

    def _confirm(self, track, box: np.ndarray) -> None:
        track.update(box)
        confirmed = track.number_of_successful_consecutive_updates
        if confirmed >= self.minimum_consecutive_frames and track.tracker_id == -1:
            track.tracker_id = self._allocate_tracker_id()


def _boost(alike: np.ndarray, cap: float) -> np.ndarray:
    """Deep OC-SORT's Eq 4 and 5: the mean of each row's and column's lead of best over
    second best, capped. With one row or column there is no second best, so no lead."""

    def lead(values: np.ndarray) -> np.ndarray:  # one per row
        if values.shape[1] < 2:
            return np.zeros(len(values))
        top = np.partition(values, -2, axis=1)
        return np.minimum(top[:, -1] - top[:, -2], cap)

    return (lead(alike)[:, None] + lead(alike.T)[None, :]) / 2


def _overlapped(boxes: np.ndarray, limit: float) -> np.ndarray:
    """For each box, whether another overlaps it more than `limit`; 0: none are."""
    if limit <= 0 or len(boxes) < 2:
        return np.zeros(len(boxes), bool)
    overlap = sv.box_iou_batch(boxes, boxes)
    np.fill_diagonal(overlap, 0)
    return overlap.max(axis=1) > limit


def _empty() -> sv.Detections:
    result = sv.Detections.empty()
    result.tracker_id = np.array([], dtype=int)
    return result
