import numpy as np
import pytest
from conftest import detections

from pathsense.analytics import Event
from pathsense.eval import Identity, ground_truth, match_crossings, tracking_metrics, write_mot

FPS = 10.0


def people(*boxes):
    """Tracked boxes as (track_id, x1, y1, x2, y2)."""
    tracked = detections([b[1:] for b in boxes])
    tracked.tracker_id = np.array([b[0] for b in boxes])
    return tracked


def test_a_crossing_is_right_only_for_the_person_who_crossed():
    # Our track 5 covers labelled person 1 in frames 0-9, then the ID slides onto person 2.
    identity = Identity()
    for frame in range(20):
        label = 1 if frame < 10 else 2
        identity.add(frame, people((5, 0, 0, 10, 40)), people((label, 0, 0, 10, 40)))
    truth = [Event(0.2, 2, 1, "exit"), Event(1.2, 12, 2, "exit")]
    ours = [Event(0.3, 3, 5, "exit"), Event(1.1, 11, 5, "enter")]  # the swap looks like a U-turn
    score = match_crossings(ours, truth, identity, FPS)
    assert (score.tp["exit"], score.fp["enter"], score.fn["exit"]) == (1, 1, 1)
    assert score.right == [ours[0]] and score.wrong == [ours[1]] and score.missed == [truth[1]]
    assert score.precision("exit") == 1.0 and score.recall("exit") == 0.5


def test_a_crossing_too_far_in_time_does_not_match():
    identity = Identity()
    for frame in range(100):
        identity.add(frame, people((5, 0, 0, 10, 40)), people((1, 0, 0, 10, 40)))
    ours, truth = [Event(0.0, 0, 5, "enter")], [Event(5.0, 50, 1, "enter")]
    score = match_crossings(ours, truth, identity, FPS)
    assert (score.tp["enter"], score.fp["enter"], score.fn["enter"]) == (0, 1, 1)


def test_scores_add_up_across_scenes():
    identity = Identity()
    identity.add(0, people((5, 0, 0, 10, 40)), people((1, 0, 0, 10, 40)))
    one = match_crossings([Event(0, 0, 5, "exit")], [Event(0, 0, 1, "exit")], identity, FPS)
    both = one + one
    assert both.tp["exit"] == 2 and len(both.right) == 2 and both.f1() == 1.0


def test_the_labels_scored_as_tracks_are_perfect(tmp_path):
    rows = []
    for frame in range(1, 31):  # two people walking right, one standing still (class 7)
        rows.append(f"{frame},1,{10 + frame},10,20,60,1,1,1.0")
        rows.append(f"{frame},2,{200 - frame},50,20,60,1,1,0.8")
        rows.append(f"{frame},3,400,50,20,60,0,7,1.0")
    labels = tmp_path / "gt.txt"
    labels.write_text("\n".join(rows) + "\n")
    frames = ground_truth(labels)
    assert set(frames[0].tracker_id) == {1, 2, 3}  # class 7 is a person for counting
    write_mot(tmp_path / "tracks.txt", sorted(frames.items()))
    metrics = tracking_metrics(labels, tmp_path / "tracks.txt")
    assert metrics["MOTA"] == pytest.approx(1.0) and metrics["IDF1"] == pytest.approx(1.0)
    assert metrics["HOTA"] == pytest.approx(1.0) and metrics["IDSW"] == 0


def test_ground_truth_needs_nine_columns(tmp_path):
    bad = tmp_path / "gt.txt"
    bad.write_text("1,1,10,10,20,60\n")
    with pytest.raises(ValueError, match="9 values"):
        ground_truth(bad)
