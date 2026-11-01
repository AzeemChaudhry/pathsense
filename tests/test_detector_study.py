"""The model comparison must not score models on different scenes, or rank noise."""

import csv
import runpy

import numpy as np

study = runpy.run_path("scripts/detector_study/__main__.py", run_name="study")


def write(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, study["MODEL_FIELDS"])
        writer.writeheader()
        for name, scene, tp in rows:
            row = dict(detector=name, imgsz=1920, confidence=0.1, candidate="default", scene=scene)
            row |= dict(
                group="mot", tp=tp, wrong=0, missed=10 - tp, ids=0, line_error=0, zone_error=0, ms=1
            )
            writer.writerow(row)


def test_a_model_that_failed_part_way_is_named_and_cuts_the_scenes(tmp_path, capsys):
    path = tmp_path / "models.csv"
    rows = [("full", f"s{k}", 5) for k in range(4)] + [("full", "hard", 1)]
    write(path, rows + [("failed", f"s{k}", 5) for k in range(4)])
    study["compare"](path, [], ["full", "failed", "never"])
    out = capsys.readouterr().out
    assert "4 scenes; short: failed (4 of 5), never (0 of 5)" in out
    assert "tie" in out  # both score the same on the shared scenes


def test_rows_of_a_dropped_model_are_left_out(tmp_path, capsys):
    path = tmp_path / "models.csv"
    write(path, [("kept", "s1", 5), ("kept", "s2", 6), ("dropped", "s1", 5)])
    study["compare"](path, [], ["kept"])
    assert "2 scenes; short: none" in capsys.readouterr().out


def test_models_with_no_shared_scenes_are_not_compared(tmp_path, capsys):
    path = tmp_path / "models.csv"
    write(path, [("a", "s1", 5), ("b", "s2", 5)])
    study["compare"](path, [], ["a", "b"])
    assert "too few shared scenes" in capsys.readouterr().out


def test_comparing_nothing_says_so(tmp_path, capsys):
    path = tmp_path / "models.csv"
    write(path, [])
    study["compare"](path, [], ["full"])
    assert "nothing scored yet" in capsys.readouterr().out


def test_only_a_clear_paired_gap_counts_as_a_difference():
    a = np.linspace(0.5, 0.7, 40)
    assert study["paired"](a + 0.01, a).endswith("tie")  # under TIE
    assert study["paired"](a + 0.1, a).endswith("differs")
    noisy = np.resize([0.3, -0.25], 40)
    assert study["paired"](a + noisy, a).endswith("tie")  # the interval crosses 0


def test_a_band_keeps_its_own_setting_and_each_scene_is_left_out(capsys):
    found = []
    base = dict(candidate="default", wrong=0, ids=1, line_error=0, zone_error=0)
    for name in ("a", "b"):
        for confidence, crowd, sparse in ((0.1, 8, 1), (0.3, 2, 9)):
            hits = {"c1": crowd, "c2": crowd, "s1": sparse, "s2": sparse}
            for scene, tp in hits.items():
                row = dict(detector=name, confidence=confidence, scene=scene, tp=tp, missed=10 - tp)
                found.append(base | row)
    study["band_report"](found, {"crowd": {"c1", "c2"}, "sparse": {"s1", "s2"}})
    crowd, sparse = capsys.readouterr().out.split("sparse: ")
    assert "0.1 default" in crowd and "0.3 default" in sparse
    assert "0.889" in crowd  # left out score, at the crowd's setting
