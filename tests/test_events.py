import json

from pathsense.analytics import Event, EventLog, count_events, load_events


def test_the_log_round_trips_and_loads_in_time_order(tmp_path):
    path = tmp_path / "runs" / "clip.jsonl"  # the folder is created
    with EventLog(path) as log:
        log.write([Event(2.0, 50, 1, "enter")])
        log.write([])
        log.write([Event(1.5, 37, 2, "exit"), Event(3.0, 75, 3, "enter")])
    assert log.written == 3
    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {"t": 2.0, "frame": 50, "track_id": 1, "type": "enter"}
    assert load_events(path) == [
        Event(1.5, 37, 2, "exit"),  # confirmed later, but happened first
        Event(2.0, 50, 1, "enter"),
        Event(3.0, 75, 3, "enter"),
    ]


def test_each_write_reaches_the_file_before_close(tmp_path):
    # A run stopped by a crash keeps the events written so far.
    log = EventLog(tmp_path / "live.jsonl")
    log.write([Event(0.1, 1, 4, "enter")])
    assert load_events(log.path) == [Event(0.1, 1, 4, "enter")]
    log.close()


def test_counting_and_blank_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"t": 1, "frame": 1, "track_id": 1, "type": "enter"}\n\n')
    counts = count_events(load_events(path))
    assert counts["enter"] == 1 and counts["exit"] == 0


def test_a_last_line_cut_short_is_skipped(tmp_path):
    path = tmp_path / "killed.jsonl"
    path.write_text('{"t": 1.0, "frame": 25, "track_id": 1, "type": "enter"}\n{"t": 2.0, "fr')
    assert load_events(path) == [Event(1.0, 25, 1, "enter")]


def test_other_bad_lines_name_the_line_number(tmp_path):
    import pytest

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text('nonsense\n{"t": 1.0, "frame": 25, "track_id": 1, "type": "enter"}\n')
    with pytest.raises(ValueError, match="line 1: not JSON"):
        load_events(bad_json)
    bad_key = tmp_path / "key.jsonl"
    bad_key.write_text('{"t": 1.0, "frame": 25, "track_id": 1, "type": "enter", "x": 3}\n')
    with pytest.raises(ValueError, match="line 1: not an event"):
        load_events(bad_key)


def test_an_existing_log_is_replaced_and_ties_keep_frame_order(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text("old run\n")
    with EventLog(path) as log:
        log.write([Event(1.0, 26, 2, "exit"), Event(1.0, 25, 1, "enter")])
    assert [e.frame for e in load_events(path)] == [25, 26]


def test_wrong_field_types_name_the_line(tmp_path):
    import pytest

    path = tmp_path / "types.jsonl"
    good = '{"t": 1.0, "frame": 25, "track_id": 1, "type": "enter"}'
    path.write_text(good + '\n{"t": "x", "frame": null, "track_id": 1, "type": "enter"}\n')
    with pytest.raises(ValueError, match="line 2: not an event"):
        load_events(path)
