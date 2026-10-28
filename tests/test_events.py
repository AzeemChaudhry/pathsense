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
