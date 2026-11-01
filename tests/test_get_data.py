import http.client
import importlib.util
import io
import urllib.error
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "get_data", Path(__file__).resolve().parents[1] / "train" / "get_data.py"
)
get_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_data)

RECORD = {
    "ID": "x",
    "gtboxes": [
        {"tag": "person", "fbox": [100, 50, 200, 400], "vbox": [100, 50, 200, 300], "extra": {}},
        {"tag": "person", "fbox": [-20, 300, 100, 400], "vbox": [0, 300, 80, 200], "extra": {}},
        {"tag": "mask", "fbox": [10, 10, 50, 50], "vbox": [10, 10, 50, 50], "extra": {"ignore": 1}},
        {
            "tag": "person",
            "fbox": [500, 10, 40, 80],
            "vbox": [500, 10, 40, 80],
            "extra": {"ignore": 1},
        },
        {"tag": "person", "fbox": [600, 100, 4, 4], "vbox": [600, 100, 4, 4], "extra": {}},
    ],
}


def test_drops_masks_ignored_and_tiny_boxes():
    lines = get_data.odgt_to_yolo(RECORD, 800, 600)
    assert len(lines) == 2


def test_box_is_normalised_and_clipped():
    first, second = (list(map(float, ln.split())) for ln in get_data.odgt_to_yolo(RECORD, 800, 600))
    assert first == pytest.approx([0, 0.25, 250 / 600, 0.25, 400 / 600], abs=1e-6)
    # second box starts at x=-20 and ends past the bottom edge; both get clipped
    cls, cx, cy, w, h = second
    assert cx - w / 2 >= 0 and cy + h / 2 <= 1.0
    assert abs(w - 80 / 800) < 1e-6 and abs(h - 300 / 600) < 1e-6


def test_openimages_maps_people_and_drops_groups_and_depictions():
    dets = [
        {"label": "Man", "bbox": [0.1, 0.2, 0.2, 0.4], "group": False, "depiction": False},
        {"label": "Woman", "bbox": [0.9, 0.5, 0.3, 0.3], "group": False, "depiction": False},
        {"label": "Person", "bbox": [0.0, 0.0, 1.0, 1.0], "group": True, "depiction": False},
        {"label": "Girl", "bbox": [0.3, 0.3, 0.1, 0.1], "group": False, "depiction": True},
        {"label": "Car", "bbox": [0.3, 0.3, 0.1, 0.1], "group": False, "depiction": False},
        {"label": "Boy", "bbox": [0.5, 0.5, 0.001, 0.2], "group": False, "depiction": False},
    ]
    lines = get_data.openimages_to_yolo(dets)
    assert len(lines) == 2
    man = list(map(float, lines[0].split()))
    assert man == pytest.approx([0, 0.2, 0.4, 0.2, 0.4], abs=1e-6)
    cls, cx, cy, w, h = map(float, lines[1].split())  # Woman box clipped at the right edge
    assert cx + w / 2 <= 1.0 + 1e-9 and w == pytest.approx(0.1, abs=1e-6)


CLASSES = {"/m/01g317": "Person", "/m/04yx4": "Man"}
HEADER = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,"
HEADER += "IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside"


def test_person_classes_reads_only_people(tmp_path):
    path = tmp_path / "classes.csv"
    path.write_text("/m/01g317,Person\n/m/0cmf2,Airplane\n/m/04yx4,Man\n")
    assert get_data.person_classes(path) == CLASSES


def test_boxes_from_the_csv_become_yolo_lines(tmp_path):
    path = tmp_path / "boxes.csv"
    rows = [
        HEADER,
        "a,xclick,/m/01g317,1,0.1,0.5,0.2,0.8,0,0,0,0,0",
        "a,xclick,/m/0cmf2,1,0,1,0,1,0,0,0,0,0",  # an airplane
        "b,xclick,/m/04yx4,1,0,1,0,1,0,0,1,0,0",  # a group
        "c,xclick,/m/04yx4,1,0,1,0,1,0,0,0,1,0",  # a drawing
        "d,xclick,/m/04yx4,1,0,1,0,1,0,0,-1,-1,0",  # not known: dropped
    ]
    path.write_text("\n".join(rows) + "\n")
    boxes = get_data.read_boxes(path, CLASSES)
    assert sorted(boxes) == ["a", "b", "c", "d"]
    assert get_data.openimages_to_yolo(boxes["a"]) == ["0 0.300000 0.500000 0.400000 0.600000"]
    assert boxes["b"][0]["group"] and boxes["c"][0]["depiction"]
    assert boxes["d"][0]["group"] and boxes["d"][0]["depiction"]


def test_only_usable_images_are_chosen_and_the_seed_fixes_them():
    good = {"label": "Man", "bbox": [0, 0, 1, 1], "group": False, "depiction": False}
    boxes = {f"i{k}": [good] for k in range(20)} | {"crowd": [good | {"group": True}]}
    first = get_data.choose_images(boxes, 5, seed=1)
    assert len(first) == 5 and "crowd" not in first
    assert first == get_data.choose_images(dict(reversed(boxes.items())), 5, seed=1)
    assert first != get_data.choose_images(boxes, 5, seed=2)
    assert len(get_data.choose_images(boxes, 500, seed=1)) == 20


class Reply(io.BytesIO):
    def __init__(self, body, size=None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if size is None else size)}


def test_fetch_retries_then_writes_the_whole_file(tmp_path, monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 503, "slow down", None, None)
        return Reply(b"short", size=9) if len(calls) == 2 else Reply(b"whole")

    monkeypatch.setattr(get_data.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(get_data.time, "sleep", lambda _: None)
    dst = tmp_path / "x.jpg"
    assert get_data.fetch("u", dst) and dst.read_bytes() == b"whole" and len(calls) == 3
    assert get_data.fetch("u", dst) and len(calls) == 3  # already there
    assert [f.name for f in tmp_path.iterdir()] == ["x.jpg"]


def test_fetch_survives_a_body_cut_short_and_a_part_left_by_a_killed_run(tmp_path, monkeypatch):
    class Cut(Reply):
        def read(self, *_):
            raise http.client.IncompleteRead(b"")

    replies = iter([Cut(b""), Reply(b"whole")])
    monkeypatch.setattr(get_data.urllib.request, "urlopen", lambda url, timeout: next(replies))
    monkeypatch.setattr(get_data.time, "sleep", lambda _: None)
    (tmp_path / "x.jpg.part").write_bytes(b"old")
    assert get_data.fetch("u", tmp_path / "x.jpg")
    assert [f.name for f in tmp_path.iterdir()] == ["x.jpg"]


def test_a_box_table_with_other_columns_is_refused(tmp_path):
    path = tmp_path / "boxes.csv"
    path.write_text("ImageID,LabelName\na,/m/01g317\n")
    with pytest.raises(SystemExit, match="no column"):
        get_data.read_boxes(path, CLASSES)


def test_a_bad_box_names_its_line(tmp_path):
    path = tmp_path / "boxes.csv"
    path.write_text(HEADER + "\na,xclick,/m/01g317,1,,0.5,0.2,0.8,0,0,0,0,0\n")
    with pytest.raises(SystemExit, match=":2: bad box"):
        get_data.read_boxes(path, CLASSES)


def test_an_earlier_selection_is_removed(tmp_path):
    (tmp_path / "keep.jpg").write_bytes(b"")
    (tmp_path / "old.jpg").write_bytes(b"")
    assert get_data.remove_others(tmp_path, {"keep": ["0 0 0 0 0"]}) == 1
    assert [f.name for f in tmp_path.iterdir()] == ["keep.jpg"]


def test_fetch_gives_up_on_a_missing_file_at_once(tmp_path, monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "gone", None, None)

    monkeypatch.setattr(get_data.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(get_data.time, "sleep", lambda _: None)
    assert not get_data.fetch("u", tmp_path / "x.jpg") and len(calls) == 1
    assert list(tmp_path.iterdir()) == []
