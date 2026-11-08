import argparse
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


def _people(n, height):
    return [
        {
            "label": "Person",
            "bbox": [0.1 * k, 0.1, 0.05, height],
            "group": False,
            "depiction": False,
        }
        for k in range(n)
    ]


def test_crowded_images_fill_their_share_and_the_rest_is_random():
    boxes = {f"crowd{k}": _people(5, 0.1) for k in range(4)}
    boxes |= {f"few{k}": _people(4, 0.1) for k in range(10)}  # one short of crowded
    boxes |= {f"big{k}": _people(6, 0.5) for k in range(10)}  # people too large
    for seed in range(20):
        chosen = get_data.choose_images(boxes, 6, seed=seed, crowded=0.5)
        assert len(chosen) == 6 and sum(k.startswith("crowd") for k in chosen) >= 3
        every = get_data.choose_images(boxes, 6, seed=seed, crowded=1.0)
        assert len(every) == 6  # only 4 crowded exist, so 2 are drawn from the rest
        assert {f"crowd{k}" for k in range(4)} <= set(every)
    assert get_data.choose_images(boxes, 6, seed=1) == get_data.choose_images(
        boxes, 6, seed=1, crowded=0.0
    )


def test_group_boxes_are_the_person_crowds_only():
    dets = [
        {"label": "Person", "bbox": [0.1, 0.2, 0.3, 0.4], "group": True, "depiction": False},
        {"label": "Man", "bbox": [0.5, 0.5, 0.1, 0.1], "group": False, "depiction": False},
        {"label": "Woman", "bbox": [0.0, 0.0, 0.2, 0.2], "group": False, "depiction": True},
    ]
    assert get_data.group_boxes(dets) == [pytest.approx((0.1, 0.2, 0.4, 0.6))]


def test_a_crowd_is_greyed_out_but_a_labelled_person_in_it_is_kept(tmp_path):
    from PIL import Image

    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    Image.new("RGB", (100, 100), (200, 30, 30)).save(src, quality=100)
    person = "0 0.300000 0.300000 0.200000 0.200000"  # 20 to 40 px each way, inside the crowd
    get_data.grey_out(src, dst, [(0.1, 0.1, 0.6, 0.6)], [person])
    with Image.open(dst) as out:
        grey, kept, outside = out.getpixel((50, 50)), out.getpixel((30, 30)), out.getpixel((80, 80))
    assert all(abs(v - get_data.GREY) <= 3 for v in grey)
    assert abs(kept[0] - 200) <= 6 and abs(outside[0] - 200) <= 6


def test_a_crowded_share_outside_0_to_1_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(get_data, "DATA", tmp_path)
    args = argparse.Namespace(max_train=2, max_val=2, seed=0, crowded_share=1.5, grey_groups=True)
    with pytest.raises(SystemExit, match="share"):
        get_data.cmd_openimages(args)


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


def test_a_short_row_names_its_line(tmp_path):
    path = tmp_path / "boxes.csv"
    path.write_text(HEADER + "\na,xclick,/m/01g317,1,0.1,0.5,0.2,0.8,0,0\n")
    with pytest.raises(SystemExit, match=":2: short row"):
        get_data.read_boxes(path, CLASSES)


def test_a_reply_of_unknown_length_is_not_kept(tmp_path, monkeypatch):
    def urlopen(url, timeout):
        reply = Reply(b"maybe cut")
        reply.headers = {}
        return reply

    monkeypatch.setattr(get_data.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(get_data.time, "sleep", lambda _: None)
    assert not get_data.fetch("u", tmp_path / "x.csv")
    assert list(tmp_path.iterdir()) == []


def test_an_earlier_selection_is_removed(tmp_path):
    (tmp_path / "keep.jpg").write_bytes(b"")
    (tmp_path / "old.jpg").write_bytes(b"")
    assert get_data.remove_others(tmp_path, {"keep": ["0 0 0 0 0"]}) == 1
    assert [f.name for f in tmp_path.iterdir()] == ["keep.jpg"]


def _run_openimages(tmp_path, monkeypatch, missing):
    def fake_fetch(url, dst, tries=get_data.TRIES):
        if url.endswith(".csv"):
            return True
        if dst.stem in missing:
            return False
        dst.write_bytes(b"jpg")
        return True

    chosen = {"new": ["0 0 0 0 0"], "other": ["0 0 0 0 0"]}
    monkeypatch.setattr(get_data, "DATA", tmp_path)
    monkeypatch.setattr(get_data, "fetch", fake_fetch)
    monkeypatch.setattr(get_data, "person_classes", lambda path: {"/m/p": "Person"})
    monkeypatch.setattr(get_data, "read_boxes", lambda path, classes: {})
    monkeypatch.setattr(get_data, "choose_images", lambda boxes, limit, seed, crowded: chosen)
    args = argparse.Namespace(max_train=2, max_val=2, seed=0, crowded_share=0.0, grey_groups=True)
    get_data.cmd_openimages(args)


def _earlier_files(tmp_path):
    images = tmp_path / "openimages" / "images" / "train"
    labels = tmp_path / "openimages" / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "old.jpg").write_bytes(b"jpg")
    (images / get_data.KIND).write_text(f"grey {get_data.MAX_GREY}\n")
    (labels / "old.txt").write_text("0 0 0 0 0\n")
    return images, labels


def test_earlier_files_go_once_every_download_worked(tmp_path, monkeypatch):
    images, labels = _earlier_files(tmp_path)
    _run_openimages(tmp_path, monkeypatch, set())
    assert sorted(path.stem for path in images.glob("*.jpg")) == ["new", "other"]
    assert sorted(path.stem for path in labels.iterdir()) == ["new", "other"]


@pytest.mark.parametrize("missing", [{"new"}, {"new", "other"}])
def test_a_failed_download_keeps_earlier_files_and_fails_the_run(tmp_path, monkeypatch, missing):
    images, labels = _earlier_files(tmp_path)
    with pytest.raises(SystemExit, match="incomplete: train, val"):
        _run_openimages(tmp_path, monkeypatch, missing)
    assert (images / "old.jpg").exists() and (labels / "old.txt").exists()


def test_fetch_gives_up_on_a_missing_file_at_once(tmp_path, monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "gone", None, None)

    monkeypatch.setattr(get_data.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(get_data.time, "sleep", lambda _: None)
    assert not get_data.fetch("u", tmp_path / "x.jpg") and len(calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("grey", [True, False])
def test_the_download_greys_a_crowd_unless_told_to_keep_it(tmp_path, monkeypatch, grey):
    from PIL import Image

    def fake_fetch(url, dst, tries=get_data.TRIES):
        if not url.endswith(".csv"):
            Image.new("RGB", (100, 100), (200, 30, 30)).save(dst, quality=100)
        return True

    crowd = {"label": "Person", "bbox": [0.5, 0.5, 0.4, 0.4], "group": True, "depiction": False}
    alone = _people(1, 0.2)
    monkeypatch.setattr(get_data, "DATA", tmp_path)
    monkeypatch.setattr(get_data, "fetch", fake_fetch)
    monkeypatch.setattr(get_data, "person_classes", lambda path: {"/m/p": "Person"})
    monkeypatch.setattr(get_data, "read_boxes", lambda path, classes: {"x": alone + [crowd]})
    args = argparse.Namespace(max_train=1, max_val=1, seed=0, crowded_share=0.0, grey_groups=grey)
    get_data.cmd_openimages(args)
    images = tmp_path / "openimages" / "images" / "train"
    assert [path.name for path in images.glob("*.jpg")] == ["x.jpg"]
    assert not list(images.glob("*.whole.*")) and not list(images.glob("*.part"))
    with Image.open(images / "x.jpg") as out:
        red = out.getpixel((70, 70))[0]
    assert abs(red - (get_data.GREY if grey else 200)) <= 6


def test_grey_share_counts_the_crowd_less_the_people_in_it():
    crowd = {"label": "Person", "bbox": [0.0, 0.0, 0.5, 1.0], "group": True, "depiction": False}
    person = {"label": "Man", "bbox": [0.0, 0.0, 0.1, 0.5], "group": False, "depiction": False}
    assert get_data.grey_share([crowd]) == pytest.approx(0.5)
    assert get_data.grey_share([crowd, person]) == pytest.approx(0.45)
    assert get_data.grey_share([person]) == 0.0


def test_an_image_that_would_be_mostly_grey_is_left_out(tmp_path, monkeypatch):
    crowd = {"label": "Person", "bbox": [0.0, 0.0, 0.9, 0.9], "group": True, "depiction": False}
    seen = {}
    monkeypatch.setattr(get_data, "DATA", tmp_path)
    monkeypatch.setattr(get_data, "fetch", lambda url, dst, tries=get_data.TRIES: True)
    monkeypatch.setattr(get_data, "person_classes", lambda path: {"/m/p": "Person"})
    boxes = {"grey": _people(1, 0.2) + [crowd], "fine": _people(1, 0.2)}
    monkeypatch.setattr(get_data, "read_boxes", lambda path, classes: boxes)

    def choose(found, limit, seed, crowded):
        seen.update(found)
        return {}

    monkeypatch.setattr(get_data, "choose_images", choose)
    args = argparse.Namespace(max_train=1, max_val=1, seed=0, crowded_share=0.0, grey_groups=True)
    get_data.cmd_openimages(args)
    assert sorted(seen) == ["fine"]


@pytest.mark.parametrize("held", ["groups kept", None])
def test_a_folder_of_another_kind_is_refused(tmp_path, held):
    (tmp_path / "old.jpg").write_bytes(b"jpg")
    if held:
        (tmp_path / get_data.KIND).write_text(held + "\n")
    with pytest.raises(SystemExit, match="another kind"):
        get_data.same_kind(tmp_path, grey=True)


def test_a_folder_of_the_same_kind_or_empty_is_marked(tmp_path):
    get_data.same_kind(tmp_path, grey=False)
    get_data.same_kind(tmp_path, grey=False)
    assert (tmp_path / get_data.KIND).read_text() == "groups kept\n"


def test_a_crowd_on_an_unreadable_image_fails_and_is_fetched_again(tmp_path, monkeypatch):
    def fake_fetch(url, dst, tries=get_data.TRIES):
        if not url.endswith(".csv"):
            dst.write_bytes(b"not a jpeg")
        return True

    crowd = {"label": "Person", "bbox": [0.5, 0.5, 0.4, 0.4], "group": True, "depiction": False}
    monkeypatch.setattr(get_data, "DATA", tmp_path)
    monkeypatch.setattr(get_data, "fetch", fake_fetch)
    monkeypatch.setattr(get_data, "person_classes", lambda path: {"/m/p": "Person"})
    monkeypatch.setattr(
        get_data, "read_boxes", lambda path, classes: {"x": _people(1, 0.2) + [crowd]}
    )
    args = argparse.Namespace(max_train=1, max_val=1, seed=0, crowded_share=0.0, grey_groups=True)
    with pytest.raises(SystemExit, match="incomplete"):
        get_data.cmd_openimages(args)
    assert not list((tmp_path / "openimages" / "images" / "train").glob("*.jpg"))


def test_a_turned_image_is_greyed_as_it_is_shown(tmp_path):
    from PIL import Image

    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    stored = Image.new("RGB", (100, 50), (200, 30, 30))  # shown turned a quarter: 50 x 100
    exif = stored.getexif()
    exif[0x0112] = 6
    stored.save(src, quality=100, exif=exif)
    get_data.grey_out(src, dst, [(0.0, 0.0, 1.0, 0.5)], [])
    with Image.open(dst) as out:
        assert out.size == (50, 100) and out.getexif().get(0x0112, 1) == 1
        top, bottom = out.getpixel((25, 20)), out.getpixel((25, 80))
    assert abs(top[0] - get_data.GREY) <= 3 and abs(bottom[0] - 200) <= 6
