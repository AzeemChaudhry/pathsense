"""Detector adapters with fake models: no weights or GPU needed."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import detections

from pathsense.config import ConfigError, DetectorConfig
from pathsense.detect import DETECTORS, PERSON_CLASS_ID, build_detector, select_people
from pathsense.detect.rfdetr import RfDetrDetector
from pathsense.detect.yolo import YoloDetector


def test_select_people_keeps_the_person_class_and_relabels_it():
    mixed = detections([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 2, 2]])
    mixed.class_id[:] = [1, 3, 1]  # RF-DETR numbering: 1 person, 3 car
    people = select_people(mixed, model_person_id=1)
    assert len(people) == 2
    assert (people.class_id == PERSON_CLASS_ID).all()
    assert mixed.class_id.tolist() == [1, 3, 1]  # input left untouched


class FakeRfDetr:
    def __init__(self, result):
        self.result, self.calls = result, []

    def predict(self, image, threshold, include_source_image):
        self.calls.append((image, threshold, include_source_image))
        return self.result


def test_rfdetr_adapter_sends_rgb_and_returns_people_only():
    raw = detections([[0, 0, 10, 10], [5, 5, 20, 20]])
    raw.class_id[:] = [1, 2]  # person, bicycle
    model = FakeRfDetr(raw)
    detector = RfDetrDetector(model, name="rfdetr-nano", conf=0.2)
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    out = detector(bgr)
    image, threshold, include = model.calls[0]
    assert image[0, 0].tolist() == [0, 0, 255]  # blue is last in RGB
    assert threshold == 0.2 and include is False
    assert len(out) == 1 and out.class_id.tolist() == [PERSON_CLASS_ID]


class FakeYolo:
    """Stands in for ultralytics.YOLO; returns an object sv.Detections.from_ultralytics reads."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, frame, **kwargs):
        self.kwargs = kwargs
        boxes = SimpleNamespace(
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
            conf=np.array([0.8]),
            cls=np.array([0.0]),
            id=None,
            data=np.zeros((1, 6)),
        )
        return [SimpleNamespace(boxes=boxes, obb=None, masks=None, names={0: "person"})]


def test_yolo_adapter_asks_for_people_only_and_returns_detections(monkeypatch):
    import supervision as sv

    def from_ultralytics(result):
        b = result.boxes
        return sv.Detections(xyxy=b.xyxy, confidence=b.conf, class_id=b.cls.astype(int))

    monkeypatch.setattr(sv.Detections, "from_ultralytics", staticmethod(from_ultralytics))
    model = FakeYolo()
    detector = YoloDetector(model, name="yolo26n", imgsz=320, conf=0.3, device="cpu")
    out = detector(np.zeros((48, 64, 3), np.uint8))
    assert model.kwargs["classes"] == [0] and model.kwargs["imgsz"] == 320
    assert model.kwargs["conf"] == 0.3 and model.kwargs["device"] == "cpu"
    assert len(out) == 1 and out.class_id.tolist() == [PERSON_CLASS_ID]


def test_registry_has_both_families():
    assert {"yolo26n", "rfdetr-nano"} <= set(DETECTORS)


def test_build_detector_dispatches_by_name(monkeypatch):
    seen = []
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: seen.append(cfg) or "built")
    cfg = DetectorConfig(name="fake", conf=0.4)
    assert build_detector(cfg) == "built" and seen == [cfg]


def test_unknown_detector_lists_the_choices():
    with pytest.raises(ConfigError, match="yolo26n"):
        build_detector(DetectorConfig(name="yolo99"))


@pytest.mark.slow
@pytest.mark.parametrize("name", ["yolo26n", "rfdetr-nano"])
def test_real_detectors_find_people_in_the_sample_clip(name):
    import cv2

    cap = cv2.VideoCapture("data/samples/people-walking.mp4")
    ok, frame = cap.read()
    if not ok:
        pytest.skip("sample clip missing; see docs/DEVELOPMENT.md")
    out = build_detector(DetectorConfig(name=name, conf=0.3))(frame)
    assert len(out) >= 10  # the first frame shows about 20 people
    assert (out.class_id == PERSON_CLASS_ID).all()


def test_a_partial_view_of_the_same_person_is_dropped():
    from pathsense.detect.base import drop_partial_duplicates

    # The grocery shopper at 6 s: upper body and whole body, same width and top.
    shopper = detections([[2104, 900, 2305, 1149], [2102, 901, 2306, 1402]])
    assert drop_partial_duplicates(shopper).xyxy.tolist() == [[2102, 901, 2306, 1402]]
    # A person behind another, seen from above: inside their box, top close but not equal.
    queue = detections([[845, 1135, 1066, 1553], [842, 1100, 1070, 1700]])
    assert len(drop_partial_duplicates(queue)) == 2
    # Two people in a crowd: one partly in front of the other, different tops and widths.
    crowd = detections([[100, 100, 200, 400], [150, 180, 260, 420], [400, 100, 500, 400]])
    assert len(drop_partial_duplicates(crowd)) == 3
    # A child in front of an adult: inside the adult's box, but the tops differ.
    child = detections([[100, 100, 200, 400], [110, 250, 190, 400]])
    assert len(drop_partial_duplicates(child)) == 2


def test_dropping_partial_boxes_does_not_depend_on_their_order():
    from pathsense.detect.base import drop_partial_duplicates

    # Review of PR 10: B is a partial view of A, and C one of B but not of A. Once B is
    # dropped it cannot drop C, whichever order the detector listed them in.
    a, b, c = [100, 100, 200, 500], [102, 105, 198, 400], [106, 113, 194, 300]
    for boxes in ([a, b, c], [c, b, a], [b, c, a]):
        kept = drop_partial_duplicates(detections(boxes)).xyxy.tolist()
        assert sorted(kept) == sorted([a, c])
