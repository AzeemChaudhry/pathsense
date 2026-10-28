import numpy as np
import pytest
from conftest import detections

from pathsense.config import ConfigError, load_config
from pathsense.detect import DETECTORS, build_detector
from pathsense.detect.slice import SlicedDetector


class SmallImagesOnly:
    """Fake detector that, like a real one shrinking a big frame, only finds a person in
    images no larger than 400 px: the box around the bright pixels it is given."""

    name = "fake"

    def __init__(self):
        self.sizes = []

    def __call__(self, image):
        self.sizes.append(image.shape[:2])
        bright = np.argwhere(image[..., 0] > 0)
        if max(image.shape[:2]) > 400 or len(bright) == 0:
            return detections([])
        (y1, x1), (y2, x2) = bright.min(axis=0), bright.max(axis=0) + 1
        return detections([[x1, y1, x2, y2]])


def frame_with_person_at(x, y, size=(1200, 1600)):
    image = np.zeros((*size, 3), np.uint8)
    image[y + 60 : y + 160, x + 50 : x + 90] = 255
    return image


def test_tiles_find_a_person_the_whole_frame_misses():
    fake = SmallImagesOnly()
    frame = frame_with_person_at(400, 320)
    assert len(fake(frame)) == 0  # whole frame: too large
    found = SlicedDetector(fake, tile=400)(frame)
    assert len(found) == 1
    assert found.xyxy[0].tolist() == [450, 380, 490, 480]  # back in frame coordinates
    assert all(max(size) <= 400 for size in fake.sizes[2:])  # [1]: the whole frame pass


def test_a_person_on_a_seam_is_found_once_whole():
    # Tiles of 400 px start every 320 px; this person spans the seam at y = 400.
    found = SlicedDetector(SmallImagesOnly(), tile=400)(frame_with_person_at(400, 300))
    assert found.xyxy.tolist() == [[450, 360, 490, 460]]


def test_tiles_cover_the_frame_and_end_on_its_edges():
    from pathsense.detect.slice import tiles

    rects = tiles(1000, 500, 400)
    assert rects[0] == (0, 0, 400, 400) and rects[-1] == (600, 100, 1000, 500)
    covered = np.zeros((500, 1000), bool)
    for x1, y1, x2, y2 in rects:
        covered[y1:y2, x1:x2] = True
    assert covered.all()


def test_a_small_frame_is_not_cut():
    fake = SmallImagesOnly()
    SlicedDetector(fake, tile=640)(np.zeros((300, 400, 3), np.uint8))
    assert fake.sizes == [(300, 400)]


def test_config_wraps_any_detector(monkeypatch):
    monkeypatch.setitem(DETECTORS, "fake", lambda cfg: SmallImagesOnly())
    config = load_config(overrides=["detector.name=fake", "detector.slice=960"])
    detector = build_detector(config.detector)
    assert detector.name == "fake+slice960" and detector.tile == 960
    plain = build_detector(load_config(overrides=["detector.name=fake"]).detector)
    assert plain.name == "fake"


@pytest.mark.parametrize("value", ["100", "true", "abc", "1.5"])
def test_bad_slice_sizes_are_config_errors(value):
    with pytest.raises(ConfigError, match="detector.slice"):
        load_config(overrides=[f"detector.slice={value}"])
