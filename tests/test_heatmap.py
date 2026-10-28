import numpy as np
import pytest
from conftest import detections

from pathsense.analytics import Heatmap


def standing(x, y, track_id=1):
    tracked = detections([[x - 10, y - 50, x + 10, y]])
    tracked.tracker_id = np.array([track_id])
    return tracked


def test_time_adds_up_in_the_cell_under_the_feet():
    heat = Heatmap(100, 80, cell=10)
    for frame in range(11):  # 1 s at 10 FPS
        heat.add(standing(35, 45), t=frame / 10)
    assert heat.seconds[4, 3] == pytest.approx(1.0)
    assert heat.seconds.sum() == pytest.approx(1.0)


def test_a_stall_adds_at_most_a_second_and_off_picture_feet_are_ignored():
    heat = Heatmap(100, 80, cell=10)
    heat.add(standing(35, 45), t=0.0)
    heat.add(standing(35, 45), t=30.0)  # the camera froze for 30 s
    heat.add(standing(35, 200), t=30.1)  # feet below the picture
    assert heat.seconds.sum() == pytest.approx(1.0)


def test_render_colours_only_where_people_stood():
    heat = Heatmap(100, 80, cell=10)
    for frame in range(20):
        heat.add(standing(35, 45), t=frame / 10)
    background = np.full((80, 100, 3), 128, np.uint8)
    out = heat.render(background)
    assert out.shape == background.shape
    assert (out[45, 35] != 128).any() and (out[5, 95] == 128).all()
    assert (Heatmap(100, 80).render(background) == background).all()  # nobody: unchanged
