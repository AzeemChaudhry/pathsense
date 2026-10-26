import pytest

from pathsense.timing import FpsMeter


def test_fps_meter_starts_at_first_reading_then_smooths():
    meter = FpsMeter(smoothing=0.5)
    assert meter.update(0.1) == pytest.approx(10.0)
    assert meter.update(0.05) == pytest.approx(15.0)  # halfway between 10 and 20
    assert meter.ms == pytest.approx(1000 / 15)


def test_fps_meter_ignores_zero_duration_and_rejects_bad_smoothing():
    meter = FpsMeter()
    assert meter.update(0.0) == 0.0 and meter.ms == 0.0
    with pytest.raises(ValueError):
        FpsMeter(smoothing=1.0)
