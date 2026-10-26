import pathsense
from pathsense.detect import Detector


def test_package_imports():
    assert pathsense.__version__


def test_detector_protocol_exists():
    assert hasattr(Detector, "__call__")
