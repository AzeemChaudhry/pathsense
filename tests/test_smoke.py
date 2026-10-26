import pathsense
from pathsense.detect import Detector


def test_package_imports():
    assert pathsense.__version__


def test_detector_protocol_exists():
    assert "__call__" in Detector.__dict__
