import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "webcam_baseline", Path(__file__).resolve().parents[1] / "scripts" / "webcam_baseline.py"
)
webcam_baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webcam_baseline)


def test_parser_defaults():
    args = webcam_baseline.build_parser().parse_args([])
    assert args.source == "0" and args.imgsz == 640 and args.conf == 0.1 and args.save is None
