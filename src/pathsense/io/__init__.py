"""Frame sources and sinks: webcam, video files, streams, and the annotated video writer."""

from pathsense.io.source import Frame, VideoSource, parse_source
from pathsense.io.writer import RealTimeVideoWriter

__all__ = ["Frame", "RealTimeVideoWriter", "VideoSource", "parse_source"]
