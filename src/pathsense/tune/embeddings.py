"""Save one appearance vector per saved box, so a replay can use looks without the video.

    vectors = cached_embeddings(video, saved, config.appearance, saved_path)
    replay(pipeline, saved, vectors)

Rows match `saved.xyxy`. The file is a float16 .npy, read through a memory map so parallel
workers share one copy. Its name carries what it was made from: the detector key (settings
and video) and the weights file.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from pathsense.appearance import Embedder
from pathsense.appearance.embed import FEATURES
from pathsense.config import AppearanceConfig, ConfigError
from pathsense.io import VideoSource
from pathsense.tune.cache import SavedDetections, _stamp


def embeddings_path(saved_path: Path, saved: SavedDetections, config: AppearanceConfig) -> Path:
    if not Path(config.weights).is_file():
        raise ConfigError(f"appearance.weights not found: {config.weights}")
    key = f"{saved.detector}|{_stamp(config.weights)}".encode()
    return Path(saved_path).with_suffix(f".osnet-{hashlib.sha1(key).hexdigest()[:8]}.npy")


def embed_video(video: Path | str, saved: SavedDetections, embedder: Embedder) -> np.ndarray:
    """Every saved box of `video`, embedded; rows match `saved.xyxy`."""
    vectors = np.zeros((len(saved.xyxy), FEATURES), np.float16)
    starts = saved.starts
    with VideoSource(str(video)) as source:
        for k, frame in enumerate(source):
            if k >= len(saved) or frame.index != saved.index[k]:
                raise ConfigError(f"{video} does not match its saved detections")
            rows = slice(starts[k], starts[k + 1])
            vectors[rows] = embedder(frame.image, saved.xyxy[rows])
    return vectors


def cached_embeddings(
    video: Path | str,
    saved: SavedDetections,
    config: AppearanceConfig,
    saved_path: Path,
    build=Embedder,
) -> np.ndarray:
    """The saved vectors, made if missing."""
    path = embeddings_path(saved_path, saved, config)
    if not path.exists():
        vectors = embed_video(video, saved, build(config.weights, config.device, config.batch))
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npy")
        try:
            with os.fdopen(handle, "wb") as out:
                np.save(out, vectors)
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    return load_embeddings(path)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise ConfigError(f"no saved embeddings at {path}; `pathsense tune` makes them first")
    return np.load(path, mmap_mode="r")


__all__ = ["cached_embeddings", "embed_video", "embeddings_path", "load_embeddings"]
