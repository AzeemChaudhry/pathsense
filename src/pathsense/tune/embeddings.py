"""Save one appearance vector per saved box, so a replay can use looks without the video.

    vectors = cached_embeddings(video, saved, config.appearance, saved_path)
    replay(pipeline, saved, vectors)

Rows match `saved.xyxy`. The file is a float16 .npy, read through a memory map so parallel
workers share one copy. Its name carries what it was made from: the detector key (settings
and video), the boxes themselves, the weights file and FORMAT, so different boxes or a
changed crop make a new file. Unlike a live video, these vectors are on disk: research
clips only, in data/cache. A replay renormalises them, since float16 rounds their length.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from pathsense.config import AppearanceConfig, ConfigError
from pathsense.io import VideoSource
from pathsense.tune.cache import SavedDetections, _stamp

FEATURES = 512  # as pathsense.appearance.embed; not imported, so a plain tune loads none of it
FORMAT = 1  # raise when the crop, the normalising or the network changes


def embeddings_path(saved_path: Path, saved: SavedDetections, config: AppearanceConfig) -> Path:
    if not Path(config.weights).is_file():
        raise ConfigError(f"appearance.weights not found: {config.weights}")
    key = hashlib.sha1(f"{FORMAT}|{saved.detector}|{_stamp(config.weights)}|".encode())
    key.update(saved.xyxy.tobytes())
    return Path(saved_path).with_suffix(f".osnet-{key.hexdigest()[:8]}.npy")


def embed_video(video: Path | str, saved: SavedDetections, embedder) -> np.ndarray:
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
    build=None,
) -> np.ndarray:
    """The saved vectors, made if missing."""
    path = embeddings_path(saved_path, saved, config)
    if not path.exists():
        if build is None:
            from pathsense.appearance.embed import Embedder as build
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
    return load_embeddings(path, saved)


def load_embeddings(path: Path, saved: SavedDetections) -> np.ndarray:
    if not path.exists():
        raise ConfigError(
            f"no saved embeddings at {path}; `pathsense tune` or `scripts/appearance_study "
            "embed` makes them"
        )
    vectors = np.load(path, mmap_mode="r")
    if len(vectors) != len(saved.xyxy):
        raise ConfigError(f"{path} has {len(vectors)} rows for {len(saved.xyxy)} boxes")
    return vectors


__all__ = ["cached_embeddings", "embed_video", "embeddings_path", "load_embeddings"]
