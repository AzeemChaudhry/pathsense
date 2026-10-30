"""Turn each person's box into a unit vector that describes how they look.

    embedder = Embedder("data/models/osnet_x0_25_msmt17.pt")
    vectors = embedder(frame.image, detections.xyxy)   # (people, 512), length 1

The vectors describe clothing and build, not faces. They stay in memory for one video and
are never tied to a name.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pathsense.config import ConfigError

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, as OSNet was trained
STD = np.array([0.229, 0.224, 0.225], np.float32)
HEIGHT, WIDTH, FEATURES = 256, 128, 512


def crops(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Boxes of a BGR image -> the network's input (n, 3, 256, 128), RGB and normalised.
    A box wholly outside the image gives an all zero crop."""
    height, width = image.shape[:2]
    out = np.zeros((len(boxes), HEIGHT, WIDTH, 3), np.float32)
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(np.ceil(x2)), width), min(int(np.ceil(y2)), height)
        if x2 > x1 and y2 > y1:
            patch = cv2.resize(image[y1:y2, x1:x2], (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            out[k] = patch[:, :, ::-1]
    return ((out / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)


class Embedder:
    def __init__(
        self, weights: str | Path, device: str | None = None, batch: int = 256, half: bool = False
    ) -> None:
        if not Path(weights).is_file():
            raise ConfigError(
                f"appearance.weights not found: {weights}. OSNet x0.25 (MSMT17) comes from the "
                "torchreid model zoo; see docs/TRACKER_STUDY.md, Appearance"
            )
        import torch  # heavy, and only needed with appearance on

        from pathsense.appearance.osnet import load

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        self.model = load(Path(weights)).to(self.device)

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, FEATURES), np.float32)
        x = crops(image, boxes)
        parts = []
        with self.torch.no_grad():
            for at in range(0, len(x), self.batch):
                batch = self.torch.from_numpy(x[at : at + self.batch]).to(self.device)
                parts.append(self.model(batch).cpu().numpy())
        vectors = np.concatenate(parts)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        if not np.isfinite(vectors).all():  # one bad vector would poison a track's memory
            raise RuntimeError("the appearance model gave a non finite vector")
        return vectors
