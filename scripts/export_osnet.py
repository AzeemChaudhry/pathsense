"""Export OSNet x0.25, a small person re-identification network, to ONNX.

    uv run python scripts/export_osnet.py data/models/osnet_x0_25_msmt17.pt

Writes the .onnx next to the weights: input `images` (n, 3, 256, 128), RGB scaled to 0..1
then normalised with the ImageNet mean and std; output `features` (n, 512). Phase 3 step 5
tested these embeddings (docs/TRACKER_STUDY.md); the same file is what the browser would
run (docs/RESEARCH.md).

The network itself is `pathsense.appearance.osnet`; onnx and onnxruntime are needed here
only (`uv run --with onnx --with onnxruntime python scripts/export_osnet.py ...`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pathsense.appearance.osnet import SIZE, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("weights", type=Path, help="torchreid checkpoint (.pt or .pth)")
    args = parser.parse_args()
    out = args.weights.with_suffix(".onnx")
    model = load(args.weights)
    sample = torch.rand(4, 3, *SIZE)
    with torch.no_grad():
        expected = model(sample).numpy()
    torch.onnx.export(
        model,
        (sample,),
        out,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={"images": {0: "n"}, "features": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime

    session = onnxruntime.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = session.run(None, {"images": sample.numpy()})[0]
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, max difference from torch "
          f"{float(np.abs(got - expected).max()):.2e}")  # fmt: skip


if __name__ == "__main__":
    main()
