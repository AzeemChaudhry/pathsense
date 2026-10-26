"""Download people detection data and convert it to YOLO format, ready for training.

Sources
-------
crowdhuman  CrowdHuman (Megvii), mirrored at hf.co/datasets/Funtion/CrowdHuman.
            15,000 train / 4,370 val images, about 23 people per image.
            License: CC BY NC 4.0, non commercial research and education only.
roboflow    Any Roboflow Universe project exported in YOLO format, for example an
            overhead people dataset. Needs a free API key in ROBOFLOW_API_KEY.

Usage
-----
    uv sync --extra data
    uv run python train/get_data.py crowdhuman --max-train 5000
    uv run python train/get_data.py roboflow \
        --workspace krupal --project overhead-person-94pcq --version 1

Output (git ignored)
--------------------
    data/crowdhuman/images/{train,val}/*.jpg
    data/crowdhuman/labels/{train,val}/*.txt     one "0 cx cy w h" line per person, normalised
    data/crowdhuman.yaml                          pass to `yolo train data=...`
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HF_REPO = "Funtion/CrowdHuman"

# Train zips hold about 5,000 images each; train01 alone is plenty for a GTX 1650.
SPLIT_FILES = {
    "train": (["CrowdHuman_train01.zip"], "annotation_train.odgt"),
    "val": (["CrowdHuman_val.zip"], "annotation_val.odgt"),
}
ALL_TRAIN_ZIPS = ["CrowdHuman_train01.zip", "CrowdHuman_train02.zip", "CrowdHuman_train03.zip"]


# ----------------------------------------------------------------------------- conversion
def odgt_to_yolo(record: dict, width: int, height: int, box_key: str = "fbox",
                 min_size: int = 8) -> list[str]:
    """Convert one CrowdHuman odgt record to YOLO label lines (class 0 = person).

    Drops "mask" regions (crowds, reflections, posters) and boxes marked ignore,
    clips full body boxes that run outside the image, and skips boxes under min_size pixels.
    """
    lines = []
    for box in record.get("gtboxes", []):
        if box.get("tag") != "person" or box.get("extra", {}).get("ignore", 0) == 1:
            continue
        x, y, w, h = box[box_key]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        bw, bh = x2 - x1, y2 - y1
        if bw < min_size or bh < min_size:
            continue
        cx, cy = (x1 + bw / 2) / width, (y1 + bh / 2) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
    return lines


def read_odgt(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------- crowdhuman
def download_crowdhuman(raw: Path, all_train: bool) -> None:
    from huggingface_hub import hf_hub_download

    files = {"annotation_train.odgt", "annotation_val.odgt", "CrowdHuman_val.zip"}
    files |= set(ALL_TRAIN_ZIPS if all_train else ["CrowdHuman_train01.zip"])
    for name in sorted(files):
        print(f"[download] {name}")
        local = hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=raw)
        if name.endswith(".zip"):
            marker = raw / f".{name}.extracted"
            if not marker.exists():
                print(f"[extract]  {name}")
                with zipfile.ZipFile(local) as z:
                    z.extractall(raw)
                marker.touch()


def build_crowdhuman(raw: Path, out: Path, max_train: int | None, max_val: int | None,
                     box_key: str, seed: int) -> None:
    from PIL import Image

    images_dir = raw / "Images"
    for split, (_, ann_name) in SPLIT_FILES.items():
        records = [r for r in read_odgt(raw / ann_name) if (images_dir / f"{r['ID']}.jpg").exists()]
        limit = max_train if split == "train" else max_val
        if limit and len(records) > limit:
            random.Random(seed).shuffle(records)
            records = records[:limit]

        img_out, lbl_out = out / "images" / split, out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        people = empty = 0
        for r in records:
            src = images_dir / f"{r['ID']}.jpg"
            with Image.open(src) as im:
                width, height = im.size
            lines = odgt_to_yolo(r, width, height, box_key=box_key)
            if not lines:
                empty += 1
            people += len(lines)
            dst = img_out / src.name
            if not dst.exists():
                try:
                    os.link(src, dst)  # hard link: no extra disk space
                except OSError:
                    shutil.copy2(src, dst)
            (lbl_out / f"{r['ID']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"[{split}] {len(records)} images, {people} people, {empty} with no usable boxes")

    yaml_path = DATA / "crowdhuman.yaml"
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n"
    )
    print(f"[done] dataset config written to {yaml_path}")


def cmd_crowdhuman(args: argparse.Namespace) -> None:
    raw = DATA / "raw" / "crowdhuman"
    raw.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_crowdhuman(raw, args.all_train)
    build_crowdhuman(raw, DATA / "crowdhuman", args.max_train, args.max_val, args.box, args.seed)


# ----------------------------------------------------------------------------- roboflow
def cmd_roboflow(args: argparse.Namespace) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free at app.roboflow.com > Settings > API).")
    from roboflow import Roboflow

    target = DATA / args.project
    project = Roboflow(api_key=key).workspace(args.workspace).project(args.project)
    project.version(args.version).download("yolov8", location=str(target))
    print(f"[done] {target} (use its data.yaml; check the license on the project page)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="source", required=True)

    ch = sub.add_parser("crowdhuman", help="CrowdHuman from the Hugging Face Hub")
    ch.add_argument(
        "--all-train", action="store_true", help="all 3 train zips (~8.4 GB) instead of 1"
    )
    ch.add_argument("--max-train", type=int, default=None, help="random subset size for train")
    ch.add_argument("--max-val", type=int, default=1000, help="random subset size for val")
    ch.add_argument("--box", choices=["fbox", "vbox"], default="fbox",
                    help="full body (default, matches tracking) or visible region box")
    ch.add_argument("--seed", type=int, default=0)
    ch.add_argument("--skip-download", action="store_true", help="reuse files in data/raw")
    ch.set_defaults(func=cmd_crowdhuman)

    rf = sub.add_parser("roboflow", help="a Roboflow Universe project in YOLO format")
    rf.add_argument("--workspace", required=True)
    rf.add_argument("--project", required=True)
    rf.add_argument("--version", type=int, required=True)
    rf.set_defaults(func=cmd_roboflow)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
