"""Download people detection data and convert it to YOLO format, ready for training.

Sources
-------
openimages  Open Images V7 (Google), person boxes (Person, Man, Woman, Boy, Girl -> person).
            Annotations CC BY 4.0, images listed as CC BY 2.0: allows commercial use with
            attribution. Downloaded with FiftyOne. Use this for any model you deploy or sell.
crowdhuman  CrowdHuman (Megvii), mirrored at hf.co/datasets/Funtion/CrowdHuman.
            15,000 train / 4,370 val images, about 23 people per image.
            License: CC BY NC 4.0, non commercial research and education only.
roboflow    Any Roboflow Universe project exported in YOLO format, for example an
            overhead people dataset. Needs a free API key in ROBOFLOW_API_KEY.

Usage
-----
    uv sync --extra data
    uv run python train/get_data.py openimages --max-train 8000
    uv run python train/get_data.py crowdhuman --max-train 5000
    uv run python train/get_data.py roboflow \
        --workspace krupal --project overhead-person-94pcq --version 1

Output (git ignored)
--------------------
    data/openimages/..., data/openimages.yaml     same layout as below
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
def odgt_to_yolo(
    record: dict, width: int, height: int, box_key: str = "fbox", min_size: int = 8
) -> list[str]:
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


# Open Images splits a "person" across several leaf classes; all become class 0.
OPENIMAGES_PERSON_CLASSES = ["Person", "Man", "Woman", "Boy", "Girl"]


def openimages_to_yolo(dets: list[dict], min_frac: float = 0.005) -> list[str]:
    """Convert Open Images person boxes to YOLO lines.

    Each det is {"label", "bbox": [x, y, w, h] normalised top left, "group", "depiction"}.
    Drops group boxes (one box around a crowd) and depictions (posters, statues, drawings),
    which would teach the model the wrong thing, plus slivers under min_frac of the image.
    """
    lines = []
    for d in dets:
        if d["label"] not in OPENIMAGES_PERSON_CLASSES or d.get("group") or d.get("depiction"):
            continue
        x, y, w, h = d["bbox"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(1.0, x + w), min(1.0, y + h)
        w, h = x2 - x1, y2 - y1
        if w < min_frac or h < min_frac:
            continue
        lines.append(f"0 {x1 + w / 2:.6f} {y1 + h / 2:.6f} {w:.6f} {h:.6f}")
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


def build_crowdhuman(
    raw: Path, out: Path, max_train: int | None, max_val: int | None, box_key: str, seed: int
) -> None:
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


# ----------------------------------------------------------------------------- open images
def _flag(det, name: str) -> bool:
    try:
        return bool(det[name])
    except (KeyError, AttributeError):
        return False


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(name: str, out: Path) -> Path:
    yaml_path = DATA / f"{name}.yaml"
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n"
    )
    return yaml_path


def cmd_openimages(args: argparse.Namespace) -> None:
    import fiftyone as fo
    import fiftyone.zoo as foz

    # Raw downloads go to FiftyOne's zoo folder (default ~/fiftyone). To keep them on
    # another disk, export FIFTYONE_DATASET_ZOO_DIR=/path/to/disk/fiftyone first.
    out = DATA / "openimages"
    for split, oi_split, limit in [
        ("train", "train", args.max_train),
        ("val", "validation", args.max_val),
    ]:
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=oi_split,
            label_types=["detections"],
            classes=OPENIMAGES_PERSON_CLASSES,
            only_matching=True,
            max_samples=limit,
            shuffle=True,
            seed=args.seed,
            dataset_name=f"pathsense-openimages-{oi_split}",
            drop_existing_dataset=True,
        )
        field = next(
            name
            for name, f in ds.get_field_schema().items()
            if getattr(f, "document_type", None) is fo.Detections
        )
        img_out, lbl_out = out / "images" / split, out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        kept = people = 0
        for sample in ds.iter_samples(progress=True):
            labels = sample[field]
            dets = [
                {
                    "label": d.label,
                    "bbox": d.bounding_box,
                    "group": _flag(d, "IsGroupOf"),
                    "depiction": _flag(d, "IsDepiction"),
                }
                for d in (labels.detections if labels else [])
            ]
            lines = openimages_to_yolo(dets)
            if not lines:
                continue  # image held only crowds or drawings of people
            src = Path(sample.filepath)
            link_or_copy(src, img_out / src.name)
            (lbl_out / f"{src.stem}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
            people += len(lines)
        print(f"[{split}] {kept} images, {people} people")

    print(f"[done] dataset config written to {write_yaml('openimages', out)}")
    print(
        "[note] attribution: Open Images V7 by Google, annotations CC BY 4.0, "
        "images CC BY 2.0 (verify per image before redistributing images)"
    )


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

    oi = sub.add_parser("openimages", help="Open Images V7 people (commercial use OK)")
    oi.add_argument("--max-train", type=int, default=8000, help="train images to download")
    oi.add_argument("--max-val", type=int, default=1000, help="val images to download")
    oi.add_argument("--seed", type=int, default=0)
    oi.set_defaults(func=cmd_openimages)

    ch = sub.add_parser("crowdhuman", help="CrowdHuman from the Hugging Face Hub")
    ch.add_argument(
        "--all-train", action="store_true", help="all 3 train zips (~8.4 GB) instead of 1"
    )
    ch.add_argument("--max-train", type=int, default=None, help="random subset size for train")
    ch.add_argument("--max-val", type=int, default=1000, help="random subset size for val")
    ch.add_argument(
        "--box",
        choices=["fbox", "vbox"],
        default="fbox",
        help="full body (default, matches tracking) or visible region box",
    )
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
