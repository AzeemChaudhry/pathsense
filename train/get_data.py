"""Download people detection data and convert it to YOLO format, ready for training.

Sources
-------
openimages  Open Images V7 (Google), person boxes (Person, Man, Woman, Boy, Girl -> person).
            Annotations CC BY 4.0, images listed as CC BY 2.0: allows commercial use with
            attribution. Fetched from Google's published files. Use this for any model you
            deploy or sell.
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
import csv
import http.client
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
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

OPENIMAGES = "https://storage.googleapis.com/openimages"
OPENIMAGES_CLASSES = f"{OPENIMAGES}/v7/oidv7-class-descriptions-boxable.csv"
OPENIMAGES_BOXES = {  # our split -> Open Images' name for it, its box file
    "train": ("train", f"{OPENIMAGES}/v6/oidv6-train-annotations-bbox.csv"),
    "val": ("validation", f"{OPENIMAGES}/v5/validation-annotations-bbox.csv"),
}
OPENIMAGES_IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image}.jpg"
WORKERS = 16  # few enough that the bucket does not slow us down
TRIES = 5
BOX_COLUMNS = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"}


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
def fetch(url: str, dst: Path, tries: int = TRIES) -> bool:
    """Download `url` to `dst`, whole or not at all. False if it does not exist or keeps
    failing. A finished file is kept, so a stopped run skips it; a partial one starts over."""
    if dst.exists():
        return True
    part = dst.with_suffix(f"{dst.suffix}.part")  # one per file: the last run's is overwritten
    for attempt in range(tries):
        if attempt:
            time.sleep(2**attempt)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, 1024 * 1024)
                size = response.headers.get("Content-Length")
            if size is None or part.stat().st_size != int(size):  # a kept file is trusted
                raise OSError(f"short read from {url}")
            part.replace(dst)
            return True
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code not in (408, 429):
                break  # it is not there, and asking again will not help
        except (OSError, http.client.HTTPException):
            pass  # a dropped connection, a timeout, a body cut short
    part.unlink(missing_ok=True)
    return False


def person_classes(path: Path) -> dict[str, str]:
    """Open Images' class id -> name, for the classes that mean a person."""
    with path.open(newline="") as f:
        rows = csv.reader(f)
        return {row[0]: row[1] for row in rows if row[1] in OPENIMAGES_PERSON_CLASSES}


def read_boxes(path: Path, classes: dict[str, str]) -> dict[str, list[dict]]:
    """Image id -> its person boxes, in the form `openimages_to_yolo` takes."""
    boxes: dict[str, list[dict]] = {}
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        missing = BOX_COLUMNS - set(rows.fieldnames or ())
        if missing:
            raise SystemExit(f"{path}: no column {sorted(missing)}; has the file changed?")
        for row in rows:
            if row["LabelName"] in classes:
                if None in row.values():
                    raise SystemExit(f"{path}:{rows.line_num}: short row")
                try:
                    x, y = float(row["XMin"]), float(row["YMin"])
                    w, h = float(row["XMax"]) - x, float(row["YMax"]) - y
                except ValueError:
                    raise SystemExit(f"{path}:{rows.line_num}: bad box") from None
                boxes.setdefault(row["ImageID"], []).append(
                    {
                        "label": classes[row["LabelName"]],
                        "bbox": [x, y, w, h],
                        "group": row["IsGroupOf"] != "0",  # -1, not known, could be either
                        "depiction": row["IsDepiction"] != "0",
                    }
                )
    return boxes


def choose_images(boxes: dict[str, list[dict]], limit: int, seed: int) -> dict[str, list[str]]:
    """`limit` images, the same ones for a seed, each with at least one usable box: image id
    -> its YOLO lines. An image holding only crowds or drawings of people is never picked."""
    labels = {image: openimages_to_yolo(dets) for image, dets in sorted(boxes.items())}
    usable = [image for image, lines in labels.items() if lines]
    random.Random(seed).shuffle(usable)
    return {image: labels[image] for image in usable[:limit]}


def write_yaml(name: str, out: Path) -> Path:
    yaml_path = DATA / f"{name}.yaml"
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n"
    )
    return yaml_path


def remove_others(folder: Path, keep: dict[str, list[str]]) -> int:
    """Delete what an earlier run chose, so a split holds this run's images and no more."""
    others = [path for path in folder.iterdir() if path.stem not in keep]
    for path in others:
        path.unlink()
    return len(others)


def cmd_openimages(args: argparse.Namespace) -> None:
    out = DATA / "openimages"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    names = raw / "class-descriptions.csv"
    if not fetch(OPENIMAGES_CLASSES, names):
        raise SystemExit(f"could not download {OPENIMAGES_CLASSES}")
    classes = person_classes(names)
    short: list[str] = []
    for split, limit in (("train", args.max_train), ("val", args.max_val)):
        oi_split, url = OPENIMAGES_BOXES[split]
        table = raw / Path(url).name
        print(f"[download] {table.name}")
        if not fetch(url, table):
            raise SystemExit(f"could not download {url}")
        chosen = choose_images(read_boxes(table, classes), limit, args.seed)
        img_out, lbl_out = out / "images" / split, out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        def get(image: str, oi_split: str = oi_split, img_out: Path = img_out) -> bool:
            url = OPENIMAGES_IMAGE.format(split=oi_split, image=image)
            return fetch(url, img_out / f"{image}.jpg")

        with ThreadPoolExecutor(WORKERS) as pool:
            done = list(pool.map(get, chosen))
        kept = people = 0
        for (image, lines), ok in zip(chosen.items(), done, strict=True):
            if ok:
                (lbl_out / f"{image}.txt").write_text("\n".join(lines) + "\n")
                kept += 1
                people += len(lines)
        print(f"[{split}] {kept} images, {people} people, {len(chosen) - kept} failed")
        if kept < len(chosen):  # an outage must not cost the images a split already has
            print(f"[{split}] earlier files kept, as {len(chosen) - kept} downloads failed")
            short.append(split)
            continue
        removed = remove_others(img_out, chosen) + remove_others(lbl_out, chosen)
        if removed:
            print(f"[{split}] removed {removed} files of an earlier selection")

    print(f"[done] dataset config written to {write_yaml('openimages', out)}")
    print(
        "[note] attribution: Open Images V7 by Google, annotations CC BY 4.0, "
        "images CC BY 2.0 (verify per image before redistributing images)"
    )
    if short:
        raise SystemExit(f"incomplete: {', '.join(short)} may hold files of an earlier selection")


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
