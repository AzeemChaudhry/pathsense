# Training data and fine tuning (Phase 4)

## Get the data

Two sources, one per purpose:

| Source | Use it for | Command |
| --- | --- | --- |
| **Open Images V7** | Any model you deploy, share or sell | `uv run python train/get_data.py openimages --max-train 8000 --crowded-share 0.5` |
| CrowdHuman | Portfolio benchmarks on dense crowds only | `uv run python train/get_data.py crowdhuman --max-train 5000` |

Run `uv sync --extra data` once first.

### Open Images (commercial use allowed)

Downloads Open Images V7 images that contain people from Google's published files, maps `Person`, `Man`,
`Woman`, `Boy` and `Girl` to one `person` class, drops group boxes (one box around a crowd)
and depictions (posters, statues, drawings), and writes YOLO labels plus
`data/openimages.yaml`. Annotations are CC BY 4.0 and images are listed as CC BY 2.0, so a
model trained on it can be deployed publicly or commercially with attribution:
"Trained on Open Images V7 (Google), CC BY 4.0".

Box tables go to `data/openimages/raw` (2.3 GB for train). A stopped run skips finished files
and starts any partial one again, and an image that keeps failing is reported and left out.
Once every download has worked, files from an earlier selection are deleted, so a split holds
exactly this run's images: a smaller `--max-train` removes the rest. If any failed they are
kept and the run ends with an error. Group boxes and
depictions include boxes whose flag is unknown (-1). `--max-train` and `--max-val` count
images with at least one usable box, and a seed always picks the same ones.

Open Images is mostly a few large people per photo: a median 2 per image at a third of its
height. The MOT20 crowds hold 46 to 195 people a frame at a median 11 to 16% of the frame
height. Two options aim training at that:

- **Group boxes are greyed out.** A group box is one box around a crowd, so the people in it
  have no box each; left in the image they would likely be taught as background. 52,002
  train images have one. Ultralytics has no ignore regions for detection, so the box is
  filled with its padding grey (114), and each labelled person's box inside it is pasted
  back whole, with whatever else falls inside it: a third of the labelled people in these
  images stand mostly inside a group box. Whether greying helps training is not measured
  yet. An image that would be more than half grey is left out (14,625 train, 766 val),
  with `--keep-groups` too, so both choose the same images: in
  the first 2,000 image download (seed 0, half crowded), 27 were over half grey, and by eye
  the greyest were a few people cut out on a flat grey. `--keep-groups` turns all this off.
  Each images folder has a `.kind` file saying which way its images were made; a run with
  other options refuses that folder, since a kept image is never fetched again and the two
  kinds would mix unseen. A folder from before this marker exists is refused too: delete it
  and download again.
- **`--crowded-share 0.5`** draws half the train images from crowded ones: 5 or more people
  under 15% of the image height (49,567 train images qualify). The rest are random. Val is
  never oversampled, but it is greyed and loses its mostly grey images like train, so it is
  a random sample of the images kept, not of all Open Images, and its mAP does not compare
  with published Open Images numbers.

### CrowdHuman (non commercial)

This downloads CrowdHuman from the Hugging Face Hub (one train zip of about 3 GB plus val,
about 6 GB on disk), keeps only `person` boxes, drops crowd and reflection regions, clips
boxes to the image, and writes YOLO labels plus `data/crowdhuman.yaml`. Everything lands in
`data/`, which git ignores.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-train N` | all (~5,000) | random train subset; 5,000 suits a 4 GB GPU |
| `--max-val N` | 1000 | random val subset |
| `--all-train` | off | download all 3 train zips (15,000 images, ~8.4 GB) |
| `--box` | `fbox` | `fbox` full body (best for tracking) or `vbox` visible region |
| `--skip-download` | off | rebuild labels from files already in `data/raw` |

Overhead views (the PathSense edge case) can be added from Roboflow Universe:

```bash
export ROBOFLOW_API_KEY=...   # free, app.roboflow.com > Settings > API
uv run python train/get_data.py roboflow --workspace krupal --project overhead-person-94pcq --version 1
```

Check the license on the project page before using any Roboflow dataset.

## License notes

| Dataset | License | Use in PathSense |
| --- | --- | --- |
| Open Images V7 | annotations CC BY 4.0, images CC BY 2.0 | Training any deployable model; credit Google |
| CrowdHuman | CC BY NC 4.0, no redistribution of images | Portfolio and research only; never publish the images; weights trained on it are not for commercial use |
| MOT17 / MOT20 | CC BY NC SA 3.0 | Evaluation only (Phase 3) |
| Roboflow Universe projects | varies, often CC BY 4.0 | Check each project |

## Train (first run)

```bash
uv run yolo detect train model=yolo26n.pt data=data/openimages.yaml \
    imgsz=640 epochs=30 batch=8 workers=4 amp=True
```

`batch=8` fits a GTX 1650 (4 GB). Log the result in NOTES.md and the README benchmark table.
