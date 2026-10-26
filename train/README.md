# Training data and fine tuning (Phase 4)

## Get the data

```bash
uv sync --extra data
uv run python train/get_data.py crowdhuman --max-train 5000
```

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
| CrowdHuman | CC BY NC 4.0, no redistribution of images | Portfolio and research only; never publish the images; weights trained on it are not for paid client work |
| MOT17 / MOT20 | CC BY NC SA 3.0 | Evaluation only (Phase 3) |
| Roboflow Universe projects | varies, often CC BY 4.0 | Check each project |

## Train (first run)

```bash
uv run yolo detect train model=yolo26n.pt data=data/crowdhuman.yaml \
    imgsz=640 epochs=30 batch=8 workers=4 amp=True
```

`batch=8` fits a GTX 1650 (4 GB). Log the result in NOTES.md and the README benchmark table.
