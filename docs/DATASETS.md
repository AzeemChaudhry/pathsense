# Dataset research

Researched 2026-10-27. Goal: people detection data that allows a model to be deployed
publicly (Hugging Face Space, website, client cameras), not only used in research.

**Decision:** train deployable models on **Open Images V7**. Use CrowdHuman only for
portfolio benchmarks. Treat MEVA as the next step for real security camera footage.

Download both with `train/get_data.py` (see `train/README.md`). Images never go into git:
CrowdHuman forbids redistribution, and the files are several GB.

## Summary

| Dataset | What it is | License | Commercial / public model? | Status in PathSense |
| --- | --- | --- | --- | --- |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | Google, 1.9M images, 16M boxes, 600 classes; people under Person, Man, Woman, Boy, Girl | Annotations CC BY 4.0; images listed as CC BY 2.0 | **Yes**, with attribution | `get_data.py openimages` |
| [CrowdHuman](https://hf.co/datasets/Funtion/CrowdHuman) | Megvii, 15,000 train / 4,370 val, about 23 people per image, full body, visible and head boxes | CC BY NC 4.0, no image redistribution | No | `get_data.py crowdhuman` (benchmarks only) |
| [PersonPath22](https://amazon-science.github.io/tracking-dataset/personpath22.html) | Amazon, 236 videos, mostly fixed cameras, every person boxed with an ID at 5 FPS; 181 downloadable, the rest are external | CC BY NC 4.0; authors ask for no identification or biometric use | No | Evaluation only (Phase 3): `docs/DETECTOR_STUDY.md`. Appearance stays off on it |
| [MEVA](https://mevadata.org/) | 328 h of real security camera video (EO, thermal, indoor and outdoor) plus drone video, free on AWS S3 | CC BY 4.0 | Yes | Backlog: auto label frames with a large RF-DETR, then train on them |
| MOT17 / MOT20 | Pedestrian tracking benchmarks | CC BY NC SA 3.0 | No | Evaluation only (Phase 3) |
| [CEPDOF / WEPDTOF / HABBOF](https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof) | Overhead fisheye people, rotated boxes | Non commercial, request form | No | Not used |
| [GOTPD](https://geintra-uah.org/datasets/gotpd1) | Overhead Kinect, depth and infrared (RGB only frontal) | CC BY SA 4.0 | Yes | Not used: not RGB overhead |
| [BDD100K](https://huggingface.co/datasets/ShantyCam/bdd100k-det) | Dashcam driving scenes | BSD style, commercial rights only for BDD/BAIR members | Needs Berkeley license | Not used |
| [Objects365](https://docs.ultralytics.com/datasets/detect/objects365/) | 1.7M images, Person class | Listed as CC BY 4.0 by Ultralytics; not confirmed at source | Unverified | Not used: unverified and very large |
| [Roboflow overhead person](https://universe.roboflow.com/krupal/overhead-person-94pcq) | Overhead people, community project | Unverified (page did not load) | Check page | `get_data.py roboflow` if the license allows |

PersonPath22 download (12.2 GB, no account needed), into `data/personpath22/`, needing no
third party script: `annotation/anno_amodal.zip`, `annotation/splits.json` and
`raw_data/videos.zip` from
`https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/dataset/`. Unzip the first
in `annotation/` and the last to `raw/videos/`. One connection is slow (about 1.3 MB/s),
so fetch byte ranges in parallel.

## Notes

- **Two separate licenses matter.** The data license (above) and the model license.
  Ultralytics YOLO26 is AGPL 3.0: fine for this public repo, but a closed client product
  needs an Ultralytics Enterprise license. RF-DETR is Apache 2.0, code and weights.
  Commercial track = RF-DETR + Open Images (+ the client's own footage).
- **Open Images caveat:** Google states it makes no warranty on each image's license and
  asks users to verify per image. Training on the images and publishing the model is the
  normal use; redistributing the images themselves needs that per image check.
- **Attribution line** for model cards and the README:
  "Trained on Open Images V7 (Google), annotations CC BY 4.0."
- **Bus and overhead cameras:** passenger counting usually uses a camera above the door
  looking down and counts heads crossing a line. Open Images has few top down views, so
  the real fix is client footage or auto labelled MEVA frames.
- Not legal advice. Read each dataset's terms before selling a model to a client.
