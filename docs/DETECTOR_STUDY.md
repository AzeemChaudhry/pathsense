# Detector study: model, input size and confidence

Phase 3, added before the gate ([PLAN.md](PLAN.md)). The tracker study varied everything
after the detector and held the detector fixed. This study varies the detector, on far
more cameras: 7 MOT scenes and 80 from [PersonPath22](DATASETS.md), 56,564 frames, each run
through 10 detector settings and up to 11 tracker setups at up to 4 confidence levels.

## In short

| scene F1, held out scenes (39 cameras never used to choose) | |
| --- | --- |
| `yolo26n` at 640, the default | 0.411 |
| `yolo26n` at 1920, what scene files use | 0.528 |
| `yolo26s` at 1920 | 0.567 |
| **`yolo26s` at 1920, confidence 0.3, crowd settings, low bar** | **0.600** |

- **Input size matters most, then the model.** Going from 640 to 1920 is worth more than
  every tracker setting together. The best setups across trackers differ by about 0.02.
- **A bigger model helps, but needs its own settings.** `yolo26s` is less sure of itself
  than `yolo26n` in dense night crowds. With ByteTrack's first pass bar at 0.6, it recalled
  28% of MOT20-05's crossings. `high_conf_det_threshold` was never swept before. Lowering it
  took that scene from F1 0.42 to 0.70.
- **No setup wins everywhere.** The chosen one improves the held out scenes and MOT20-03,
  but loses to the scene files on MOT20-05 and on the sample clips. So **no default changed**.
- **Choosing per scene from a scan gained nothing.** Tried with one split on any measure,
  scored on held out folds: 0.596 against 0.594 for one setup.
- **Larger is not always better.** Above 1920, `yolo26n` on MOT20-02 finds few more people
  and counts more wrong crossings (64 at 1920, 95 at 3200), at 3.5 times the time.
- **This is too slow for the browser as it stands.** `yolo26s` at 1920 is about 10 FPS on
  the GTX 1650 and 1.9 on a CPU; `yolo26n` at 640 is 39.9 on a CPU. Phase 5 must measure
  what it can afford, since a small input size is the largest loss found here.

## Setup

Choosing scenes are MOT plus 40 PersonPath22 training videos; `test` holds 40 others.
Scenes with fewer than 4 true crossings are not scored, for any setup, leaving 45 choosing
and 39 test scenes. Setups compare by mean F1 per scene, so each camera counts once and one
crowd cannot decide alone. Detections are saved once with confidence 0.1. Other confidences
filter those boxes, which matches a detector set to them.

## Every model and size

Confidence 0.1, default tracker, all 45 choosing scenes. Time is the detector alone on the
GTX 1650.

| detector | size | ms a frame | MOT | PersonPath22 | all | found | wrong | missed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26n | 640 | 7 | 0.423 | 0.416 | 0.417 | 318 | 132 | 1,738 |
| yolo26s | 640 | 13 | 0.481 | 0.448 | 0.453 | 365 | 148 | 1,691 |
| yolo26n | 960 | 12 | 0.616 | 0.468 | 0.491 | 678 | 202 | 1,378 |
| yolo26n | 1280 | 19 | 0.739 | 0.512 | 0.547 | 1,010 | 340 | 1,046 |
| yolo26s | 1280 | 44 | 0.682 | 0.544 | 0.565 | 769 | 217 | 1,287 |
| yolo26n | 1600 | 29 | 0.747 | 0.535 | 0.568 | 1,177 | 452 | 879 |
| yolo26n | 1920 | 38 | 0.776 | 0.553 | 0.587 | 1,230 | 560 | 826 |
| yolo26s | 1920 | 96 | 0.778 | 0.579 | 0.610 | 1,146 | 313 | 910 |
| rfdetr-nano | 384 | 38 | 0.486 | 0.425 | 0.434 | 384 | 238 | 1,672 |
| rfdetr-small | 512 | 65 | 0.502 | 0.479 | 0.482 | 427 | 202 | 1,629 |

RF-DETR was only run at sizes near its own defaults. Larger sizes were too slow here
(`rfdetr-small` at 960 took 226 ms a frame; `rfdetr-nano` at 640, 91 ms, scene F1 0.526 on
MOT), so this does not compare the models at equal size. Bars were also set for YOLO.

Best overall: `yolo26s` at 1920, confidence 0.3, `crowd, low bar`, 0.630 (MOT 0.791,
PersonPath22 0.600). Best possible with a perfect choice per scene: 0.746, though much
of that gap is chance from picking the best of 110 setups for each scene.

## The scene analyzer

`pathsense analyze <video>` and `pathsense run --auto` scan 40 frames from the first two
minutes, describe the scene, and take the first rule that fits in `configs/analyzer.yaml`.
[Chameleon](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/05/Chameleon_SIGCOMM2018.pdf)
argued that sizes and speeds of objects are what settle the best setup for a camera.

We searched one split on six measures across the top 30 setups, with five folds. Fitted on
every scene it looked like a gain (0.628), but held out it was none (0.596 against 0.594),
and each fold picked a different rule. So today's rules are one setup for all, which beats
the default on any unseen video. The mechanism is what Phase 4 needs: rules are data, so
they can be fit again once the trained detectors exist, with no code change. It never sets
a line or a zone, and `pathsense tune` remains better when truth exists.

## Gate table

Scene files with their own settings, against the chosen setup. Precision and recall need
labels, so sample clips have counts only; `subway` has no truth. FPS is detector plus
tracking on the GTX 1650. Market-square uses tiling, hence its speed. Count error includes
zones. CPU FPS, detector only: `yolo26n` at 640 39.9, at 1280 11.5, at 1920 5.4; `yolo26s`
at 1920 1.9.

| scene | setup | IN / OUT ours | IN / OUT true | count error | wrong | missed | IN precision / recall | OUT precision / recall | HOTA | IDF1 | MOTA | ID switches | fragments | IDs | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02 | as shipped | 0 / 9 | 1 / 10 | 9 | 0 | 2 | 1.00 / 0.00 | 1.00 / 0.90 | 31.4 | 39.3 | 33.1 | 122 | 264 | 78 | 26.1 |
| MOT17-02 | chosen | 0 / 8 | 1 / 10 | 8 | 0 | 3 | 1.00 / 0.00 | 1.00 / 0.80 | 31.5 | 38.0 | 34.4 | 131 | 426 | 131 | 10.8 |
| MOT17-04 | as shipped | 8 / 4 | 8 / 5 | 4 | 0 | 1 | 1.00 / 1.00 | 1.00 / 0.80 | 47.9 | 55.5 | 49.5 | 128 | 368 | 83 | 24.8 |
| MOT17-04 | chosen | 8 / 6 | 8 / 5 | 27 | 1 | 0 | 1.00 / 1.00 | 0.83 / 1.00 | 53.6 | 61.9 | 52.5 | 133 | 540 | 170 | 10.7 |
| MOT17-09 | as shipped | 3 / 7 | 1 / 10 | 7 | 2 | 3 | 0.33 / 1.00 | 1.00 / 0.70 | 40.5 | 49.8 | 53.3 | 55 | 85 | 28 | 113.2 |
| MOT17-09 | chosen | 4 / 11 | 1 / 10 | 8 | 4 | 0 | 0.25 / 1.00 | 0.91 / 1.00 | 46.2 | 55.6 | 61.0 | 57 | 83 | 64 | 10.8 |
| MOT20-01 | as shipped | 10 / 11 | 14 / 21 | 16 | 2 | 16 | 1.00 / 0.71 | 0.82 / 0.43 | 38.5 | 47.4 | 43.8 | 267 | 533 | 85 | 23.4 |
| MOT20-01 | chosen | 10 / 14 | 14 / 21 | 16 | 2 | 13 | 1.00 / 0.71 | 0.86 / 0.57 | 42.9 | 53.4 | 50.4 | 209 | 566 | 139 | 10.7 |
| MOT20-02 | as shipped | 150 / 53 | 157 / 33 | 36 | 64 | 51 | 0.78 / 0.75 | 0.42 / 0.67 | 34.7 | 43.4 | 44.2 | 1285 | 3285 | 359 | 23.8 |
| MOT20-02 | chosen | 154 / 45 | 157 / 33 | 136 | 46 | 37 | 0.84 / 0.83 | 0.51 / 0.70 | 36.5 | 45.4 | 48.5 | 1321 | 3578 | 678 | 10.6 |
| MOT20-03 | as shipped | 11 / 322 | 4 / 376 | 118 | 46 | 93 | 0.27 / 0.75 | 0.88 / 0.76 | 38.2 | 55.1 | 51.1 | 1626 | 8026 | 931 | 16.3 |
| MOT20-03 | chosen | 8 / 306 | 4 / 376 | 195 | 23 | 89 | 0.38 / 0.75 | 0.94 / 0.77 | 39.4 | 58.9 | 52.5 | 1053 | 8815 | 1009 | 7.3 |
| MOT20-05 | as shipped | 41 / 411 | 46 / 594 | 231 | 83 | 271 | 0.41 / 0.37 | 0.86 / 0.59 | 27.0 | 36.6 | 36.9 | 2402 | 9582 | 1284 | 17.9 |
| MOT20-05 | chosen | 16 / 251 | 46 / 594 | 482 | 33 | 406 | 0.50 / 0.17 | 0.90 / 0.38 | 17.0 | 22.0 | 17.9 | 1187 | 10769 | 1171 | 8.6 |
| people-walking | as shipped | 11 / 7 | 11 / 7 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 39 | 124.0 |
| people-walking | chosen | 12 / 7 | 11 / 7 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 97 | 10.3 |
| grocery-store | as shipped | 1 / 0 | 1 / ? | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1 | 25.1 |
| grocery-store | chosen | 2 / 0 | 1 / ? | 6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 7 | 10.2 |
| market-square | as shipped | 1 / 16 | 0 / 16 | 2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 115 | 3.6 |
| market-square | chosen | 1 / 20 | 0 / 16 | 5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 188 | 0.3 |
| subway | as shipped | 6 / 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 65 | 23.5 |
| subway | chosen | 15 / 14 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 221 | 10.1 |

Against the gate:

- **MOT20-03 false crossings:** 46 to 23 with the chosen setup, and 93 to 89 missed. Met.
- **Check scenes:** 68 to 53 wrong and 73 to 53 missed. Met.
- **MOT20-05, the other crowd:** 83 to 33 wrong but 271 to 406 missed, and HOTA 27.0 to 17.0.
  Filtering at confidence 0.3 costs it the most: at 0.1 it found 327 where this finds 234.
- **Sample clips:** the scene files are better: people-walking 12 IN, not 11; grocery-store
  2 IN and 7 IDs, not 1 and 1; market-square 20 OUT, not 16. Lower bars split tracks.
- **Subway:** cannot be scored. The chosen setup counts 15 and 14 against 6 and 10, with 221
  IDs against 65, so more detections, and likely more splits. It needs box labels.

## Limits

- **PersonPath22's lines are automatic:** square to the main flow, placed where most tracks
  cross. They score counting wherever people flow, not at a doorway anyone cares about.
- **PersonPath22's labels are at 5 FPS** and interpolated between, so its fast or turning
  people are approximate. Its scores are lower than MOT's partly for these reasons.
- **The held out scenes were looked at twice.** The final row was added after the first
  look showed a failure on MOT20-05. It was chosen on the other scenes alone, but the
  decision to look for it came from the gate, not the test. Its 0.600 is a little kind.
- **About 16 crossings a scene.** 706 true crossings in the 39 test scenes, so small
  differences between setups (0.579 against 0.577) mean nothing.
- **55 of 236 videos are absent:** external sources Amazon may not redistribute.
- **Not tested:** other YOLO sizes (`yolo26m`), tiling at scale, frame skipping, and each
  bar for each model. `pathsense tune`'s per scene picks were not rechecked on held out time.
- **No trained model yet.** Everything here uses COCO weights. Phase 4 changes all of it.

## Reproduce

```bash
uv run python scripts/personpath22.py            # after the download in docs/DATASETS.md
uv run python scripts/detector_study detect      # GPU, about 3 h; run under systemd-inhibit
uv run python scripts/detector_study stage1      # every model and size
uv run python scripts/detector_study stage2      # the best, with every setting
uv run python scripts/detector_study report
uv run python scripts/detector_study test        # held out scenes
uv run python scripts/detector_study/gate.py     # the gate table
```

Every step skips what it has done. Defined in `configs/study/detectors.yaml`.
