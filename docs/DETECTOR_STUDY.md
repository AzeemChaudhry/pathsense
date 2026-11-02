# Detector study: model, input size and confidence

Phase 3, added before the gate ([PLAN.md](PLAN.md)). The tracker study varied everything
after the detector and held the detector fixed. This study varies the detector, on far
more cameras: 7 MOT scenes and 80 from [PersonPath22](DATASETS.md), 56,564 frames, each run
through 10 detector settings and up to 11 tracker setups at up to 4 confidence levels.

## In short

| scene F1, held out scenes (39 PersonPath22 cameras never used to choose) | |
| --- | --- |
| `yolo26n` at 640, the default | 0.411 |
| `yolo26n` at 1920, what scene files use | 0.528 |
| `yolo26s` at 1920 | 0.567 |
| **`yolo26s` at 1920, confidence 0.3, crowd settings, low bar** | **0.600** |

- **Input size matters most, then the model,** of what was varied. Going from 640 to 1920 is
  worth more than every tracker setting together. The best setups across trackers differ by
  about 0.02. The detector's head and NMS overlap were left at the library's defaults.
- **A bigger model helps, but needs its own settings.** `yolo26s` is less sure of itself
  than `yolo26n` in dense night crowds. With ByteTrack's first pass bar at 0.6, it recalled
  28% of MOT20-05's crossings. `high_conf_det_threshold` was never swept before. Lowering it
  took that scene from F1 0.42 to 0.70 at confidence 0.1. The chosen setup, at 0.3, gets
  0.52 there, against 0.68 for its scene file.
- **No setup wins everywhere.** The chosen one improves the held out scenes and MOT20-03,
  but loses to the scene files on MOT20-05. On the seven MOT scenes its line count error is
  worse on three and better on four; zone count error is worse on six. So **no default
  changed**; it is what `pathsense run --auto` uses, with `tracker.stitch_top` 0.1.
- **The held out scenes say nothing about crowds.** All 39 are PersonPath22 and mostly
  sparse. Every MOT scene helped choose the setup, so no crowd was held out.
- **Choosing per scene from a scan lost to one setup.** One split on any measure, scored on
  held out folds: 0.601 against 0.619.
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
of that gap is chance from picking the best of 146 setups for each scene.

The sweep above 1920 quoted earlier was a one off on MOT20-02, MOT17-02 and subway with
`yolo26n`; it is in NOTES.md, not in this study's scripts.

## The scene analyzer

`pathsense analyze <video>` and `pathsense run --auto` scan 40 frames from the first two
minutes, describe the scene, and take the first rule that fits in `configs/analyzer.yaml`.
[Chameleon](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/05/Chameleon_SIGCOMM2018.pdf)
argued that sizes and speeds of objects are what settle the best setup for a camera.

`scripts/detector_study/rules.py` searched one split on the scan's five measures across the
top 30 setups, with five folds. Fitted on every scene it looked like a small gain (0.635
against 0.630). Held out it lost (0.601 against 0.619), and the folds disagreed on the rule.
So today's rules are one setup for all, and **`--auto` currently just means "use that
setup"**: the scan reports the scene but changes nothing. It is better than the default on
average across unseen cameras, not on each one, and it is not the default. The mechanism is
what Phase 4 needs: rules are data, so they can be fit again once the trained detectors
exist, with no code change. It never sets a line or a zone, refuses a scene file, and
`pathsense tune` remains better when truth exists.

## Gate table

Scene files with their own settings, against the chosen setup as `--auto` runs it, read from
`configs/analyzer.yaml`, so with `tracker.stitch_top` 0.1. The sample clips have every
crossing listed by hand from plain frames, so they score crossings but have no box metrics;
`subway` has no truth. FPS is detector plus
tracking on the GTX 1650. Market-square uses tiling, hence its speed. Line error is the IN
and OUT counts against the truth, zone error the zone counts, each summed. CPU FPS, detector only: `yolo26n` at 640 39.9, at 1280 11.5, at 1920 5.4; `yolo26s`
at 1920 1.9.

| scene | setup | IN / OUT ours | IN / OUT true | line error | zone error | wrong | missed | IN precision / recall | OUT precision / recall | HOTA | IDF1 | MOTA | ID switches | fragments | IDs | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02 | as shipped | 0 / 9 | 1 / 10 | 2 | 7 | 0 | 2 | 1.00 / 0.00 | 1.00 / 0.90 | 31.4 | 39.3 | 33.1 | 122 | 264 | 78 | 25.9 |
| MOT17-02 | chosen | 0 / 8 | 1 / 10 | 3 | 5 | 0 | 3 | 1.00 / 0.00 | 1.00 / 0.80 | 31.1 | 37.3 | 34.4 | 123 | 426 | 127 | 10.6 |
| MOT17-04 | as shipped | 8 / 4 | 8 / 5 | 1 | 3 | 0 | 1 | 1.00 / 1.00 | 1.00 / 0.80 | 47.9 | 55.5 | 49.5 | 128 | 368 | 83 | 24.6 |
| MOT17-04 | chosen | 8 / 5 | 8 / 5 | 0 | 23 | 0 | 0 | 1.00 / 1.00 | 1.00 / 1.00 | 54.0 | 62.1 | 52.6 | 126 | 540 | 163 | 10.6 |
| MOT17-09 | as shipped | 3 / 7 | 1 / 10 | 5 | 2 | 2 | 3 | 0.33 / 1.00 | 1.00 / 0.70 | 40.5 | 49.8 | 53.3 | 55 | 85 | 28 | 119.5 |
| MOT17-09 | chosen | 4 / 11 | 1 / 10 | 4 | 3 | 4 | 0 | 0.25 / 1.00 | 0.91 / 1.00 | 46.2 | 55.6 | 61.0 | 57 | 83 | 61 | 10.8 |
| MOT20-01 | as shipped | 10 / 11 | 14 / 21 | 14 | 2 | 2 | 16 | 1.00 / 0.71 | 0.82 / 0.43 | 38.5 | 47.4 | 43.8 | 267 | 533 | 85 | 24.5 |
| MOT20-01 | chosen | 10 / 13 | 14 / 21 | 12 | 9 | 2 | 14 | 1.00 / 0.71 | 0.85 / 0.52 | 44.3 | 56.0 | 50.5 | 187 | 567 | 122 | 10.6 |
| MOT20-02 | as shipped | 150 / 53 | 157 / 33 | 27 | 9 | 64 | 51 | 0.78 / 0.75 | 0.42 / 0.67 | 34.7 | 43.4 | 44.2 | 1285 | 3285 | 359 | 23.5 |
| MOT20-02 | chosen | 155 / 40 | 157 / 33 | 9 | 91 | 42 | 37 | 0.83 / 0.82 | 0.60 / 0.73 | 37.5 | 47.1 | 48.6 | 1222 | 3577 | 601 | 10.5 |
| MOT20-03 | as shipped | 11 / 322 | 4 / 376 | 61 | 57 | 46 | 93 | 0.27 / 0.75 | 0.88 / 0.76 | 38.2 | 55.1 | 51.1 | 1626 | 8026 | 931 | 15.9 |
| MOT20-03 | chosen | 7 / 305 | 4 / 376 | 74 | 122 | 20 | 88 | 0.43 / 0.75 | 0.95 / 0.77 | 39.9 | 59.7 | 52.5 | 982 | 8816 | 956 | 7.3 |
| MOT20-05 | as shipped | 41 / 411 | 46 / 594 | 188 | 43 | 83 | 271 | 0.41 / 0.37 | 0.86 / 0.59 | 27.0 | 36.6 | 36.9 | 2402 | 9582 | 1284 | 17.6 |
| MOT20-05 | chosen | 16 / 250 | 46 / 594 | 374 | 110 | 32 | 406 | 0.50 / 0.17 | 0.90 / 0.38 | 17.2 | 22.2 | 17.9 | 1168 | 10769 | 1159 | 8.5 |
| people-walking | as shipped | 11 / 7 | 12 / 7 | 1 | 0 | 0 | 1 | 1.00 / 0.92 | 1.00 / 1.00 | n/a | n/a | n/a | n/a | n/a | 39 | 115.2 |
| people-walking | chosen | 12 / 7 | 12 / 7 | 0 | 0 | 0 | 0 | 1.00 / 1.00 | 1.00 / 1.00 | n/a | n/a | n/a | n/a | n/a | 93 | 10.2 |
| grocery-store | as shipped | 1 / 0 | 1 / 0 | 0 | 0 | 0 | 0 | 1.00 / 1.00 | 1.00 / 1.00 | n/a | n/a | n/a | n/a | n/a | 1 | 25.0 |
| grocery-store | chosen | 1 / 0 | 1 / 0 | 0 | 2 | 0 | 0 | 1.00 / 1.00 | 1.00 / 1.00 | n/a | n/a | n/a | n/a | n/a | 3 | 10.2 |
| market-square | as shipped | 1 / 16 | 1 / 18 | 2 | 1 | 0 | 2 | 1.00 / 1.00 | 1.00 / 0.89 | n/a | n/a | n/a | n/a | n/a | 115 | 3.6 |
| market-square | chosen | 1 / 20 | 1 / 18 | 2 | 0 | 2 | 0 | 1.00 / 1.00 | 0.90 / 1.00 | n/a | n/a | n/a | n/a | n/a | 183 | 0.3 |
| subway | as shipped | 6 / 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 65 | 23.4 |
| subway | chosen | 13 / 15 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 215 | 10.0 |

Against the gate. PLAN.md tunes on MOT20-03 and MOT20-05 and validates on the other MOT
scenes and the sample clips. All seven MOT scenes also helped choose this setup, so the only
check it never saw is the sample clips, and their scene files were tuned on those same clips,
which favours "as shipped". The listed crossings take their time and x from the chosen run
once confirmed in plain frames, which favours "chosen". On them the chosen setup is better on
one and about level on two, each on few crossings (19, 19 and 1). The gate names
a tracker
against ByteTrack; this table changes the detector and tracker together against each scene
file, so a gain cannot be credited to the tracker alone.

- **MOT20-03 false crossings:** 46 to 20 with the chosen setup, and 93 to 88 missed, on a
  scene used to choose. Line error rises from 61 to 74, as it counts fewer in total and
  false crossings had been filling that gap. Zone error rises from 57 to 122.
- **Check scenes:** 68 to 48 wrong and 73 to 54 missed in total, on scenes used to choose.
  Wrong rose on one (MOT17-09 2 to 4), and MOT17-02 lost a real crossing (OUT recall 0.90 to
  0.80). Line error is better on four and worse on MOT17-02. Zone error is worse on four
  (MOT17-04 3 to 23, MOT17-09 2 to 3, MOT20-01 2 to 9, MOT20-02 9 to 91), perhaps because
  lower bars add tracks, which was not tested.
- **MOT20-05, the other crowd:** 83 to 32 wrong but 271 to 406 missed, line error 188 to
  374, zone error 43 to 110, and HOTA 27.0 to 17.2.
  Filtering at confidence 0.3 costs it the most: at 0.1 it found 327 where this finds 234.
- **Sample clips, listed crossing by crossing:**
  - people-walking: 0 wrong and 0 missed against 0 and 1. The missed one is a man at the left
    edge. The first hand count, 11 IN, missed him too; the clip is 12 IN.
  - market-square: the count was wrong, 18 OUT and 1 IN, not 16 and 0. Line error is 2 each
    way: 2 wrong against 2 missed. From the frames, shipped's total flatters it: it misses 4
    who cross, and its doubles earn credit, as matching by time and place cannot tell people
    apart. One of the 18 stays unsure, feet hidden at the right edge.
    F1 is 0.944 against 0.950 at 2 s and 200 px, 0.889 against 0.950 at 1 s and 100 px, and
    0.778 against 0.900 at 0.5 s and 100 px (`scripts/detector_study/tolerance.py`). "About
    level" holds at the loosest setting only. The tighter gaps partly reflect where the times
    came from, so this leans to chosen but does not settle it.
  - grocery-store: level on crossings, zone error 2 against 0. Without `stitch_top` one
    shopper is counted twice: the track breaks while they browse, and the new box then jumps
    from upper body to whole body across the line. The 0.3 confidence filter alone causes it
    on `yolo26s`; see [linking by the top of the box](#linking-by-the-top-of-the-box).
- **Subway:** cannot be scored. The chosen setup counts 13 and 15 against 6 and 10, with 215
  IDs against 65, so more detections, and likely more splits. It needs box labels.

## Head and overlap

Every YOLO26 result above used the one-to-many head with NMS at IoU 0.7, not the NMS free
head: Ultralytics 8.4.190 selects that head only with `nms=False`, which this study never
set. A first look at 40 frames from each of MOT20-05, MOT17-04 and MOT17-09, at 1920 and
scored box by box against the labels, not on crossings, so it is weak evidence
(`scripts/detector_study/frames.py`):

| question | result, best F1 per scene |
| --- | --- |
| NMS free head against NMS, five sizes | worse in 11 of 15, better in 3, tied in 1, by small gaps |
| IoU 0.9 against 0.7, three sizes | worse in all 9 |
| IoU 0.5 against 0.7, two sizes | no real change |
| `max_det` 2000 against 300, NMS free head | no change, on six frames from three scenes |
| largest model | best on two scenes, by under 0.03, at 663 ms a frame against 46 |

The best confidence differs by scene: 0.1 in the dense crowd, 0.2 to 0.4 elsewhere. Models
are compared on crossings with `scripts/detector_study models`, on the scenes every model
finished, by paired gaps per scene. A gap under 0.03, or a 95 % interval that includes 0, is a tie.

Speeds are all from the same GTX 1650, saved boxes from the first study included. The NMS
head's cap was not tested on its own: the largest number of boxes in any saved frame of the
study is 207, so raising the library's cap of
300 to 1000 changes none of the saved boxes above.

## Models by density

A mean over all scenes hides the crowds: sparse scenes outnumber them and pick the setting.
`yolo26s` at its pooled confidence of 0.3 scored 0.52 on MOT20-05, against 0.653 at 0.1. So
results are split by people per frame, with bands fixed before they were split. Each setting
is chosen with the scored scene left out, among 4 confidences and 4 tracker setups, at 1920:

| band | scenes | best models, left out mean scene F1 | setting of the best |
| --- | --- | --- | --- |
| sparse, under 25 | 19 | `yolo26s` NMS free 0.633; `yolo26n`, `yolo11s` and `yolo26s` tied with it | 0.4, crowd, low bar |
| busy, 25 to 60 | 19 | `yolo26s` 0.567; `yolo26n` 0.533 tied with it: a gap of -0.034, interval -0.075 to +0.006 | 0.3, crowd, low bar |
| crowd, 60 or more | 7 | `yolo11s` 0.776; `yolo26s` 0.774 tied with it | 0.2 and 0.1, crowd, low bar |

`yolo26s` is the only model never clearly behind a band's best: every other one is behind in
at least one band. On 7 to 19 scenes a tie means no difference was found, not that there is
none, so this is a reason to carry it forward, not proof that it is best. Each model is
compared with the band's leader only, so ties do not chain, and a leader that is the best of
several is favoured a little. The NMS free head is not free
of cost: `yolo26s` loses 0.072 on busy scenes and `yolo26n` loses 0.053 on sparse ones. These
are choosing scenes, and the crowd band is seven scenes, five of them from MOT. A summed count
error is ruled by a few crowds, so bands report the median per scene, for line counts only:
across all 45 scenes it is 5 or 6 for every model (zone error is reported beside it), and in the crowd band 7 for `yolo26s`, 11
for `yolo11s` and 16 for `yolo26n`. `yolo26x` was stopped
on its first video, and RF-DETR was not in this run.

Choosing confidence by density did not earn a rule. In the crowd band, 0.1 against 0.3 at
`yolo26s`'s best tracker setup adds 0.022, under the tie line: MOT20-05 gains 0.137, MOT20-01
0.054 and MOT20-03 0.032, while the other four lose up to 0.036 or do not change. The scan
cannot see the bands either: its `people` correlates 0.64 with labelled people per frame and
reads 18.5 on MOT20-05, which has 226. No threshold on it moved the mean scene F1 by more
than 0.003. So `yolo26s` keeps one setting, and MOT20-05 stays a known failure.

## Linking by the top of the box

`tracker.stitch_top`, off by default. On grocery-store the chosen setup counts one shopper
twice. As the cart hides their legs, the whole body box falls under the 0.3 filter and an
upper body box takes over: the height halves and the feet jump about 470 px, so the stitcher
refuses the link, but the top of the box moves 7 px. The new ID has no remembered height and
crosses the line when the whole body box returns.

The rule applies only to a box outside the stitcher's height range, and needs the top centre
near where it should be and a similar width. Results for the chosen setup, from saved boxes
(`scripts/detector_study/stitch_top.py`), wrong and missed crossings, off against 0.1:

| scenes | off | 0.1 |
| --- | --- | --- |
| grocery-store | 1 / 0, zone error 5 | 0 / 0, zone error 2 |
| people-walking, market-square | no event changes | no event changes |
| MOT, 7 | 109 / 548 | 100 / 548 |
| PersonPath22 chosen on, 38 | 194 / 312 | 196 / 310 |
| PersonPath22 held out, 39 | 129 / 344 | 120 / 350 |

- Outside grocery-store this is a trade, not a gain: fewer wrong crossings, some more missed.
  Mean scene F1 goes 0.791 to 0.795 on MOT, stays 0.600 on the scenes chosen on, and goes 0.600
  to 0.598 held out. MOT is the only clear gain, and MOT was used to shape the rule.
- The case for it is one wrong crossing in one scene, on a clip the rule was designed around.
- Any value from 0.1 to 0.35 fixes grocery-store. 0.1 is the most cautious, though the rule's
  shape was refined after seeing these results. An earlier, wider version also fired on boxes
  of the same height and, from 0.15, added a wrong crossing on market-square.
- **Links checked against the MOT labels**, counting a link when both boxes overlap a labelled
  person at IoU 0.3. At 0.1, 289 of 364 links could be judged: 205 joined the same person and
  84 different people, so about 3 in 10 are wrong. Narrowing cut the number of links, not
  much the wrong share (32% before). The check is weakest for these boxes: a partial box is
  exactly what can overlap a neighbour.
- The same check on the existing rule judges 2,219 of 2,369 links: 576 same and 1,643
  different, about 74%. That is a rough count of a different set of links, so not a like for
  like comparison, but it may mean the stitcher links the wrong person often in crowds.
- **That look, by eye and by turning it off.** Crops of sampled links on MOT20-05, MOT20-03
  and MOT17-04 agree with the labels: links judged different join different people, often a
  body apart after a gap of a few frames. Turning the stitcher off for the chosen setup
  (`stitch_seconds` 0) gives wrong / missed and scene F1:

  | scenes | stitch 1 s | off |
  | --- | --- | --- |
  | sample clips | 3 / 0, 0.872 | 2 / 0, 0.880 |
  | MOT | 109 / 548, 0.791 | 86 / 555, 0.803 |
  | PersonPath22 chosen on, 38 | 194 / 312, 0.600 | 162 / 338, 0.587 |
  | PersonPath22 held out, 39 | 129 / 344, 0.600 | 81 / 382, 0.570 |

  In crowds most of its wrong links add wrong crossings (MOT20-05 14 wrong off, 33 on), but
  on the sparser held out scenes it recovers more real ones than it adds wrong. So it stays
  on; stitching less where it is crowded is left for later. Measured before `stitch_top` was
  in `--auto`, so these rows are without it.
- The tracker default stays 0. Phase 3 closed with 0.1 in `--auto` (`configs/analyzer.yaml`),
  the setup in the gate table above.

## Limits

- **`tracker.stitch_top` rests on little:** one crossing on one clip it was designed around.
  Its link check uses MOT, which also shaped the rule, so it is not independent, and 75 of
  364 links could not be judged, which could move the wrong share.
- **PersonPath22's lines are automatic:** square to the main flow, placed where most tracks
  cross. They score counting wherever people flow, not at a doorway anyone cares about.
- **PersonPath22's labels are at 5 FPS** and interpolated between, so its fast or turning
  people are approximate. Its scores are lower than MOT's partly for these reasons.
- **The held out scenes were looked at twice.** The final row was added after the first
  look showed a failure on MOT20-05. It was chosen on the other scenes alone, but the
  decision to look for it came from the gate, not the test. Its 0.600 is a little kind.
- **About 16 crossings a scene.** 706 true crossings in the 39 test scenes, so small
  differences between setups (0.579 against 0.577) mean nothing.
- **The line maker is rough.** It counts any track spanning the line, even someone swaying
  on it, and does not check that the line reaches them; clipping an end to the frame can
  tilt the line. Scenes with fewer than 4 true crossings by the scorer's own rules are
  dropped, which removes the worst cases. Fixing it would change every scene and need a
  rerun, so it waits for the next rerun.
- **55 of 236 videos are absent** (external sources Amazon may not redistribute), and
  videos over 1,800 frames were left out, so the sample leans to short clips.
- **No crowd was held out.** The held out scenes are all PersonPath22, so a setup's result
  on crowds comes only from scenes that chose it.
- **Not tested on crossings:** `yolo26m` and `yolo26l`, NMS overlap and box cap (see Head and
  overlap), tiling at scale, frame skipping, and each bar for each model. Ultralytics runs
  YOLO26 on its one-to-many head with NMS at 0.7 unless told otherwise, so the tables above
  are NMS results. `pathsense tune`'s per scene picks were not rechecked on held out time.
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
uv run python scripts/detector_study/rules.py    # can a scan choose per scene?
uv run python scripts/detector_study/frames.py   # the 40 frame first look
uv run python scripts/detector_study models      # more models, on the choosing scenes
uv run python scripts/detector_study density     # those results by people per frame
```

Every step skips what it has done. Defined in `configs/study/detectors.yaml` and
`configs/study/models.yaml`.
