# Backlog

Improvements found while building and testing, for after the current phase. Each one is a
separate piece of work; none is started until the current phase's gate passes and it is
picked into a phase in [PLAN.md](PLAN.md). Most came from the four public sample scenes
(`configs/scenes/`, NOTES.md 2026-10-28).

## Detection: finding people the model misses

- **Zones guide detection.** Run the expensive passes (larger input, tiles) only inside the
  zones and lines that matter: the train doors, an aisle, a shop entrance. Far cheaper
  than tiling the whole frame, which also makes it possible in the browser.
- **Lower confidence where people are expected.** Inside a zone, or on a path the heatmap
  shows people use, start tracks from weaker detections (for example 0.4 instead of 0.6);
  keep 0.6 elsewhere, where weak boxes are more often shelves and posters.
- **Clusters.** People leaving a train together are boxed as a few blobs or not at all.
  Options to compare: a head detector (heads overlap less than bodies), a crowd density
  model that estimates how many people a blob holds, and motion (frame differences) to
  flag areas where something moves but nobody is detected.
- **Stronger teacher offline.** Run a large model (RF-DETR large, SAM 3) on recorded clips
  to label them, then compare or distil (already Stretch C in the plan).

## Tracking: keeping one ID per person

- **Appearance cue for re-entry.** A clothing colour histogram of the box, kept in memory
  for one video and never stored or compared across videos, added to the stitcher's
  position and motion check. Faces are never used (privacy rule in PLAN.md).
- **Tracker settings per scene.** `lost_track_buffer` and the activation threshold tuned by
  scene tuning (Phase 3) rather than one default.
- **Entry and exit points.** A new track that starts in the middle of the floor, away from
  doors and picture edges, is probably a lost person; prefer stitching it.
- **Stitching across an exit.** The stitcher can hand a person's ID to someone else who
  steps into the spot they just left within 2 s, at a door or the picture's edge. Refuse to
  stitch a person last seen at an exit, and cap how far their last speed is carried
  (review of PR 10). The MOT scores show how often it happens.
- **Stitch on corrected heights.** The stitcher's size check uses raw box heights; someone
  lost behind a cart and found whole can fail it. Use the `Feet` full height instead.

## Zones and lines

- **Floor plan (homography).** Map the camera picture to a top down floor plan with four
  points. Zones are then drawn on the plan, sizes and speeds are in metres, and a far
  person counts the same as a near one; the perspective problems (zones that look right
  but miss people at the back) go away.
- **Zone roles.** Mark zones as a door, a queue or a counter, so analytics know where
  people may appear and disappear, and queue zones report length and waiting time.
- **Per zone settings.** Hold time, margin and hidden legs set per zone, not per scene.
- **Occupancy events.** Write how many are inside a zone at fixed intervals to the event
  log, so questions like "how full was the platform at 8:15" have an answer.

## Heatmaps

- **Time slices.** One heatmap per minute or per segment (before, during, after a train),
  so changes over time show.
- **Traffic versus dwell.** Separate maps for where people walk and where they stand.
- **Flow.** Average walking direction per cell (arrows), showing main paths.
- **On the floor plan.** Drawn on the homography floor plan, so near and far areas compare.
- **Heatmap as a prior.** Where people are often seen tells detection and tracking where
  to look harder (see "lower confidence where people are expected"), and where boxes are
  never people (a poster), so they can be ignored.

## Scene tuning (Phase 3, then further)

- Start with `pathsense tune` as planned in Phase 3, then: tune live cameras from their
  first minutes, re-tune when the scene changes (lighting, layout), and suggest lines and
  zones from the heatmap's main paths.
