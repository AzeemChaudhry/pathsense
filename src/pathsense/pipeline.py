"""Glues the stages together, one frame at a time.

    frame -> Detector -> Tracker -> Analytics -> annotated frame + events

Phase 1 TODO: implement `Pipeline.process(frame)` using the Detector protocol from
`pathsense.detect` and the tracker from `pathsense.track`.
"""
