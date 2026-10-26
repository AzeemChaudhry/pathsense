"""Tracker wrapper around the Roboflow `trackers` library.

Phase 1 TODO: expose `build_tracker(name, **params)` for "bytetrack", "botsort" and "ocsort",
returning an object with `update(detections, frame=None) -> sv.Detections` (with tracker_id).
"""
