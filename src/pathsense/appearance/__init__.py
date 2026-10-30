"""Optional: tell people apart by how they look (`appearance.enabled` in the config, off).

    Embedder             a box -> a unit vector describing clothing (embed, osnet)
    AppearanceByteTrack  ByteTrack's first matching pass with looks (tracker)
    AppearanceStitcher   the Stitcher, refusing links that look wrong (stitch)
    AppearanceLine       the bounce rule decided by looks (line)

Off by default because on the MOT scenes it did not earn its cost: an embedding for every
person on every frame (docs/TRACKER_STUDY.md, Appearance). It stays so that scenes the
study could not test, such as people leaving a shop and coming back, can be measured.

Vectors stay in memory for one video: no faces, no names, nothing kept afterwards. Only
`pathsense tune` and the study save them, for research clips, in data/cache.
"""

from pathsense.appearance.embed import Embedder
from pathsense.appearance.line import AppearanceLine
from pathsense.appearance.stitch import AppearanceStitcher
from pathsense.appearance.tracker import AppearanceByteTrack

__all__ = ["AppearanceByteTrack", "AppearanceLine", "AppearanceStitcher", "Embedder"]
