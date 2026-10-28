"""OpenCV window set up.

cv2.imshow opens its window with Qt, a GUI toolkit that opencv-python bundles inside the
wheel. That bundled copy is trimmed down: it has no fonts and only the X11 ("xcb") display
plugin. Importing cv2 even points QT_QPA_FONTDIR at an empty folder, so Qt prints
"Qt no longer ships fonts" on the first window. We point it at the DejaVu fonts the system
already has (Arch: TTF/, Debian and Ubuntu: truetype/dejavu/).
"""

from __future__ import annotations

import os

SYSTEM_FONT_DIRS = ("/usr/share/fonts/TTF", "/usr/share/fonts/truetype/dejavu")


def configure_qt() -> None:
    """Call after importing cv2 and before the first window; Qt reads these variables then."""
    # On a Wayland desktop Qt looks for a "wayland" plugin the wheel does not have, warns,
    # then falls back to X11. Asking for X11 ("xcb") directly skips the warning; the window
    # still shows normally through XWayland.
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    # Respect a font folder the user set themselves, as long as it exists.
    if os.path.isdir(os.environ.get("QT_QPA_FONTDIR", "")):
        return
    for font_dir in SYSTEM_FONT_DIRS:
        if os.path.isdir(font_dir):
            os.environ["QT_QPA_FONTDIR"] = font_dir
            return


def open_window(name: str, width: int, height: int, fit: tuple[int, int] = (1600, 900)) -> None:
    """A resizable window no larger than `fit`, keeping the picture's shape. A 4K or
    portrait video would otherwise open at full size, larger than the screen."""
    import cv2

    scale = min(1.0, fit[0] / width, fit[1] / height)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
