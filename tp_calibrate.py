#!/usr/bin/env python3
"""tp_calibrate: interactive pixel-coordinate inspector for SW2 frames.

Loads one frame from a video file (via ffmpeg, weave-decoded like tp_measure)
or a PNG, and lets the operator zoom, pan, and click to record pixel
coordinates of features on the chart. Used to calibrate tp_chart.py region
coordinates against real captures.

CLI:
    python tp_calibrate.py /path/to/capture.mov [--frame 60]
    python tp_calibrate.py --image /path/to/frame.png

Interaction:
    mouse wheel : zoom in / out, centred on the pointer
    left click  : print the current pixel (x, y) [and RGB] to stdout
    'r'         : reset view to fit the whole frame
    'q'         : quit
"""

from __future__ import annotations
import argparse
import sys
from typing import Tuple

import numpy as np


_ZOOM_FACTOR = 1.25


def _load_frame_from_video(path: str, frame_index: int) -> np.ndarray:
    """Decode one yuv422p10le frame via tp_measure, convert to 8-bit RGB."""
    import tp_measure
    import tp_synthesize

    Y, U, V, _meta = tp_measure.extract_frame(path, frame_index=frame_index)
    # Reuse the synthesizer's BT.601 limited-range YUV→BGR converter so the
    # displayed image matches what the comparison HTML swatches show.
    bgr = tp_synthesize._yuv422p10_to_bgr8(Y, U, V)
    return bgr[..., ::-1]  # matplotlib expects RGB


def _load_frame_from_image(path: str) -> np.ndarray:
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"could not read image: {path}")
    return bgr[..., ::-1]


class _Calibrator:
    def __init__(self, img: np.ndarray, title: str):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        self.img = img
        self.h, self.w = img.shape[:2]
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(img, interpolation="nearest")
        self.ax.set_xlim(-0.5, self.w - 0.5)
        self.ax.set_ylim(self.h - 0.5, -0.5)  # image coords (y down)
        self.ax.set_aspect("equal")
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass  # Some backends don't expose window_title

        self._initial_xlim = self.ax.get_xlim()
        self._initial_ylim = self.ax.get_ylim()

        # Pixel highlight: 1x1 outline that follows the pointer.
        self._marker = Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="red", linewidth=1.5,
        )
        self.ax.add_patch(self._marker)
        self._marker.set_visible(False)

        self._title_artist = self.ax.set_title(
            "hover to inspect — mouse wheel: zoom — left click: print — r: reset — q: quit"
        )
        self._click_count = 0

        c = self.fig.canvas
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("button_press_event", self._on_click)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("key_press_event", self._on_key)

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    # --- event handlers --------------------------------------------------

    def _pixel_under_event(self, event) -> Tuple[int, int]:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        return x, y

    def _on_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            if self._marker.get_visible():
                self._marker.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        x, y = self._pixel_under_event(event)
        if not (0 <= x < self.w and 0 <= y < self.h):
            if self._marker.get_visible():
                self._marker.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        px = self.img[y, x]
        r, g, b = int(px[0]), int(px[1]), int(px[2])
        self._marker.set_xy((x - 0.5, y - 0.5))
        self._marker.set_visible(True)
        self._title_artist.set_text(
            f"(x={x}, y={y})   RGB=({r},{g},{b})   clicks={self._click_count}"
        )
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if (event.inaxes is not self.ax
                or event.button != 1
                or event.xdata is None
                or event.ydata is None):
            return
        x, y = self._pixel_under_event(event)
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        self._click_count += 1
        px = self.img[y, x]
        r, g, b = int(px[0]), int(px[1]), int(px[2])
        print(f"click {self._click_count:>3}: ({x}, {y})  RGB=({r},{g},{b})")

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata
        # event.button is 'up' or 'down' depending on scroll direction
        factor = 1.0 / _ZOOM_FACTOR if event.button == "up" else _ZOOM_FACTOR
        new_xlim = (x + (xlim[0] - x) * factor, x + (xlim[1] - x) * factor)
        new_ylim = (y + (ylim[0] - y) * factor, y + (ylim[1] - y) * factor)
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        import matplotlib.pyplot as plt
        if event.key == "r":
            self.ax.set_xlim(self._initial_xlim)
            self.ax.set_ylim(self._initial_ylim)
            self.fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(self.fig)


def _main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Interaction:\n"
               "  mouse wheel : zoom in/out around the pointer\n"
               "  left click  : print current pixel (x, y) and RGB to stdout\n"
               "  r           : reset view to fit the frame\n"
               "  q           : quit",
    )
    p.add_argument("capture", nargs="?",
                   help="path to video file (mov / avi / mkv / ...)")
    p.add_argument("--image",
                   help="path to a PNG to inspect instead of a video frame")
    p.add_argument("--frame", type=int, default=60,
                   help="frame index when loading from video (default 60)")
    args = p.parse_args()

    if not args.capture and not args.image:
        p.error("provide either a capture (positional) or --image")
    if args.capture and args.image:
        p.error("provide only one of capture or --image, not both")

    if args.image:
        img = _load_frame_from_image(args.image)
        title = f"tp_calibrate — {args.image}"
    else:
        img = _load_frame_from_video(args.capture, args.frame)
        title = f"tp_calibrate — {args.capture}  frame={args.frame}"

    cal = _Calibrator(img, title=title)
    print(f"loaded {img.shape[1]}x{img.shape[0]} image.")
    print("mouse wheel = zoom, left click = print, 'r' = reset view, 'q' = quit")
    cal.show()


if __name__ == "__main__":
    _main()
