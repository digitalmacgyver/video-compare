#!/usr/bin/env python3
"""tp_calibrate: interactive pixel-coordinate inspector for SW2 frames.

Loads one frame from a video file (via ffmpeg, weave-decoded like tp_measure)
or a PNG, and lets the operator zoom, pan, and click to record pixel
coordinates of features on the chart. Used to calibrate tp_chart.py region
coordinates against real captures.

CLI:
    python tp_calibrate.py /path/to/capture.mov [--frame 60]
    python tp_calibrate.py --image /path/to/frame.png

    # Label-prompt mode: walk through a list of named targets and save the
    # clicked pixel coordinate for each to a JSON file.
    python tp_calibrate.py capture.mov --preset stage1-regions --output calib.json
    python tp_calibrate.py capture.mov --targets YEL,CYN,G1 --output calib.json

Interaction:
    mouse wheel : zoom in / out, centred on the pointer
    left click  : record current pixel (label-mode) or print it (free-form)
    'r'         : reset view to fit the whole frame
    'n'         : skip the current target (label-mode only)
    'b'         : back up one target (label-mode only)
    'q'         : save labels (if any) and quit
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_ZOOM_FACTOR = 1.25

# Built-in preset lists for label-prompt mode. Operator picks one with
# --preset NAME; or supplies their own comma list via --targets.
_PRESETS: Dict[str, List[str]] = {
    "stage1-regions": [
        # 8 tartan + 4 gray-strip centres (Stage 1 measurement targets).
        "YEL", "CYN", "BLU", "RED",
        "MAG", "GRN", "RED2", "CYN2",
        "G1", "G2", "G3", "G4",
    ],
    "stage1-landmarks": [
        # 8 registration grid-intersection landmarks (Stage 1).
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    ],
    "stage1-all": [
        "YEL", "CYN", "BLU", "RED",
        "MAG", "GRN", "RED2", "CYN2",
        "G1", "G2", "G3", "G4",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    ],
    "stage2-fiducials": [
        # 4 triangles, each with back-corner-1, back-corner-2, apex.
        "TL_back_corner_1", "TL_back_corner_2", "TL_apex",
        "TR_back_corner_1", "TR_back_corner_2", "TR_apex",
        "BL_back_corner_1", "BL_back_corner_2", "BL_apex",
        "BR_back_corner_1", "BR_back_corner_2", "BR_apex",
        # Registration cross center.
        "RC_center",
        # Black circle ring sample points (12, 3, 6, 9 o'clock).
        "BC_north", "BC_east", "BC_south", "BC_west",
    ],
    "stage2-landmarks": [
        # Updated 12-anchor grid catalog (Stage 2).
        "L1", "L2", "L3", "L4", "L5", "L6",
        "L7", "L8", "L9", "L10", "L11", "L12",
    ],
}


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
    def __init__(
        self,
        img: np.ndarray,
        title: str,
        targets: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        source_info: Optional[Dict[str, Any]] = None,
    ):
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

        # Label-prompt state.
        self._targets: List[str] = list(targets) if targets else []
        self._current_idx: int = 0
        self._labels: Dict[str, List[int]] = {}
        self._output_path: Optional[str] = output_path
        self._source_info: Dict[str, Any] = source_info or {}

        # Persistent help text along the bottom of the figure.
        controls = self._build_controls_text()
        plt.subplots_adjust(bottom=0.10)
        self.fig.text(
            0.5, 0.02, controls,
            ha="center", va="bottom",
            fontsize=9, color="#666666",
            family="monospace",
        )

        # Dynamic title-line (current target / pixel info).
        self._click_count = 0
        self._title_artist = self.ax.set_title("", fontsize=10)
        self._refresh_title()

        c = self.fig.canvas
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("button_press_event", self._on_click)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("close_event", self._on_close)

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    # --- text helpers ----------------------------------------------------

    def _build_controls_text(self) -> str:
        base = (
            "mouse wheel: zoom around pointer    "
            "left click: {click_action}    "
            "r: reset view"
        )
        if self._targets:
            return (
                base.format(click_action="record current target")
                + "    n: skip target    b: back one target    q: save & quit"
            )
        return base.format(click_action="print pixel to stdout") + "    q: quit"

    def _current_target(self) -> Optional[str]:
        if not self._targets:
            return None
        if 0 <= self._current_idx < len(self._targets):
            return self._targets[self._current_idx]
        return None

    def _refresh_title(self, pixel_text: str = ""):
        if self._targets:
            target = self._current_target()
            if target is None:
                head = f"all {len(self._targets)} targets recorded — press q to save and quit"
            else:
                recorded = self._labels.get(target)
                rec_str = (f"  [already at {tuple(recorded)}]"
                           if recorded is not None else "")
                head = (
                    f"target {self._current_idx + 1}/{len(self._targets)}: "
                    f"'{target}'{rec_str}"
                )
        else:
            head = f"clicks: {self._click_count}"
        tail = ("    " + pixel_text) if pixel_text else ""
        self._title_artist.set_text(head + tail)

    # --- event handlers --------------------------------------------------

    def _pixel_under_event(self, event) -> Tuple[int, int]:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        return x, y

    def _on_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            if self._marker.get_visible():
                self._marker.set_visible(False)
                self._refresh_title()
                self.fig.canvas.draw_idle()
            return
        x, y = self._pixel_under_event(event)
        if not (0 <= x < self.w and 0 <= y < self.h):
            if self._marker.get_visible():
                self._marker.set_visible(False)
                self._refresh_title()
                self.fig.canvas.draw_idle()
            return
        px = self.img[y, x]
        r, g, b = int(px[0]), int(px[1]), int(px[2])
        self._marker.set_xy((x - 0.5, y - 0.5))
        self._marker.set_visible(True)
        self._refresh_title(f"(x={x}, y={y})  RGB=({r},{g},{b})")
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
        px = self.img[y, x]
        r, g, b = int(px[0]), int(px[1]), int(px[2])

        if self._targets:
            target = self._current_target()
            if target is None:
                print("all targets recorded; press 'q' to save and quit")
                return
            self._labels[target] = [x, y]
            print(f"recorded '{target}' = ({x}, {y})  RGB=({r},{g},{b})")
            self._save_partial()
            self._current_idx += 1
            self._refresh_title()
            self.fig.canvas.draw_idle()
        else:
            self._click_count += 1
            print(f"click {self._click_count:>3}: ({x}, {y})  RGB=({r},{g},{b})")
            self._refresh_title()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata
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
        elif event.key == "n" and self._targets:
            if self._current_idx < len(self._targets):
                skipped = self._targets[self._current_idx]
                print(f"skipped '{skipped}'")
                self._current_idx += 1
                self._refresh_title()
                self.fig.canvas.draw_idle()
        elif event.key == "b" and self._targets:
            if self._current_idx > 0:
                self._current_idx -= 1
                back_to = self._targets[self._current_idx]
                print(f"back to '{back_to}'")
                self._refresh_title()
                self.fig.canvas.draw_idle()
        elif event.key == "q":
            self._save_final()
            plt.close(self.fig)

    def _on_close(self, _event):
        # Save on window-close too (in case user clicks the X button rather
        # than pressing 'q').
        self._save_final()

    # --- persistence -----------------------------------------------------

    def _save_partial(self):
        # Save after every click so accidental window closures don't lose
        # work. Silent on errors here; _save_final reports problems.
        if not self._output_path:
            return
        try:
            self._write_json()
        except Exception:
            pass

    def _save_final(self):
        if not self._output_path:
            return
        try:
            self._write_json()
            print(
                f"wrote {self._output_path}: "
                f"{len(self._labels)}/{len(self._targets)} targets recorded"
            )
        except OSError as e:
            print(f"failed to write {self._output_path}: {e}", file=sys.stderr)

    def _write_json(self):
        data = {
            "source": self._source_info,
            "image_shape": [self.h, self.w],
            "targets_order": self._targets,
            "labels": self._labels,
        }
        with open(self._output_path, "w") as f:
            json.dump(data, f, indent=2)


def _parse_targets(args) -> List[str]:
    if args.preset and args.targets:
        raise SystemExit("use only one of --preset or --targets, not both")
    if args.preset:
        if args.preset not in _PRESETS:
            raise SystemExit(
                f"unknown preset '{args.preset}'. "
                f"available: {', '.join(sorted(_PRESETS))}"
            )
        return list(_PRESETS[args.preset])
    if args.targets:
        return [t.strip() for t in args.targets.split(",") if t.strip()]
    return []


def _main():
    presets_help = (
        "Built-in presets:\n  "
        + "\n  ".join(
            f"{name:20s} {', '.join(items)}" for name, items in _PRESETS.items()
        )
    )
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Interaction:\n"
            "  mouse wheel : zoom in/out around the pointer\n"
            "  left click  : record/print current pixel (x, y) and RGB\n"
            "  r           : reset view to fit the frame\n"
            "  n           : skip the current target (label-mode only)\n"
            "  b           : back up one target (label-mode only)\n"
            "  q           : save labels (if any) and quit\n\n"
            + presets_help
        ),
    )
    p.add_argument("capture", nargs="?",
                   help="path to video file (mov / avi / mkv / ...)")
    p.add_argument("--image",
                   help="path to a PNG to inspect instead of a video frame")
    p.add_argument("--frame", type=int, default=60,
                   help="frame index when loading from video (default 60)")
    p.add_argument("--preset", choices=sorted(_PRESETS),
                   help="use a built-in target list for label-prompt mode")
    p.add_argument("--targets",
                   help="comma-separated target list, e.g. 'YEL,CYN,G1' (overrides --preset)")
    p.add_argument("--output",
                   help="JSON path to save recorded labels (required for label-mode)")
    args = p.parse_args()

    if not args.capture and not args.image:
        p.error("provide either a capture (positional) or --image")
    if args.capture and args.image:
        p.error("provide only one of capture or --image, not both")

    targets = _parse_targets(args)
    if targets and not args.output:
        p.error("--output is required when --preset or --targets is set")

    if args.image:
        img = _load_frame_from_image(args.image)
        title = f"tp_calibrate — {args.image}"
        source_info: Dict[str, Any] = {"image": os.path.abspath(args.image)}
    else:
        img = _load_frame_from_video(args.capture, args.frame)
        title = f"tp_calibrate — {args.capture}  frame={args.frame}"
        source_info = {
            "capture": os.path.abspath(args.capture),
            "frame_index": args.frame,
        }

    cal = _Calibrator(
        img, title=title,
        targets=targets,
        output_path=args.output,
        source_info=source_info,
    )
    print(f"loaded {img.shape[1]}x{img.shape[0]} image.")
    if targets:
        print(
            f"label-prompt mode: {len(targets)} targets to record "
            f"(writing to {args.output})"
        )
        print("first target: '" + targets[0] + "'")
    else:
        print("free-form mode: left-click prints the pointed pixel.")
    cal.show()


if __name__ == "__main__":
    _main()
