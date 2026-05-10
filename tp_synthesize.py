#!/usr/bin/env python3
"""Synthesize the ideal SW2 NTSC test pattern frame at 720x486.

Stage 1 scope: grey background, black main grid, top-left tartan (4x2),
4-step gray strip below it, and a placeholder boundary-triangle cell.

Library entry point:
    Y, U, V = tp_synthesize.synthesize(720, 486)
        # uint16 yuv422p10le planes, BT.601 limited range

CLI (added in a later task):
    python tp_synthesize.py --raster 720x486 --output ideal.png
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

import tp_chart

# Main grid: 12 columns x 9 rows on 720x486 -> 60 x 54 px cells.
_GRID_COL_PX = 60
_GRID_ROW_PX = 54
# Black grid line width: 168 ns at 13.5 MHz ~= 2.27 samples. Use 3 px so
# the line is centred on integer-coord landmarks (an even width would
# straddle a sub-pixel boundary and bake a 0.5 px offset into detection).
_GRID_LINE_W = 3


def _make_grey_planes(width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = np.full((height, width), tp_chart.GREY_BACKGROUND_Y10, dtype=np.uint16)
    U = np.full((height, width // 2), tp_chart.CHROMA_CENTER, dtype=np.uint16)
    V = np.full((height, width // 2), tp_chart.CHROMA_CENTER, dtype=np.uint16)
    return Y, U, V


def _draw_grid(Y: np.ndarray) -> None:
    """Overlay the black main grid on the Y plane in place.

    With _GRID_LINE_W=3 (odd) and half=1, each line spans [center-1, center+2),
    i.e. 3 columns/rows centred exactly on the integer landmark coordinate.
    """
    h, w = Y.shape
    half = _GRID_LINE_W // 2  # 1 for width 3
    for x in range(0, w + 1, _GRID_COL_PX):
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        if x1 > x0:
            Y[:, x0:x1] = tp_chart.BLACK_Y10
    for y in range(0, h + 1, _GRID_ROW_PX):
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        if y1 > y0:
            Y[y0:y1, :] = tp_chart.BLACK_Y10


def _fill_box_yuv422(
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    box: Tuple[int, int, int, int],
    y10: float,
    u10: float,
    v10: float,
) -> None:
    """Fill a rectangular box in all three planes with the given YUV codes."""
    x, y, w, h = box
    assert x % 2 == 0 and w % 2 == 0, (
        f"_fill_box_yuv422 requires even x and w for exact 4:2:2 alignment; "
        f"got x={x}, w={w}"
    )
    Y[y:y + h, x:x + w] = int(round(y10))
    cx0, cx1 = x // 2, (x + w) // 2
    U[y:y + h, cx0:cx1] = int(round(u10))
    V[y:y + h, cx0:cx1] = int(round(v10))


def _draw_tartan(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    """Render the 4x2 tartan region by filling each box with its ideal YUV10 codes."""
    for r in tp_chart.TARTAN_REGIONS:
        e = r["expected"]
        _fill_box_yuv422(Y, U, V, r["ideal_box"], e["y10"], e["u10"], e["v10"])


def _draw_gray_strip(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    """Render the 4-step gray strip by filling each box with its ideal Y10 code."""
    for r in tp_chart.GRAY_REGIONS:
        e = r["expected"]
        _fill_box_yuv422(Y, U, V, r["ideal_box"], e["y10"], e["u10"], e["v10"])


def _draw_boundary_triangle_upper_left(Y: np.ndarray) -> None:
    """Black filled triangle inside the (0..30, 81..108) cell, pointing right.

    For Stage 1 we render only the upper-left boundary-triangle cell; the
    other three corners are added in Stage 2 along with their detectors.
    The shape is a simple right-pointing triangle, vertices roughly:
       (3, 84)  top-left
       (3, 105) bottom-left
       (24, 94) right tip
    Drawn by filling row-by-row.
    """
    for r in range(84, 106):
        # Distance from the apex row 94 (range 0..11)
        d = abs(r - 94)
        x_right = max(4, 24 - int(round(21 * d / 11)))
        Y[r, 3:x_right] = tp_chart.BLACK_Y10


def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ideal SW2 frame as (Y, U, V) uint16 planes (yuv422p10le)."""
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    _draw_tartan(Y, U, V)
    _draw_gray_strip(Y, U, V)
    _draw_boundary_triangle_upper_left(Y)
    return Y, U, V


# =====================================================================
# CLI
# =====================================================================

def _yuv422p10_to_bgr8(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Convert yuv422p10le planes to 8-bit BGR for image writing."""
    h, w = Y.shape
    U_full = np.repeat(U, 2, axis=1)[:, :w]
    V_full = np.repeat(V, 2, axis=1)[:, :w]
    y = (Y.astype(np.float32) - tp_chart.BLACK_Y10) / tp_chart.Y_RANGE
    cb = (U_full.astype(np.float32) - tp_chart.CHROMA_CENTER) / 896.0
    cr = (V_full.astype(np.float32) - tp_chart.CHROMA_CENTER) / 896.0
    r = np.clip(y + 1.402 * cr, 0.0, 1.0)
    g = np.clip(y - 0.344136 * cb - 0.714136 * cr, 0.0, 1.0)
    b = np.clip(y + 1.772 * cb, 0.0, 1.0)
    bgr = np.stack([b, g, r], axis=-1) * 255.0
    return bgr.astype(np.uint8)


def _write_yuv422p10le(path: str, Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(Y.astype("<u2").tobytes())
        f.write(U.astype("<u2").tobytes())
        f.write(V.astype("<u2").tobytes())


def _parse_raster(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raster", default="720x486", help="WIDTHxHEIGHT")
    p.add_argument("--output", required=True, help="output file (.png or .yuv)")
    args = p.parse_args()
    w, h = _parse_raster(args.raster)
    Y, U, V = synthesize(w, h)
    if args.output.endswith(".png"):
        import cv2
        bgr = _yuv422p10_to_bgr8(Y, U, V)
        cv2.imwrite(args.output, bgr)
    elif args.output.endswith(".yuv"):
        _write_yuv422p10le(args.output, Y, U, V)
    else:
        raise SystemExit("output must end in .png or .yuv")
    print(f"wrote {args.output} ({w}x{h})")


if __name__ == "__main__":
    _main()
