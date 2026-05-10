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


def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ideal SW2 frame as (Y, U, V) uint16 planes (yuv422p10le)."""
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    # Tartan + gray + boundary triangle added in subsequent tasks.
    return Y, U, V
