#!/usr/bin/env python3
"""Synthesize the ideal SW2 NTSC test pattern frame at 720x486.

Stage 1 scope: grey background, black main grid, top-left tartan (4x2),
4-step gray strip below it.
Stage 2 adds: all 4 boundary triangles, the registration cross, and the
black circle.

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


def _fill_triangle_y(Y: np.ndarray, p1, p2, p3, y10: int = None) -> None:
    """Fill a triangle on the Y plane only (chroma not affected — boundary
    triangles are black on the chroma-center-grey background).

    Uses cv2.fillPoly which handles the geometry. p1/p2/p3 are (x, y)
    integer coords."""
    import cv2
    if y10 is None:
        y10 = tp_chart.BLACK_Y10
    pts = np.array([[p1, p2, p3]], dtype=np.int32)
    cv2.fillPoly(Y, pts, int(y10))


def _draw_boundary_triangles(Y: np.ndarray) -> None:
    """Render all 4 boundary triangles from tp_chart.BOUNDARY_TRIANGLES."""
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        bc1 = tuple(int(v) for v in tri["ideal_back_corner_1"])
        bc2 = tuple(int(v) for v in tri["ideal_back_corner_2"])
        apex = tuple(int(v) for v in tri["ideal_apex"])
        # cv2.fillPoly handles out-of-frame apex coords (clips them).
        _fill_triangle_y(Y, bc1, bc2, apex)


def _draw_black_circle(Y: np.ndarray) -> None:
    """Render the black circle ring from tp_chart.BLACK_CIRCLE."""
    import cv2
    bc = tp_chart.BLACK_CIRCLE
    cv2.circle(
        Y,
        center=(bc["ideal_cx"], bc["ideal_cy"]),
        radius=bc["expected_radius_px"],
        color=int(tp_chart.BLACK_Y10),
        thickness=bc["ring_thickness_px"],
        lineType=cv2.LINE_AA,
    )


def _draw_registration_cross(Y: np.ndarray) -> None:
    """Render the registration cross from tp_chart.REGISTRATION_CROSS.

    A black square (box_size_px wide) with a white plus inside it
    (ideal_arm_len_px tip-to-tip, ideal_arm_thickness_px wide).
    """
    rc = tp_chart.REGISTRATION_CROSS
    cx, cy = rc["ideal_x"], rc["ideal_y"]
    box = rc["box_size_px"]
    arm_len = rc["ideal_arm_len_px"]
    arm_th  = rc["ideal_arm_thickness_px"]
    half_box = box // 2
    half_len = arm_len // 2
    half_th  = arm_th // 2
    # Black box
    Y[cy - half_box:cy + half_box, cx - half_box:cx + half_box] = tp_chart.BLACK_Y10
    # White cross arms
    Y[cy - half_th:cy + half_th + 1, cx - half_len:cx + half_len + 1] = tp_chart.WHITE_Y10
    Y[cy - half_len:cy + half_len + 1, cx - half_th:cx + half_th + 1] = tp_chart.WHITE_Y10


def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ideal SW2 frame as (Y, U, V) uint16 planes (yuv422p10le)."""
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    _draw_tartan(Y, U, V)
    _draw_gray_strip(Y, U, V)
    _draw_boundary_triangles(Y)
    _draw_black_circle(Y)
    _draw_registration_cross(Y)
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
