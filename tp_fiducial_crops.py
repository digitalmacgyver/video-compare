#!/usr/bin/env python3
"""tp_fiducial_crops: build a composite PNG of zoomed-in Stage 2 fiducial
crops so a human operator can visually verify the geometry detection.

Crops shown per capture:
    - 4 boundary triangles (TL, TR, BL, BR) with back-corner and apex markers.
    - The registration cross (RC).
    - 4 black-circle ring sample points (north, east, south, west of center).

Each tile is scaled up so individual pixels and the detected markers are
easy to see, and labeled with the detected coords.
"""
from __future__ import annotations
import json

import numpy as np

import tp_chart
import tp_measure


_SCALE_TRIANGLE = 6
_SCALE_CROSS    = 6
_SCALE_CIRCLE   = 6
_CROP_HALF_TRI  = 24   # triangle crop is (2*half) x (2*half)
_CROP_HALF_CR   = 24
_CROP_HALF_BC   = 18
_TILE_LABEL_H   = 28   # px tall label band above each tile
_TILE_GAP       = 8


def _yuv_to_bgr(Y, U, V):
    """Convert yuv422p10le planes to 8-bit BGR (BT.601 limited range)."""
    import tp_synthesize
    return tp_synthesize._yuv422p10_to_bgr8(Y, U, V)


def _crop(bgr, cx, cy, half):
    h, w = bgr.shape[:2]
    x0 = max(0, int(round(cx)) - half)
    y0 = max(0, int(round(cy)) - half)
    x1 = min(w, int(round(cx)) + half)
    y1 = min(h, int(round(cy)) + half)
    return bgr[y0:y1, x0:x1].copy(), (x0, y0)


def _scale_up(tile, factor):
    import cv2
    h, w = tile.shape[:2]
    return cv2.resize(tile, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def _draw_marker(tile, cx, cy, color, radius=4, thickness=2):
    import cv2
    cv2.circle(tile, (int(round(cx)), int(round(cy))), radius, color, thickness, cv2.LINE_AA)


def _draw_cross(tile, cx, cy, color, arm=5, thickness=2):
    import cv2
    cx, cy = int(round(cx)), int(round(cy))
    cv2.line(tile, (cx - arm, cy), (cx + arm, cy), color, thickness, cv2.LINE_AA)
    cv2.line(tile, (cx, cy - arm), (cx, cy + arm), color, thickness, cv2.LINE_AA)


def _label_tile(tile, lines, title):
    """Prepend a dark label band to a tile with title + numeric lines."""
    import cv2
    h, w = tile.shape[:2]
    band = np.full((_TILE_LABEL_H, w, 3), 30, dtype=np.uint8)
    cv2.putText(band, title, (4, 14), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (220, 220, 220), 1, cv2.LINE_AA)
    text = " ".join(lines)
    cv2.putText(band, text, (4, 24), cv2.FONT_HERSHEY_PLAIN, 0.8,
                (160, 200, 220), 1, cv2.LINE_AA)
    return np.vstack([band, tile])


def _triangle_tile(bgr, tri_id, tri_data, label_color=(255, 255, 255)):
    """Crop, scale, and annotate one triangle. tri_data has back_corner_1/2
    and apex_inferred (capture coords). Returns the labeled tile."""
    bc1 = tri_data.get("back_corner_1")
    bc2 = tri_data.get("back_corner_2")
    apex_inf = tri_data.get("apex_inferred")
    apex_det = tri_data.get("apex_detected")
    if bc1 is None or bc2 is None or apex_inf is None:
        return None
    cx = (bc1[0] + bc2[0] + apex_inf[0]) / 3.0
    cy = (bc1[1] + bc2[1] + apex_inf[1]) / 3.0
    tile, (x0, y0) = _crop(bgr, cx, cy, _CROP_HALF_TRI)
    if tile.size == 0:
        return None
    tile = _scale_up(tile, _SCALE_TRIANGLE)
    s = _SCALE_TRIANGLE
    _draw_marker(tile, (bc1[0] - x0) * s, (bc1[1] - y0) * s, (0, 255, 0))  # bc1 green
    _draw_marker(tile, (bc2[0] - x0) * s, (bc2[1] - y0) * s, (0, 255, 0))  # bc2 green
    _draw_cross(tile, (apex_inf[0] - x0) * s, (apex_inf[1] - y0) * s,
                (0, 165, 255), arm=8, thickness=2)  # apex inferred orange
    if apex_det is not None:
        _draw_cross(tile, (apex_det[0] - x0) * s, (apex_det[1] - y0) * s,
                    (0, 255, 255), arm=6, thickness=2)  # apex detected yellow
    lines = [
        f"bc1=({bc1[0]:.1f},{bc1[1]:.1f})",
        f"bc2=({bc2[0]:.1f},{bc2[1]:.1f})",
        f"apex=({apex_inf[0]:.1f},{apex_inf[1]:.1f})",
    ]
    return _label_tile(tile, lines, f"{tri_id} triangle")


def _cross_tile(bgr, cross_data):
    if cross_data is None:
        return None
    cx = cross_data["center"][0]
    cy = cross_data["center"][1]
    tile, (x0, y0) = _crop(bgr, cx, cy, _CROP_HALF_CR)
    if tile.size == 0:
        return None
    tile = _scale_up(tile, _SCALE_CROSS)
    s = _SCALE_CROSS
    _draw_cross(tile, (cx - x0) * s, (cy - y0) * s, (0, 255, 0),
                arm=10, thickness=2)
    lines = [
        f"center=({cx:.2f},{cy:.2f})",
        f"h_arm={cross_data.get('h_arm_len_px', 0):.0f}",
        f"v_arm={cross_data.get('v_arm_len_px', 0):.0f}",
    ]
    return _label_tile(tile, lines, "RC registration cross")


def _circle_sample_tile(bgr, label, cx, cy, expected_r):
    """Crop around one of N/S/E/W points on the expected circle ring."""
    tile, (x0, y0) = _crop(bgr, cx, cy, _CROP_HALF_BC)
    if tile.size == 0:
        return None
    tile = _scale_up(tile, _SCALE_CIRCLE)
    s = _SCALE_CIRCLE
    _draw_marker(tile, (cx - x0) * s, (cy - y0) * s,
                 (255, 200, 0), radius=10, thickness=2)
    lines = [f"({cx:.0f},{cy:.0f}) r_exp={expected_r:.0f}"]
    return _label_tile(tile, lines, f"BC {label}")


def _compose_grid(tiles, ncols=4, bg=(20, 20, 25)):
    """Lay out tiles in a fixed-column grid, padding with bg."""
    tiles = [t for t in tiles if t is not None]
    if not tiles:
        return np.full((40, 200, 3), bg, dtype=np.uint8)
    max_h = max(t.shape[0] for t in tiles)
    max_w = max(t.shape[1] for t in tiles)
    nrows = (len(tiles) + ncols - 1) // ncols
    out_h = nrows * max_h + (nrows + 1) * _TILE_GAP
    out_w = ncols * max_w + (ncols + 1) * _TILE_GAP
    out = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r = i // ncols
        c = i % ncols
        x = _TILE_GAP + c * (max_w + _TILE_GAP)
        y = _TILE_GAP + r * (max_h + _TILE_GAP)
        h, w = t.shape[:2]
        out[y:y + h, x:x + w] = t
    return out


def build(capture_path: str, json_path: str, frame_index: int = 60) -> np.ndarray:
    """Return a BGR composite image of all fiducial crops, scaled and labeled.
    Returns a small placeholder if the geometry block is missing.
    """
    with open(json_path) as f:
        data = json.load(f)
    Y, U, V, _ = tp_measure.extract_frame(capture_path, frame_index)
    Y_p, U_p, V_p, _ = tp_measure.pad_to_486(Y, U, V)
    bgr = _yuv_to_bgr(Y_p, U_p, V_p)

    geom = data.get("geometry") or {}
    fids = geom.get("fiducials") or {}

    tiles = []
    # Triangles
    tris = fids.get("triangles") or {}
    for tid in ("TL", "TR", "BL", "BR"):
        td = tris.get(tid)
        if td is None:
            continue
        tile = _triangle_tile(bgr, tid, td)
        if tile is not None:
            tiles.append(tile)

    # Cross
    cross = fids.get("cross")
    cross_tile = _cross_tile(bgr, cross)
    if cross_tile is not None:
        tiles.append(cross_tile)

    # Black-circle ring sample points (N/E/S/W on the expected ring).
    bc_cfg = tp_chart.BLACK_CIRCLE
    bc_fid = fids.get("circle")
    if bc_fid is not None:
        cx = bc_fid["center"][0]
        cy = bc_fid["center"][1]
        rx = bc_fid.get("rx", bc_cfg["expected_radius_px"])
        ry = bc_fid.get("ry", bc_cfg["expected_radius_px"])
    else:
        cx, cy = bc_cfg["ideal_cx"], bc_cfg["ideal_cy"]
        rx = ry = bc_cfg["expected_radius_px"]
    samples = [
        ("N", cx,      cy - ry, ry),
        ("E", cx + rx, cy,      rx),
        ("S", cx,      cy + ry, ry),
        ("W", cx - rx, cy,      rx),
    ]
    for label, sx, sy, r_exp in samples:
        tile = _circle_sample_tile(bgr, label, sx, sy, r_exp)
        if tile is not None:
            tiles.append(tile)

    return _compose_grid(tiles, ncols=4)


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture video path")
    p.add_argument("json", help="tp_measure JSON output")
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--output", required=True, help="output PNG")
    args = p.parse_args()
    bgr = build(args.capture, args.json, args.frame)
    import cv2
    cv2.imwrite(args.output, bgr)
    print(f"wrote {args.output} ({bgr.shape[1]}x{bgr.shape[0]})")


if __name__ == "__main__":
    _main()
