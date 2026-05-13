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


_SCALE_TRIANGLE  = 6
_SCALE_CROSS     = 6
_SCALE_INTERSECT = 8   # circle-intersection tiles want more detail
_CROP_HALF_TRI   = 24
_CROP_HALF_CR    = 24
_CROP_HALF_INT   = 14  # crop is 28x28 raw -> 224 scaled at 8x
_TILE_LABEL_H    = 28
_TILE_GAP        = 8
# NTSC pixel aspect ratio: pixels are taller than wide when the 720x486
# raster is displayed at 4:3 (PAR = 10/11). The chart's logically-round
# ring therefore appears elliptical in raster coords, widened horizontally
# by 11/10. Used for the fallback ring geometry when the detector hasn't
# returned a fitted ellipse.
_NTSC_PAR_X_OVER_Y = tp_chart.NTSC_PAR_X_OVER_Y
# Short arc segment drawn through each intersection (pixels of arc length
# either side of the predicted intersection point).
_ARC_HALF_LEN_PX = 18


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


def _circle_intersection_tile(bgr, label, px, py,
                              ring_cx, ring_cy, ring_rx, ring_ry):
    """Crop around the predicted intersection of the chart ring with a grid
    line, overlay a short arc segment of the predicted ellipse so the
    operator can see whether the ring actually passes through this point."""
    import math
    import cv2
    tile, (x0, y0) = _crop(bgr, px, py, _CROP_HALF_INT)
    if tile.size == 0:
        return None
    tile = _scale_up(tile, _SCALE_INTERSECT)
    s = _SCALE_INTERSECT
    # Predicted intersection: small open marker in cyan-ish.
    _draw_marker(tile, (px - x0) * s, (py - y0) * s,
                 (255, 200, 0), radius=6, thickness=2)
    # Short arc segment of the predicted ellipse through this intersection.
    if ring_rx > 0 and ring_ry > 0:
        angle_center = math.atan2(py - ring_cy, px - ring_cx)
        r_avg = (ring_rx + ring_ry) / 2.0
        dtheta = _ARC_HALF_LEN_PX / r_avg
        n_arc = 41
        for i in range(n_arc):
            t = angle_center - dtheta + (2.0 * dtheta) * i / (n_arc - 1)
            ax = ring_cx + ring_rx * math.cos(t)
            ay = ring_cy + ring_ry * math.sin(t)
            lx = (ax - x0) * s
            ly = (ay - y0) * s
            if 0 <= lx < tile.shape[1] and 0 <= ly < tile.shape[0]:
                cv2.circle(tile, (int(round(lx)), int(round(ly))),
                           2, (255, 200, 0), -1, cv2.LINE_AA)
    lines = [f"intersection=({px:.1f},{py:.1f})"]
    return _label_tile(tile, lines, label)


def _resolve_ring_geometry(data):
    """Return (cx, cy, rx, ry) in capture coords for the chart ring.

    Uses the detected ellipse if present; otherwise projects the chart-spec
    circle through the Stage 2 affine and applies NTSC 11:10 PAR scaling to
    the x semi-axis (the captured ring is wider than tall on real rasters)."""
    bc_cfg = tp_chart.BLACK_CIRCLE
    geom = data.get("geometry") or {}
    fids = geom.get("fiducials") or {}
    bc_fid = fids.get("circle")
    if bc_fid is not None:
        cx, cy = bc_fid["center"]
        r_min = float(bc_fid.get("rx", bc_cfg["expected_radius_px"]))
        r_max = float(bc_fid.get("ry", bc_cfg["expected_radius_px"]))
        # The detector orders axes as min/max from cv2.fitEllipse, not by
        # x/y. Under NTSC PAR (and assuming the detector's rotation_deg is
        # close to 0 / 180 i.e. axes are roughly aligned with the raster),
        # the horizontal axis is the larger one. Map accordingly.
        return float(cx), float(cy), r_max, r_min
    affine = (data.get("_meta") or {}).get("registration", {}).get("affine")
    ideal_cx, ideal_cy = bc_cfg["ideal_cx"], bc_cfg["ideal_cy"]
    if affine:
        M = affine
        cx = M[0][0] * ideal_cx + M[0][1] * ideal_cy + M[0][2]
        cy = M[1][0] * ideal_cx + M[1][1] * ideal_cy + M[1][2]
    else:
        cx, cy = ideal_cx, ideal_cy
    r_y = float(bc_cfg["expected_radius_px"])
    r_x = r_y * _NTSC_PAR_X_OVER_Y
    return float(cx), float(cy), r_x, r_y


def _circle_intersections(cx, cy, rx, ry):
    """Compute the 8 vertical-grid-line and 10 horizontal-grid-line
    intersections with the chart ring at the cells documented above.

    Returns a list of (label, x, y) in capture coords. Skips any
    intersection that would lie outside the ellipse (|dx|>=rx or |dy|>=ry).
    """
    import math
    pts = []
    # 8 vertical-line intersections: (x_grid, sign_for_y, label).
    v_specs = [
        (240, -1, "x=240 / row 1"),
        (480, -1, "x=480 / row 1"),
        (120, -1, "x=120 / row 3"),
        (600, -1, "x=600 / row 3"),
        (120, +1, "x=120 / row 7"),
        (600, +1, "x=600 / row 7"),
        (240, +1, "x=240 / row 9"),
        (480, +1, "x=480 / row 9"),
    ]
    for x_grid, sgn, label in v_specs:
        ratio = (x_grid - cx) / rx
        if abs(ratio) >= 1.0:
            continue
        dy = ry * math.sqrt(max(0.0, 1.0 - ratio * ratio))
        pts.append((label, float(x_grid), cy + sgn * dy))
    # 10 horizontal-line intersections.
    h_specs = [
        (108, -1, "y=108 / col 3"),
        (108, +1, "y=108 / col 10"),
        (162, -1, "y=162 / col 2"),
        (216, -1, "y=216 / col 2"),
        (270, -1, "y=270 / col 2"),
        (324, -1, "y=324 / col 2"),
        (162, +1, "y=162 / col 11"),
        (216, +1, "y=216 / col 11"),
        (270, +1, "y=270 / col 11"),
        (324, +1, "y=324 / col 11"),
    ]
    for y_grid, sgn, label in h_specs:
        ratio = (y_grid - cy) / ry
        if abs(ratio) >= 1.0:
            continue
        dx = rx * math.sqrt(max(0.0, 1.0 - ratio * ratio))
        pts.append((label, cx + sgn * dx, float(y_grid)))
    return pts


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

    # Black-circle ring intersections with chart grid lines. These give
    # 18 unambiguous sample points distributed around the ring at non-
    # tangent angles, replacing the older cardinal N/S/E/W approach
    # (which placed markers at the tangent extrema, where the ring's
    # slope is locally flat and the top/bottom extrema merge with the
    # picture frame).
    cx, cy, ring_rx, ring_ry = _resolve_ring_geometry(data)
    top_grid = _compose_grid(tiles, ncols=5)

    intersection_tiles = []
    for label, px, py in _circle_intersections(cx, cy, ring_rx, ring_ry):
        t = _circle_intersection_tile(bgr, label, px, py,
                                      cx, cy, ring_rx, ring_ry)
        if t is not None:
            intersection_tiles.append(t)
    bottom_grid = _compose_grid(intersection_tiles, ncols=6,
                                 bg=(20, 20, 25))

    if intersection_tiles:
        # Stack the two composites vertically with a small separator band.
        sep_h = _TILE_GAP * 2
        h1, w1, _ = top_grid.shape
        h2, w2, _ = bottom_grid.shape
        max_w = max(w1, w2)
        out = np.full((h1 + sep_h + h2, max_w, 3), 20, dtype=np.uint8)
        out[:h1, :w1] = top_grid
        out[h1 + sep_h:h1 + sep_h + h2, :w2] = bottom_grid
        return out
    return top_grid


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
