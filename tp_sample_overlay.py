#!/usr/bin/env python3
"""tp_sample_overlay: produce a diagnostic PNG of a capture frame with
overlays showing where each tartan/gray sample was taken, where each
grid landmark and Stage 2 fiducial was detected, and where they were
expected.

Usage:
    python tp_sample_overlay.py capture.mov capture.json --output diag.png

Reads the affine from the capture JSON and re-projects each region's
ideal center into capture coords, drawing the sample window box and a
label. Also draws every detected landmark/triangle/cross/circle from
the geometry block.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

import tp_chart
import tp_measure
import tp_synthesize


def _apply_affine(M, x, y):
    a, b, tx = M[0]
    c, d, ty = M[1]
    return (a * x + b * y + tx, c * x + d * y + ty)


def _draw_box(bgr, x0, y0, x1, y1, color, thickness=1):
    import cv2
    cv2.rectangle(bgr, (int(round(x0)), int(round(y0))),
                  (int(round(x1)), int(round(y1))), color, thickness)


def _draw_label(bgr, x, y, text, color):
    import cv2
    cv2.putText(bgr, text, (int(round(x)), int(round(y))),
                cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1, cv2.LINE_AA)


def _draw_cross(bgr, x, y, color, size=4):
    import cv2
    x, y = int(round(x)), int(round(y))
    cv2.line(bgr, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
    cv2.line(bgr, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)


def annotate(capture_path: str, json_path: str, frame_index: int = 60) -> np.ndarray:
    """Return a BGR uint8 image with diagnostic overlays."""
    import cv2

    with open(json_path) as f:
        data = json.load(f)

    Y, U, V, _ = tp_measure.extract_frame(capture_path, frame_index=frame_index)
    Y_p, U_p, V_p, _ = tp_measure.pad_to_486(Y, U, V)
    bgr = tp_synthesize._yuv422p10_to_bgr8(Y_p, U_p, V_p).copy()

    reg = data["_meta"]["registration"]
    M = np.asarray(reg["affine"], dtype=np.float64) if reg["affine"] is not None else None

    # ---- Tartan sample windows (cyan) ----
    for r in tp_chart.TARTAN_REGIONS:
        x, y, w, h = r["ideal_box"]
        cx_ideal = x + w / 2.0
        cy_ideal = y + h / 2.0
        if M is None:
            cx_cap, cy_cap = cx_ideal, cy_ideal
        else:
            cx_cap, cy_cap = _apply_affine(M, cx_ideal, cy_ideal)
        size_frac = float(r["sample"]["size_frac"])
        hw = max(1, int(round(w * size_frac / 2.0)))
        hh = max(1, int(round(h * size_frac / 2.0)))
        _draw_box(bgr, cx_cap - hw, cy_cap - hh, cx_cap + hw, cy_cap + hh,
                  (255, 255, 0), 1)  # cyan in BGR
        _draw_label(bgr, cx_cap + hw + 2, cy_cap + 3, r["id"], (255, 255, 0))

    # ---- Gray sample windows (yellow) ----
    for r in tp_chart.GRAY_REGIONS:
        x, y, w, h = r["ideal_box"]
        cx_ideal = x + w / 2.0
        cy_ideal = y + h / 2.0
        if M is None:
            cx_cap, cy_cap = cx_ideal, cy_ideal
        else:
            cx_cap, cy_cap = _apply_affine(M, cx_ideal, cy_ideal)
        size_frac = float(r["sample"]["size_frac"])
        hw = max(1, int(round(w * size_frac / 2.0)))
        hh = max(1, int(round(h * size_frac / 2.0)))
        _draw_box(bgr, cx_cap - hw, cy_cap - hh, cx_cap + hw, cy_cap + hh,
                  (0, 255, 255), 1)  # yellow in BGR
        _draw_label(bgr, cx_cap + hw + 2, cy_cap + 3, r["id"], (0, 255, 255))

    # ---- Grid landmarks (magenta crosses): expected position and used flag ----
    used = set(reg.get("landmarks_used", []) or [])
    for lm in tp_chart.GRID_LANDMARKS:
        if M is None:
            cx, cy = lm["ideal_x"], lm["ideal_y"]
        else:
            cx, cy = _apply_affine(M, lm["ideal_x"], lm["ideal_y"])
        color = (255, 0, 255) if lm["id"] in used else (128, 0, 128)
        _draw_cross(bgr, cx, cy, color, size=3)
        _draw_label(bgr, cx + 4, cy - 4, lm["id"], color)

    # ---- Stage 2 fiducials (green/red) ----
    geo = data.get("geometry")
    if geo is not None:
        for tid, t in geo["fiducials"]["triangles"].items():
            if t is None:
                continue
            bc1 = t["back_corner_1"]; bc2 = t["back_corner_2"]
            ap_i = t["apex_inferred"]; ap_d = t["apex_detected"]
            color_back = (0, 255, 0)   # green
            color_apex = (0, 200, 0)
            _draw_cross(bgr, bc1[0], bc1[1], color_back, 4)
            _draw_cross(bgr, bc2[0], bc2[1], color_back, 4)
            _draw_label(bgr, bc1[0] + 3, bc1[1] + 10, f"{tid}.bc1", color_back)
            if ap_i is not None:
                _draw_cross(bgr, ap_i[0], ap_i[1], color_apex, 3)
                _draw_label(bgr, ap_i[0] + 3, ap_i[1] + 10, f"{tid}.ap_i", color_apex)
            if ap_d is not None:
                _draw_cross(bgr, ap_d[0], ap_d[1], (0, 100, 255), 3)  # orange

        rc = geo["fiducials"]["cross"]
        if rc is not None:
            cx, cy = rc["center"]
            _draw_cross(bgr, cx, cy, (0, 200, 0), 6)
            _draw_label(bgr, cx + 6, cy - 6, "RC", (0, 200, 0))

        bc = geo["fiducials"]["circle"]
        if bc is not None:
            cx, cy = bc["center"]
            rx, ry = bc["rx"], bc["ry"]
            cv2.ellipse(bgr, (int(round(cx)), int(round(cy))),
                        (int(round(rx)), int(round(ry))),
                        bc["rotation_deg"], 0, 360, (0, 200, 0), 1, cv2.LINE_AA)
            _draw_label(bgr, cx + 4, cy + 4, "BC", (0, 200, 0))

    # Header overlay
    cap_name = os.path.basename(capture_path)
    _draw_label(bgr, 4, 12, cap_name, (255, 255, 255))
    flag = reg.get("quality_flag", "?")
    inliers = reg.get("inliers", "?")
    total = reg.get("total", "?")
    res_mean = reg.get("residuals_px", {}).get("mean", float("nan"))
    _draw_label(bgr, 4, 26,
                f"reg {flag} inliers {inliers}/{total} resid {res_mean:.2f}px",
                (255, 255, 255))

    return bgr


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture video path")
    p.add_argument("json", help="tp_measure JSON output")
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--output", required=True, help="output PNG")
    args = p.parse_args()

    bgr = annotate(args.capture, args.json, args.frame)
    import cv2
    cv2.imwrite(args.output, bgr)
    print(f"wrote {args.output} ({bgr.shape[1]}x{bgr.shape[0]})")


if __name__ == "__main__":
    main()
