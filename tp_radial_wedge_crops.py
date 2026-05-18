"""tp_radial_wedge_crops: extract the radial-wedge cell (8,11) from a
registered capture and produce a labeled PNG sidecar for the report.

Output: `<stem>_radial_wedge.png` — a 8×-upscaled BGR crop of the
wedge box with a thin crosshair on the chart-spec center.
"""
from __future__ import annotations
import json
from typing import Tuple

import numpy as np

import tp_chart
import tp_measure
import tp_synthesize


_UPSCALE = 8


def _apply_affine_pt(M, x: float, y: float) -> Tuple[float, float]:
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def build(capture_path: str, json_path: str,
          frame_index: int = 60) -> np.ndarray:
    """Return the BGR8 image for `<stem>_radial_wedge.png`."""
    import cv2
    with open(json_path) as f:
        data = json.load(f)
    M = np.asarray(data["_meta"]["registration"]["affine"],
                   dtype=np.float32)

    Y_raw, U_raw, V_raw, _ = tp_measure.extract_frame(capture_path,
                                                      frame_index)
    Y, U, V, _padding = tp_measure.pad_to_486(Y_raw, U_raw, V_raw)
    bgr = tp_synthesize._yuv422p10_to_bgr8(Y, U, V)

    rw = tp_chart.RADIAL_WEDGE
    cell_x, cell_y, cell_w, cell_h = rw["cell_box"]
    cx_ideal, cy_ideal = rw["center_xy"]

    # Project the cell corners through the affine to find the capture-
    # coord bounding box.
    corners = [(cell_x, cell_y),
               (cell_x + cell_w, cell_y),
               (cell_x, cell_y + cell_h),
               (cell_x + cell_w, cell_y + cell_h)]
    projected = [_apply_affine_pt(M, x, y) for x, y in corners]
    xs = [p[0] for p in projected]; ys = [p[1] for p in projected]
    x0 = max(0, int(round(min(xs))))
    y0 = max(0, int(round(min(ys))))
    x1 = min(bgr.shape[1], int(round(max(xs))))
    y1 = min(bgr.shape[0], int(round(max(ys))))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    crop = bgr[y0:y1, x0:x1].copy()
    cx_cap, cy_cap = _apply_affine_pt(M, cx_ideal, cy_ideal)
    # Mark the wedge center with a 2-px crosshair in cyan for context.
    cx_local = int(round(cx_cap - x0))
    cy_local = int(round(cy_cap - y0))
    if 0 <= cx_local < crop.shape[1] and 0 <= cy_local < crop.shape[0]:
        cv2.line(crop, (cx_local - 2, cy_local), (cx_local + 2, cy_local),
                 (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(crop, (cx_local, cy_local - 2), (cx_local, cy_local + 2),
                 (255, 200, 0), 1, cv2.LINE_AA)
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * _UPSCALE, h * _UPSCALE),
                      interpolation=cv2.INTER_NEAREST)


def _main():
    import argparse
    import cv2
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture file (mov/avi/...)")
    p.add_argument("json",    help="per-capture _stage2.json from tp_measure")
    p.add_argument("--output", required=True)
    p.add_argument("--frame", type=int, default=60)
    args = p.parse_args()
    bgr = build(args.capture, args.json, args.frame)
    cv2.imwrite(args.output, bgr)
    print(f"wrote {args.output} ({bgr.shape[1]}x{bgr.shape[0]})")


if __name__ == "__main__":
    _main()
