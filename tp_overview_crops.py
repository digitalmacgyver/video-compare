"""tp_overview_crops: extract the row-2 burst + radial-wedge cells from
a registered capture and produce small PNG sidecars used by the
visual-overview table at the top of the report.

Outputs (per capture, alongside the _stage2.json):
  <stem>_overview_burst_3p58.png
  <stem>_overview_burst_4p43.png
  <stem>_overview_burst_300TVL.png
  <stem>_overview_burst_400TVL.png
  <stem>_overview_radial.png

All crops are 4× nearest-neighbor upscaled BGR8, no annotation. The
crops use the chart-ideal bounding boxes projected through the
per-capture affine so frames register correctly even when the capture
has slight geometric drift.
"""
from __future__ import annotations
import json
from typing import Dict, Tuple

import numpy as np

import tp_chart
import tp_measure
import tp_synthesize


_UPSCALE = 4

# Region ids from tp_chart.BURST_REGIONS to crop, plus the radial wedge
# (special-cased because it uses RADIAL_WEDGE.cell_box, not a
# BURST_REGIONS entry).
_BURST_IDS = ["BURST_3p58", "BURST_4p43", "BURST_300TVL_DIAG",
              "BURST_400TVL_DIAG"]

# Output filename suffixes — keep in sync with tp_compare consumers.
_SUFFIX = {
    "BURST_3p58":        "_overview_burst_3p58.png",
    "BURST_4p43":        "_overview_burst_4p43.png",
    "BURST_300TVL_DIAG": "_overview_burst_300TVL.png",
    "BURST_400TVL_DIAG": "_overview_burst_400TVL.png",
    "RADIAL_WEDGE":      "_overview_radial.png",
}


def output_suffixes() -> Dict[str, str]:
    """Mapping of region id → sidecar filename suffix (used by tp_report
    to check whether sidecars already exist before regenerating)."""
    return dict(_SUFFIX)


def _apply_affine_pt(M, x: float, y: float) -> Tuple[float, float]:
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _project_box_to_capture(M, x, y, w, h, frame_shape):
    corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    pts = [_apply_affine_pt(M, cx, cy) for cx, cy in corners]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    H, W = frame_shape[:2]
    x0 = max(0, int(round(min(xs))))
    y0 = max(0, int(round(min(ys))))
    x1 = min(W, int(round(max(xs))))
    y1 = min(H, int(round(max(ys))))
    return x0, y0, x1, y1


def _box_for_burst(region_id: str):
    for r in tp_chart.BURST_REGIONS:
        if r["id"] == region_id:
            return r["ideal_box"]
    raise KeyError(f"burst region {region_id} not found")


def build(capture_path: str, json_path: str,
          frame_index: int = 60) -> Dict[str, np.ndarray]:
    """Return a dict {region_id: bgr8_image} for the 5 overview crops."""
    import cv2
    with open(json_path) as f:
        data = json.load(f)
    M = np.asarray(data["_meta"]["registration"]["affine"],
                   dtype=np.float32)

    Y_raw, U_raw, V_raw, _ = tp_measure.extract_frame(capture_path,
                                                     frame_index)
    Y, U, V, _padding = tp_measure.pad_to_486(Y_raw, U_raw, V_raw)
    bgr = tp_synthesize._yuv422p10_to_bgr8(Y, U, V)

    out: Dict[str, np.ndarray] = {}
    for rid in _BURST_IDS:
        x, y, w, h = _box_for_burst(rid)
        x0, y0, x1, y1 = _project_box_to_capture(M, x, y, w, h, bgr.shape)
        if x1 <= x0 or y1 <= y0:
            out[rid] = np.zeros((4, 4, 3), dtype=np.uint8); continue
        crop = bgr[y0:y1, x0:x1]
        out[rid] = cv2.resize(crop,
                              (crop.shape[1] * _UPSCALE,
                               crop.shape[0] * _UPSCALE),
                              interpolation=cv2.INTER_NEAREST)

    rw = tp_chart.RADIAL_WEDGE
    cell_x, cell_y, cell_w, cell_h = rw["cell_box"]
    x0, y0, x1, y1 = _project_box_to_capture(M, cell_x, cell_y,
                                             cell_w, cell_h, bgr.shape)
    if x1 > x0 and y1 > y0:
        crop = bgr[y0:y1, x0:x1]
        out["RADIAL_WEDGE"] = cv2.resize(
            crop, (crop.shape[1] * _UPSCALE, crop.shape[0] * _UPSCALE),
            interpolation=cv2.INTER_NEAREST)
    else:
        out["RADIAL_WEDGE"] = np.zeros((4, 4, 3), dtype=np.uint8)
    return out


def _main():
    import argparse
    import os
    import cv2
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture file (mov/avi/...)")
    p.add_argument("json",    help="per-capture _stage2.json from tp_measure")
    p.add_argument("--outdir", required=True,
                   help="directory to write sidecar PNGs (uses json stem)")
    p.add_argument("--frame", type=int, default=60)
    args = p.parse_args()
    stem = os.path.splitext(os.path.basename(args.json))[0]
    crops = build(args.capture, args.json, args.frame)
    for rid, img in crops.items():
        out = os.path.join(args.outdir, stem + _SUFFIX[rid])
        cv2.imwrite(out, img)
        print(f"wrote {out} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    _main()
