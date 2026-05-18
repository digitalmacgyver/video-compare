"""tp_wedge_crops: extract the frequency-response wedge column from a
registered capture and produce a labeled PNG sidecar.

The resulting `<stem>_wedge.png` is embedded in the HTML report so the
reader can see at a glance how the decoder reproduces the chart's
continuous narrowing-stripe pattern.
"""
from __future__ import annotations
import json
from typing import Tuple

import numpy as np

import tp_chart
import tp_measure
import tp_synthesize


# Vertical upscaling for legibility. The wedge crop is 60 wide × 216
# tall in raster pixels; we upscale 4× so a viewer can resolve
# individual stripes at the high-frequency end.
_UPSCALE = 4


def _apply_affine_pt(M, x: float, y: float) -> Tuple[float, float]:
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _wedge_box_in_capture(M):
    """Map the ideal wedge column box (x, y, w, h) through the affine
    and return integer pixel bounds in capture coords."""
    w = tp_chart.WEDGE_COLUMN
    cx0, cy0 = _apply_affine_pt(M, w["x"], w["y_top"])
    cx1, cy1 = _apply_affine_pt(M, w["x"] + w["width"], w["y_bottom"])
    x0 = int(round(min(cx0, cx1)))
    y0 = int(round(min(cy0, cy1)))
    x1 = int(round(max(cx0, cx1)))
    y1 = int(round(max(cy0, cy1)))
    return x0, y0, x1, y1


def _label_strip(width_px: int) -> np.ndarray:
    """Build a narrow label strip with frequency labels at 0.5 MHz
    intervals, designed to sit to the right of the wedge crop."""
    import cv2
    w_col = tp_chart.WEDGE_COLUMN
    py_top = w_col["y_top"] * _UPSCALE
    py_bot = w_col["y_bottom"] * _UPSCALE
    h = py_bot - py_top
    label = np.full((h, width_px, 3), 24, dtype=np.uint8)
    # Tick marks + text at each 0.5 MHz step. Use a slightly bigger
    # font at integer-MHz marks to make 2/3/4/5 stand out.
    f_top = w_col["freq_top"]
    f_bot = w_col["freq_bottom"]
    for freq in np.arange(f_top, f_bot + 0.01, 0.5):
        t = (freq - f_top) / (f_bot - f_top)
        y_label = int(round(t * (h - 1)))
        is_major = abs(round(freq) - freq) < 0.01
        cv2.line(label, (0, y_label),
                 (8 if is_major else 4, y_label),
                 (220, 220, 220), 1, cv2.LINE_AA)
        text = f"{freq:.1f}".rstrip("0").rstrip(".") + " MHz"
        font_scale = 0.45 if is_major else 0.35
        color = (240, 240, 240) if is_major else (170, 170, 170)
        cv2.putText(label, text, (12, y_label + 4),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, color, 1,
                    cv2.LINE_AA)
    return label


def build(capture_path: str, json_path: str,
          frame_index: int = 60) -> np.ndarray:
    """Build the wedge composite for a single capture.

    Returns a BGR ndarray ready to imwrite as a PNG.
    """
    with open(json_path) as f:
        data = json.load(f)
    M = np.asarray(data["_meta"]["registration"]["affine"],
                   dtype=np.float32)

    Y_raw, U_raw, V_raw, _ = tp_measure.extract_frame(capture_path,
                                                      frame_index)
    Y, U, V, _padding = tp_measure.pad_to_486(Y_raw, U_raw, V_raw)
    bgr = tp_synthesize._yuv422p10_to_bgr8(Y, U, V)

    x0, y0, x1, y1 = _wedge_box_in_capture(M)
    h_img, w_img = bgr.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_img, x1); y1 = min(h_img, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    crop = bgr[y0:y1, x0:x1].copy()
    # Upscale 4× nearest-neighbour so high-frequency stripes are
    # readable in the report.
    import cv2
    h, w = crop.shape[:2]
    crop_up = cv2.resize(crop, (w * _UPSCALE, h * _UPSCALE),
                         interpolation=cv2.INTER_NEAREST)
    label = _label_strip(width_px=80)
    # Match label height to crop height (defensive: crop may differ
    # from theoretical due to clamping).
    if label.shape[0] != crop_up.shape[0]:
        label = cv2.resize(label, (label.shape[1], crop_up.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return np.concatenate([crop_up, label], axis=1)


def _main():
    import argparse
    import cv2
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture file (mov/avi/...)")
    p.add_argument("json",    help="per-capture _stage2.json from tp_measure")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--frame", type=int, default=60)
    args = p.parse_args()
    bgr = build(args.capture, args.json, args.frame)
    cv2.imwrite(args.output, bgr)
    print(f"wrote {args.output} ({bgr.shape[1]}x{bgr.shape[0]})")


if __name__ == "__main__":
    _main()
