"""Registration: detect grid-intersection landmarks and fit an affine
transform from the ideal coordinate system to the captured-frame coords.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

import tp_chart


def detect_landmark(
    Y: np.ndarray,
    ideal_x: int,
    ideal_y: int,
    search_window_px: int,
) -> Optional[Tuple[float, float, float]]:
    """Find a grid-intersection (a black '+' on grey) inside a search window.

    Approach:
      1. Crop a square search window centred on (ideal_x, ideal_y).
      2. Threshold dark pixels (Y10 < 0.3 * GREY_BACKGROUND_Y10 ~= 150).
      3. Verify the dark cluster contains both a horizontal and a vertical
         line component (a single isolated blob is rejected).
      4. Compute the dark-weighted centroid for sub-pixel position.

    Returns (x, y, confidence) in capture-image coords, or None when no
    plausible intersection is found. Confidence is the fraction of dark
    pixels in the search window (0..1).
    """
    h, w = Y.shape
    half = search_window_px // 2
    x0 = max(0, ideal_x - half)
    y0 = max(0, ideal_y - half)
    x1 = min(w, ideal_x + half)
    y1 = min(h, ideal_y + half)
    win = Y[y0:y1, x0:x1].astype(np.float32)
    if win.size == 0:
        return None

    threshold = 0.3 * tp_chart.GREY_BACKGROUND_Y10  # ~150
    dark_mask = win < threshold
    dark_count = int(dark_mask.sum())
    if dark_count < 4:
        return None
    confidence = dark_count / win.size

    # Reject single blob (no cross): require both >=3 columns and >=3 rows
    # to contain dark pixels.
    cols_with_dark = int(dark_mask.any(axis=0).sum())
    rows_with_dark = int(dark_mask.any(axis=1).sum())
    if cols_with_dark < 3 or rows_with_dark < 3:
        return None

    # Weighted centroid: weight = (grey - Y10), clipped non-negative.
    weight = np.clip(tp_chart.GREY_BACKGROUND_Y10 - win, 0.0, None) * dark_mask
    total = float(weight.sum())
    if total <= 0.0:
        return None
    yy, xx = np.indices(win.shape, dtype=np.float32)
    cx = float((weight * xx).sum() / total)
    cy = float((weight * yy).sum() / total)
    return (x0 + cx, y0 + cy, confidence)
