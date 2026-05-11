"""Registration: detect grid-intersection landmarks and fit an affine
transform from the ideal coordinate system to the captured-frame coords.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

import tp_chart


def _grid_intersection_impl(Y, ideal_x, ideal_y, search_window_px):
    """Find a grid-intersection (a black '+' on grey) inside a search window.

    Approach:
      1. Crop a square search window centred on (ideal_x, ideal_y).
      2. Threshold dark pixels (Y10 < 0.3 * GREY_BACKGROUND_Y10 ~= 150).
      3. Project the dark mask onto each axis; rows/columns whose dark
         count is above the median are treated as the cross's arm spikes.
         Reject the window if there are no spike rows or no spike columns
         (no cross structure).
      4. Compute the centroid of the spike columns and spike rows to
         recover the intersection coordinate. This is robust to the cross
         being off-centre in the search window — a 2D-mass centroid would
         bias toward the window middle.

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

    threshold = 0.3 * tp_chart.GREY_BACKGROUND_Y10
    dark_mask = win < threshold
    dark_count = int(dark_mask.sum())
    if dark_count < 4:
        return None
    confidence = dark_count / win.size

    col_proj = dark_mask.sum(axis=0).astype(np.float32)
    row_proj = dark_mask.sum(axis=1).astype(np.float32)

    col_med = float(np.median(col_proj))
    row_med = float(np.median(row_proj))
    col_high = col_proj > col_med
    row_high = row_proj > row_med

    if col_high.sum() < 1 or row_high.sum() < 1:
        return None

    col_idxs = np.where(col_high)[0].astype(np.float32)
    cx = float((col_proj[col_high] * col_idxs).sum() / col_proj[col_high].sum())
    row_idxs = np.where(row_high)[0].astype(np.float32)
    cy = float((row_proj[row_high] * row_idxs).sum() / row_proj[row_high].sum())

    return (x0 + cx, y0 + cy, confidence)


def _detect_grid_intersection(Y, fid):
    """Grid-intersection detector. fid carries ideal_x, ideal_y,
    search_window_px."""
    return _grid_intersection_impl(
        Y, fid["ideal_x"], fid["ideal_y"], fid["search_window_px"]
    )


def detect_fiducial(Y, fid):
    """Dispatch by fid['kind'] to the right detector implementation.
    Returns a detector-specific value (kind-dependent shape) or None."""
    kind = fid["kind"]
    if kind == "grid_intersection":
        return _detect_grid_intersection(Y, fid)
    # Stage 2 detectors are added in subsequent tasks.
    raise ValueError(f"unknown fiducial kind: {kind}")


def detect_landmark(
    Y: np.ndarray,
    ideal_x: int,
    ideal_y: int,
    search_window_px: int,
) -> Optional[Tuple[float, float, float]]:
    """Backward-compat alias for _grid_intersection_impl."""
    return _grid_intersection_impl(Y, ideal_x, ideal_y, search_window_px)


def fit_affine(
    detected_pts: np.ndarray,
    ideal_pts: np.ndarray,
    inlier_threshold_px: float = 1.5,
) -> Dict[str, Any]:
    """RANSAC 2D affine fit: ideal -> detected.

    Args:
        detected_pts: shape (N, 2) of detected (x, y) in capture coords.
        ideal_pts:    shape (N, 2) of corresponding ideal (x, y).
        inlier_threshold_px: max residual to be considered an inlier.

    Returns:
        {
            "affine_matrix": np.ndarray of shape (2, 3) or None,
            "residuals_px": {"mean": float, "max": float},
            "inliers": int,
            "total": int,
        }
    """
    import cv2

    detected_pts = np.asarray(detected_pts, dtype=np.float32)
    ideal_pts = np.asarray(ideal_pts, dtype=np.float32)
    total = int(len(detected_pts))
    if total < 3:
        return {
            "affine_matrix": None,
            "residuals_px": {"mean": float("nan"), "max": float("nan")},
            "inliers": 0,
            "total": total,
        }

    M, mask = cv2.estimateAffine2D(
        ideal_pts.reshape(-1, 1, 2),
        detected_pts.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(inlier_threshold_px),
        refineIters=10,
    )
    if M is None:
        return {
            "affine_matrix": None,
            "residuals_px": {"mean": float("nan"), "max": float("nan")},
            "inliers": 0,
            "total": total,
        }

    inlier_mask = mask.flatten().astype(bool) if mask is not None else np.ones(total, dtype=bool)
    inliers = int(inlier_mask.sum())

    # Compute residuals on inliers.
    ones = np.ones((total, 1), dtype=np.float32)
    ideal_h = np.hstack([ideal_pts, ones])
    pred = (M @ ideal_h.T).T  # (N, 2)
    diffs = np.linalg.norm(pred - detected_pts, axis=1)
    if inliers > 0:
        mean_res = float(diffs[inlier_mask].mean())
        max_res = float(diffs[inlier_mask].max())
    else:
        mean_res = float(diffs.mean())
        max_res = float(diffs.max())

    return {
        "affine_matrix": M,
        "residuals_px": {"mean": mean_res, "max": max_res},
        "inliers": inliers,
        "total": total,
    }


# Quality-flag thresholds. Placeholder values, calibrated on real data later.
RESIDUAL_OK_MEAN_PX = 2.0
RESIDUAL_OK_MAX_PX = 4.0
MIN_INLIERS = 4


def register(Y: np.ndarray) -> Dict[str, Any]:
    """Top-level: detect all GRID_LANDMARKS and fit an affine.

    Returns a dict suitable for embedding in tp_measure's per-capture JSON
    under `_meta.registration`.
    """
    detected: List[Tuple[float, float]] = []
    ideal: List[Tuple[float, float]] = []
    detected_lm_ids: List[str] = []
    for lm in tp_chart.GRID_LANDMARKS:
        det = detect_fiducial(Y, lm)
        if det is None:
            continue
        dx, dy, _ = det
        detected.append((dx, dy))
        ideal.append((lm["ideal_x"], lm["ideal_y"]))
        detected_lm_ids.append(lm["id"])

    if len(detected) < MIN_INLIERS:
        return {
            "affine_matrix": None,
            "residuals_px": {"mean": float("nan"), "max": float("nan")},
            "inliers": len(detected),
            "total": len(tp_chart.GRID_LANDMARKS),
            "landmarks_used": detected_lm_ids,
            "quality_flag": "failed",
            "quality_reason": f"only {len(detected)} landmark(s) detected",
        }

    fit = fit_affine(np.asarray(detected, dtype=np.float32),
                     np.asarray(ideal, dtype=np.float32))
    fit["total"] = len(tp_chart.GRID_LANDMARKS)
    fit["landmarks_used"] = detected_lm_ids
    if (fit["affine_matrix"] is None
            or fit["inliers"] < MIN_INLIERS):
        fit["quality_flag"] = "failed"
        fit["quality_reason"] = "RANSAC failed or too few inliers"
    elif (fit["residuals_px"]["mean"] > RESIDUAL_OK_MEAN_PX
          or fit["residuals_px"]["max"] > RESIDUAL_OK_MAX_PX):
        fit["quality_flag"] = "warn"
        fit["quality_reason"] = "residuals exceed threshold"
    else:
        fit["quality_flag"] = "ok"
        fit["quality_reason"] = None
    return fit
