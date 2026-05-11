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


def _detect_boundary_triangle(Y, fid):
    """Detect a boundary triangle by its back corners.

    fid carries: ideal_back_corner_1, ideal_back_corner_2,
                 ideal_back_midpoint, ideal_apex, orientation,
                 search_window_px.

    Algorithm:
      1. Crop a narrow vertical band (ideal_back_y +/- 4 rows) x full
         horizontal window centered on ideal_back_midpoint.
      2. Threshold dark pixels: Y10 < 0.3 * GREY_BACKGROUND_Y10.
      3. Reject rows with fewer than BASE_HALF - 2 dark pixels (grid lines
         give ~3 px; the triangle back edge gives ~20 px).
      4. Find the back edge row based on orientation:
           apex_up   -> bottommost qualifying row
           apex_down -> topmost qualifying row
      5. Compute center_x as a darkness-weighted centroid across all
         qualifying rows. The triangle is symmetric about bm_x.
      6. back_corner_1 = (center_x - BASE_HALF, back_y)
         back_corner_2 = (center_x + BASE_HALF, back_y)
      7. back_midpoint = (center_x, back_y)
         apex_inferred = back_midpoint + (ideal_apex - ideal_back_midpoint)
      8. apex_detected: search a small window around apex_inferred; set to
         None if the nearest dark pixel is more than 2 rows away from the
         chart-spec distance.
    """
    h, w = Y.shape
    bm_x, bm_y = fid["ideal_back_midpoint"]
    ideal_apex = fid["ideal_apex"]
    ideal_bm = fid["ideal_back_midpoint"]
    offset = (ideal_apex[0] - ideal_bm[0], ideal_apex[1] - ideal_bm[1])
    # Half-width of the back edge from chart geometry.
    base_half = (fid["ideal_back_corner_2"][0] - fid["ideal_back_corner_1"][0]) / 2.0
    # Minimum dark pixels per row to qualify as a triangle row (not a grid line).
    min_dark_for_back = max(3, int(base_half) - 2)

    orient = fid["orientation"]
    half_x = fid["search_window_px"] // 2
    x0 = max(0, int(bm_x) - half_x)
    x1 = min(w, int(bm_x) + half_x)

    # Narrow vertical band around the ideal back edge.
    back_y_margin = 4
    y_lo = max(0, int(bm_y) - back_y_margin)
    y_hi = min(h, int(bm_y) + back_y_margin + 1)
    band = Y[y_lo:y_hi, x0:x1].astype(np.float32)
    if band.size == 0:
        return None

    threshold = 0.3 * tp_chart.GREY_BACKGROUND_Y10
    darkness = np.maximum(0.0, threshold - band)
    dark_row_counts = np.array(
        [int((darkness[r, :] > 0).sum()) for r in range(darkness.shape[0])]
    )

    # Find qualifying (wide-dark) rows — these are triangle rows, not grid lines.
    wide_rows = np.where(dark_row_counts >= min_dark_for_back)[0]
    if len(wide_rows) == 0:
        return None

    confidence = min(1.0, float(dark_row_counts[wide_rows].sum()) / band.size)

    # Back row: orientation determines which extreme is the back edge.
    if orient == "apex_up":
        back_local_r = int(wide_rows.max())    # bottommost wide row
    elif orient == "apex_down":
        back_local_r = int(wide_rows.min())    # topmost wide row
    else:
        raise ValueError(f"unsupported orientation: {orient}")
    abs_back_y = float(y_lo + back_local_r)

    # Darkness-weighted centroid of x across all qualifying rows.
    total_weight = 0.0
    weighted_cx = 0.0
    for r in wide_rows:
        row_dark = darkness[r, :]
        rw = float(row_dark.sum())
        if rw == 0.0:
            continue
        xs = np.arange(row_dark.shape[0], dtype=np.float32)
        cx = float((row_dark * xs).sum() / rw)
        weighted_cx += rw * (x0 + cx)
        total_weight += rw
    if total_weight == 0.0:
        return None
    center_x = weighted_cx / total_weight

    bc1 = (center_x - base_half, abs_back_y)
    bc2 = (center_x + base_half, abs_back_y)
    back_midpoint = (center_x, abs_back_y)
    apex_inferred = (back_midpoint[0] + offset[0],
                     back_midpoint[1] + offset[1])

    # Quick clip-detection check: if the pixel at apex_inferred is in a
    # "sub-black" region (Y well below BLACK_Y10), the frame edge has been
    # zeroed (hardware clip or synthesized edge mask). Skip apex detection
    # entirely so that apex_detected = None signals the clip.
    ai_x_clamped = max(0, min(w - 1, int(round(apex_inferred[0]))))
    ai_y_clamped = max(0, min(h - 1, int(round(apex_inferred[1]))))
    apex_region_y = float(Y[ai_y_clamped, ai_x_clamped])
    _CLIP_DETECT_THRESHOLD = tp_chart.BLACK_Y10 / 2.0  # Y much < 32 => clipped
    if apex_region_y < _CLIP_DETECT_THRESHOLD:
        return {
            "back_corner_1": bc1,
            "back_corner_2": bc2,
            "back_midpoint": back_midpoint,
            "apex_inferred": apex_inferred,
            "apex_detected": None,
            "confidence": confidence,
        }

    # Apex detection: search a small window centered on apex_inferred.
    apex_margin = 3
    ai_y = apex_inferred[1]
    ay_lo = max(0, int(ai_y) - apex_margin)
    ay_hi = min(h, int(ai_y) + apex_margin + 1)
    apex_win = Y[ay_lo:ay_hi, x0:x1].astype(np.float32)
    apex_dark_mask = apex_win < threshold
    apex_ys, apex_xs = np.where(apex_dark_mask)

    chart_spec_dist = abs(offset[1])
    apex_detected = None
    if len(apex_ys) > 0:
        local_cx = center_x - x0
        # A true apex tip is narrow (≤ back-edge width). Rows where nearly
        # all pixels are dark are chart borders or clipped regions — skip them.
        # Max width of a genuine near-center apex hit = 2 * base_half.
        max_apex_width = int(2 * base_half)
        # Collect all rows within distance tolerance whose dark pixels are
        # narrow (apex-like) and near center_x. Then pick the extreme
        # (topmost for apex_up, bottommost for apex_down). This skips
        # chart-border rows (fully dark) that lie at the wrong distance,
        # while still finding the real apex within the search window.
        valid_candidates = []
        for cand_r in np.unique(apex_ys):
            cand_r = int(cand_r)
            abs_cand_y = float(ay_lo + cand_r)
            observed_dist = abs(abs_cand_y - abs_back_y)
            if abs(observed_dist - chart_spec_dist) > 2:
                continue
            mask_r = apex_ys == cand_r
            xs_at_r = apex_xs[mask_r]
            near_center = xs_at_r[np.abs(xs_at_r - local_cx) <= base_half + 2]
            # Skip rows that are too wide to be a triangle apex tip.
            if len(near_center) == 0 or len(near_center) > max_apex_width:
                continue
            valid_candidates.append((cand_r, float(near_center.mean())))
        if valid_candidates:
            if orient == "apex_up":
                cand_r, apex_lx = min(valid_candidates, key=lambda c: c[0])
            else:
                cand_r, apex_lx = max(valid_candidates, key=lambda c: c[0])
            abs_cand_y = float(ay_lo + cand_r)
            apex_detected = (x0 + apex_lx, abs_cand_y)

    return {
        "back_corner_1": bc1,
        "back_corner_2": bc2,
        "back_midpoint": back_midpoint,
        "apex_inferred": apex_inferred,
        "apex_detected": apex_detected,
        "confidence": confidence,
    }


_RC_TEMPLATE_CACHE = None


def _build_registration_cross_template():
    """Build a 24x24 template matching tp_chart.REGISTRATION_CROSS: black box
    with centered white + (3 px arm thickness, 17 px arm length)."""
    rc = tp_chart.REGISTRATION_CROSS
    size = rc["box_size_px"]
    t = np.full((size, size), 0, dtype=np.float32)  # black box
    half = size // 2
    half_len = rc["ideal_arm_len_px"] // 2
    half_th = rc["ideal_arm_thickness_px"] // 2
    t[half - half_th:half + half_th + 1, half - half_len:half + half_len + 1] = 255.0
    t[half - half_len:half + half_len + 1, half - half_th:half + half_th + 1] = 255.0
    return t


def _registration_cross_template():
    global _RC_TEMPLATE_CACHE
    if _RC_TEMPLATE_CACHE is None:
        _RC_TEMPLATE_CACHE = _build_registration_cross_template()
    return _RC_TEMPLATE_CACHE


def _parabolic_subpixel(left, center, right):
    """1D parabolic fit; returns offset in [-1, +1] from center."""
    denom = (left + right - 2.0 * center)
    if abs(denom) < 1e-9:
        return 0.0
    return float(0.5 * (left - right) / denom)


def _count_bright_run(line, center_idx):
    """Walk left and right from center_idx along `line`, counting pixels with
    Y10 > 700 contiguously. Returns total run length including center."""
    n = len(line)
    threshold = 700
    count = 1 if line[center_idx] > threshold else 0
    for i in range(center_idx - 1, -1, -1):
        if line[i] > threshold:
            count += 1
        else:
            break
    for i in range(center_idx + 1, n):
        if line[i] > threshold:
            count += 1
        else:
            break
    return count


def _detect_registration_cross(Y, fid):
    import cv2
    h, w = Y.shape
    half = fid["search_window_px"] // 2
    cx, cy = fid["ideal_x"], fid["ideal_y"]
    x0 = max(0, cx - half); y0 = max(0, cy - half)
    x1 = min(w, cx + half); y1 = min(h, cy + half)
    win = Y[y0:y1, x0:x1].astype(np.float32)
    # Normalize window to 0..255 for matchTemplate consistency with the template.
    win_n = np.clip(
        (win - tp_chart.BLACK_Y10) / (tp_chart.WHITE_Y10 - tp_chart.BLACK_Y10),
        0, 1,
    ) * 255.0
    win_n = win_n.astype(np.float32)
    template = _registration_cross_template()
    if win_n.shape[0] < template.shape[0] or win_n.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(win_n, template, cv2.TM_SQDIFF_NORMED)
    min_val, _max, min_loc, _maxloc = cv2.minMaxLoc(result)
    confidence = float(1.0 - min_val)
    if confidence < 0.4:
        return None
    px, py = min_loc
    rh, rw = result.shape
    if 1 <= px < rw - 1 and 1 <= py < rh - 1:
        dx = _parabolic_subpixel(result[py, px - 1], result[py, px], result[py, px + 1])
        dy = _parabolic_subpixel(result[py - 1, px], result[py, px], result[py + 1, px])
    else:
        dx = 0.0
        dy = 0.0
    template_h, template_w = template.shape
    cross_x_local = px + dx + template_w / 2.0
    cross_y_local = py + dy + template_h / 2.0
    cross_x = x0 + cross_x_local
    cross_y = y0 + cross_y_local
    # Measure arm lengths along center lines.
    cy_int = int(round(cross_y))
    cx_int = int(round(cross_x))
    cy_int = max(0, min(h - 1, cy_int))
    cx_int = max(0, min(w - 1, cx_int))
    h_arm_len = _count_bright_run(Y[cy_int, :], cx_int)
    v_arm_len = _count_bright_run(Y[:, cx_int], cy_int)
    return {
        "x": float(cross_x),
        "y": float(cross_y),
        "h_arm_len_px": float(h_arm_len),
        "v_arm_len_px": float(v_arm_len),
        "confidence": confidence,
    }


def _detect_black_circle(Y, fid):
    import cv2
    h, w = Y.shape
    cx_ideal = fid["ideal_cx"]
    cy_ideal = fid["ideal_cy"]
    r_ideal = fid["expected_radius_px"]
    band = fid["search_band_px"]
    # Annular mask: broad band used to catch all ring pixels.
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx_ideal) ** 2 + (yy - cy_ideal) ** 2)
    annulus = (dist >= (r_ideal - band)) & (dist <= (r_ideal + band))
    threshold = 0.3 * tp_chart.GREY_BACKGROUND_Y10
    dark = (Y < threshold) & annulus
    dark_count = int(dark.sum())
    if dark_count < 100:
        return None
    ys, xs = np.where(dark)
    if len(xs) < 5:
        return None
    # Midline selection: bin dark pixels by angle; pick the one nearest the
    # ideal radius in each angular bin. This gives one representative point
    # per degree around the ring, unbiased by ring thickness.
    actual_dist = np.sqrt((xs - cx_ideal) ** 2 + (ys - cy_ideal) ** 2)
    dev_from_ideal = np.abs(actual_dist - r_ideal)
    angles = np.arctan2(ys - cy_ideal, xs - cx_ideal)
    n_bins = 360
    bin_idx = ((angles + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins
    midline_xs = []
    midline_ys = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        best = int(np.argmin(dev_from_ideal[mask]))
        midline_xs.append(int(xs[mask][best]))
        midline_ys.append(int(ys[mask][best]))
    if len(midline_xs) < 5:
        return None
    mxs = np.array(midline_xs, dtype=np.float32)
    mys = np.array(midline_ys, dtype=np.float32)
    pts_mid = np.column_stack([mxs, mys]).astype(np.float32)
    (cx, cy), (axis_a, axis_b), rot_deg = cv2.fitEllipse(pts_mid)
    rx = min(axis_a, axis_b) / 2.0
    ry = max(axis_a, axis_b) / 2.0
    # fit_rms: radial residual on the midline point cloud.
    theta = np.arctan2(mys - cy, mxs - cx)
    rot_rad = np.deg2rad(rot_deg)
    cos_t = np.cos(theta - rot_rad)
    sin_t = np.sin(theta - rot_rad)
    expected_r = (rx * ry) / np.sqrt((ry * cos_t) ** 2 + (rx * sin_t) ** 2 + 1e-9)
    actual_r_mid = np.sqrt((mxs - cx) ** 2 + (mys - cy) ** 2)
    fit_rms = float(np.sqrt(((actual_r_mid - expected_r) ** 2).mean()))
    if fit_rms > 5.0:
        return None
    confidence = max(0.0, 1.0 - fit_rms / 10.0)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "rx": float(rx),
        "ry": float(ry),
        "rotation_deg": float(rot_deg),
        "fit_rms": fit_rms,
        "confidence": confidence,
    }


def detect_fiducial(Y, fid):
    """Dispatch by fid['kind'] to the right detector implementation.
    Returns a detector-specific value (kind-dependent shape) or None."""
    kind = fid["kind"]
    if kind == "grid_intersection":
        return _detect_grid_intersection(Y, fid)
    if kind == "boundary_triangle":
        return _detect_boundary_triangle(Y, fid)
    if kind == "registration_cross":
        return _detect_registration_cross(Y, fid)
    if kind == "black_circle":
        return _detect_black_circle(Y, fid)
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


def _apply_affine_pt(M, x, y):
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def detect_geometry(Y, M_initial):
    """Run all Stage 2 detectors against captured frame Y and compute
    derived geometry.

    M_initial: 2x3 affine mapping ideal -> capture coords. Used to project
               each fiducial's ideal search-window center into capture
               coords for the detector's search.
    """
    h, w = Y.shape
    fiducials = {"triangles": {}, "cross": None, "circle": None}

    # Triangles -- shift ideal_back_midpoint and ideal_apex through M_initial
    # so the detector searches the right capture-coord region. Keep the chart-
    # spec offsets (apex - back_midpoint) for use in apex_inferred.
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        bm = tri["ideal_back_midpoint"]
        ap = tri["ideal_apex"]
        bm_proj = _apply_affine_pt(M_initial, bm[0], bm[1])
        ap_proj = _apply_affine_pt(M_initial, ap[0], ap[1])
        tri_capture = dict(
            tri,
            ideal_back_midpoint=(int(round(bm_proj[0])), int(round(bm_proj[1]))),
            ideal_apex=(float(ap_proj[0]), float(ap_proj[1])),
        )
        result = _detect_boundary_triangle(Y, tri_capture)
        fiducials["triangles"][tri["id"]] = result

    # Cross
    rc = tp_chart.REGISTRATION_CROSS
    proj_x, proj_y = _apply_affine_pt(M_initial, rc["ideal_x"], rc["ideal_y"])
    rc_capture = dict(rc, ideal_x=int(round(proj_x)), ideal_y=int(round(proj_y)))
    fiducials["cross"] = _detect_registration_cross(Y, rc_capture)

    # Circle
    bc = tp_chart.BLACK_CIRCLE
    proj_x, proj_y = _apply_affine_pt(M_initial, bc["ideal_cx"], bc["ideal_cy"])
    bc_capture = dict(bc, ideal_cx=float(proj_x), ideal_cy=float(proj_y))
    fiducials["circle"] = _detect_black_circle(Y, bc_capture)

    derived = _derive_geometry(fiducials, M_initial, w, h)
    return {"fiducials": fiducials, "derived": derived}


def _derive_geometry(fiducials, M, width, height):
    tris = fiducials["triangles"]
    derived = {}

    def _apex_inferred(tid):
        t = tris.get(tid)
        return t["apex_inferred"] if t is not None else None

    TL = _apex_inferred("TL"); TR = _apex_inferred("TR")
    BL = _apex_inferred("BL"); BR = _apex_inferred("BR")

    if all(p is not None for p in (TL, TR, BL, BR)):
        active_picture_box = {
            "top":    (TL[1] + TR[1]) / 2.0,
            "bottom": (BL[1] + BR[1]) / 2.0,
            "left":   (TL[0] + BL[0]) / 2.0,
            "right":  (TR[0] + BR[0]) / 2.0,
        }
        derived["active_picture_box"] = active_picture_box
        derived["picture_extent_px"] = {
            "width":  active_picture_box["right"]  - active_picture_box["left"],
            "height": active_picture_box["bottom"] - active_picture_box["top"],
        }
        corners = tp_chart.IDEAL_PICTURE_BOX_CORNERS
        ideal_in_cap = [_apply_affine_pt(M, x, y) for x, y in corners]
        ideal_left = min(p[0] for p in ideal_in_cap)
        ideal_top  = min(p[1] for p in ideal_in_cap)
        derived["picture_offset_from_ideal"] = {
            "dx": active_picture_box["left"] - ideal_left,
            "dy": active_picture_box["top"]  - ideal_top,
        }
        top_w = TR[0] - TL[0]
        bot_w = BR[0] - BL[0]
        left_h = BL[1] - TL[1]
        right_h = BR[1] - TR[1]
        derived["corner_skew_px"] = {
            "top_vs_bottom_width_diff":  abs(top_w - bot_w),
            "left_vs_right_height_diff": abs(left_h - right_h),
        }
        derived["arrow_tip_coords"] = {
            "TL": TL, "TR": TR, "BL": BL, "BR": BR,
        }
    else:
        for key in ("active_picture_box", "picture_extent_px",
                    "picture_offset_from_ideal", "corner_skew_px",
                    "arrow_tip_coords"):
            derived[key] = None

    # Clip detection per triangle.
    clip = {}
    for tid in ("TL", "TR", "BL", "BR"):
        t = tris.get(tid)
        if t is None:
            clip[tid] = {"apex_visible": False, "clip_px": None,
                         "interpretation": "triangle not detected"}
            continue
        apex_visible = t["apex_detected"] is not None
        clip_px = 0.0
        interp = "no clip detected"
        ax, ay = t["apex_inferred"]
        orient = next(tt["orientation"] for tt in tp_chart.BOUNDARY_TRIANGLES
                      if tt["id"] == tid)
        if not apex_visible:
            if orient == "apex_up":
                clip_px = max(0.0, -ay)
                if ay < 1:
                    clip_px = max(clip_px, 1.0 - ay)
                interp = f"top edge clipped ~{clip_px:.0f} px" if clip_px > 0 else "apex not detected"
            elif orient == "apex_down":
                clip_px = max(0.0, ay - (height - 1))
                if ay > height - 2:
                    clip_px = max(clip_px, ay - (height - 2))
                interp = f"bottom edge clipped ~{clip_px:.0f} px" if clip_px > 0 else "apex not detected"
        clip[tid] = {
            "apex_visible": apex_visible,
            "clip_px": float(clip_px),
            "interpretation": interp,
        }
    derived["clip_detected"] = clip

    rc = fiducials["cross"]
    if rc is not None:
        ideal_cx, ideal_cy = _apply_affine_pt(
            M, tp_chart.REGISTRATION_CROSS["ideal_x"],
               tp_chart.REGISTRATION_CROSS["ideal_y"])
        derived["cross_offset_from_ideal"] = [rc["x"] - ideal_cx,
                                              rc["y"] - ideal_cy]
        h_arm = rc["h_arm_len_px"]; v_arm = rc["v_arm_len_px"]
        if max(h_arm, v_arm) > 0:
            derived["aperture_symmetry"] = float(min(h_arm, v_arm) / max(h_arm, v_arm))
        else:
            derived["aperture_symmetry"] = None
    else:
        derived["cross_offset_from_ideal"] = None
        derived["aperture_symmetry"] = None

    bc = fiducials["circle"]
    if bc is not None:
        rx, ry = bc["rx"], bc["ry"]
        derived["aspect_ratio_check"] = float(min(rx, ry) / max(rx, ry))
        if derived.get("picture_extent_px") is not None:
            derived["diameter_vs_picture_height"] = float(
                2.0 * max(rx, ry) / derived["picture_extent_px"]["height"]
            )
        else:
            derived["diameter_vs_picture_height"] = None
        derived["circle_fit_rms"] = float(bc["fit_rms"])
    else:
        derived["aspect_ratio_check"] = None
        derived["diameter_vs_picture_height"] = None
        derived["circle_fit_rms"] = None

    return derived


def register_with_geometry(Y):
    """Sequential-with-feedback registration.

    Returns:
        {
            "initial":  <Stage 1 register() result>,
            "geometry": <detect_geometry() result>,
            "final":    <fit_affine() result on grids+geometry anchors>,
            "anchors_added": [<ids appended after Stage 1>],
        }
    """
    initial = register(Y)
    if initial["affine_matrix"] is None:
        return {"initial": initial, "geometry": None, "final": initial,
                "anchors_added": []}
    M_initial = initial["affine_matrix"]
    geometry = detect_geometry(Y, M_initial)

    detected_pts = []
    ideal_pts = []
    anchors_added = []
    for lm in tp_chart.GRID_LANDMARKS:
        det = detect_fiducial(Y, lm)
        if det is None:
            continue
        dx, dy, _ = det
        detected_pts.append((dx, dy))
        ideal_pts.append((lm["ideal_x"], lm["ideal_y"]))

    tris = geometry["fiducials"]["triangles"]
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        t = tris.get(tri["id"])
        if t is None:
            continue
        for cap_key, ideal_key, suffix in (
            ("back_corner_1", "ideal_back_corner_1", "bc1"),
            ("back_corner_2", "ideal_back_corner_2", "bc2"),
        ):
            detected_pts.append(t[cap_key])
            ideal_pts.append(tri[ideal_key])
            anchors_added.append(f"{tri['id']}.{suffix}")

    rc = geometry["fiducials"]["cross"]
    if rc is not None:
        detected_pts.append((rc["x"], rc["y"]))
        ideal_pts.append((tp_chart.REGISTRATION_CROSS["ideal_x"],
                          tp_chart.REGISTRATION_CROSS["ideal_y"]))
        anchors_added.append("RC")

    final = fit_affine(np.asarray(detected_pts, dtype=np.float32),
                       np.asarray(ideal_pts, dtype=np.float32))
    final["total"] = len(detected_pts)
    if final["affine_matrix"] is None or final["inliers"] < MIN_INLIERS:
        final["quality_flag"] = "failed"
        final["quality_reason"] = "RANSAC failed or too few inliers"
    elif (final["residuals_px"]["mean"] > RESIDUAL_OK_MEAN_PX
          or final["residuals_px"]["max"] > RESIDUAL_OK_MAX_PX):
        final["quality_flag"] = "warn"
        final["quality_reason"] = "residuals exceed threshold"
    else:
        final["quality_flag"] = "ok"
        final["quality_reason"] = None

    return {
        "initial": initial,
        "geometry": geometry,
        "final": final,
        "anchors_added": anchors_added,
    }
