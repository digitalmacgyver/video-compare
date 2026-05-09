"""Pure metric functions and per-clip analysis.

All metrics are brightness-agnostic — they normalize by mean Y or use
inherently scale-invariant computations. Used by quality_report.py
(full report pipeline) and quality_metrics.py (compute-only pipeline).

This module has no presentation concerns — no HTML, CSS, or text
formatting lives here.
"""

import os
import subprocess
import numpy as np
import cv2

from common import decode_command, read_frame


# =====================================================================
# METRIC KEY CONSTANTS
# =====================================================================

DETAIL_PERCEPTUAL_KEY = "detail_perceptual"
DETAIL_PERCEPTUAL_DEPS = ("detail_blur_inv", "detail_sml", "texture_quality")
DERIVED_KEYS = {DETAIL_PERCEPTUAL_KEY}

# Available extra-detail metric keys. Only the subset listed in
# --extra-detail-metrics (default: detail_perceptual,detail_blur_inv,detail_sml)
# is computed by default; detail_tenengrad is available but opt-in.
EXTRA_DETAIL_KEYS = ["detail_tenengrad", "detail_sml", "detail_blur_inv", DETAIL_PERCEPTUAL_KEY]

# Metrics expressed as percentages (0..1 ratios stored, displayed as %).
PCT_KEYS = {"crushed_blacks", "blown_whites"}


# =====================================================================
# HELPERS
# =====================================================================

def to_float(y):
    """Convert 10-bit uint16 Y plane to float64 [0, 1]."""
    return y.astype(np.float64) / 1023.0


def parse_metric_csv(csv_text):
    """Parse a comma-separated metric key list."""
    if not csv_text:
        return []
    return [x.strip() for x in csv_text.split(",") if x.strip()]


def _zscore_with_params(arr, mu, sigma):
    """Return z-scores with guard for near-zero sigma."""
    if sigma < 1e-15:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mu) / sigma


# =====================================================================
# CLIP NAMING
# =====================================================================

def short_name(filepath):
    """Extract a readable device name from capture filenames.

    Strips common suffixes and extracts the first meaningful token.
    Works with both original (twister_*_IVTC.mov) and bakeoff (*_NOIVTC.mov) naming.
    """
    name = os.path.basename(filepath)
    base, _ = os.path.splitext(name)
    name = base.replace("_normalized", "")
    for suffix in ["_trimmed_VIEW_ACTION_SAFE_IVTC", "_trimmed_VIEW_ACTION_SAFE_NOIVTC",
                   "_svideo_direct", "_svideo_sdi1", "_svideo_sdi"]:
        name = name.replace(suffix, "")
    name = name.replace("twister_", "")
    if " " in name:
        name = name.split()[0]
    return name


# =====================================================================
# METRIC FUNCTIONS
# =====================================================================

def sharpness_laplacian(yf):
    """Laplacian variance normalized by mean brightness squared — brightness-agnostic."""
    mean_y = np.mean(yf)
    return np.var(cv2.Laplacian(yf, cv2.CV_64F)) / (mean_y ** 2 + 1e-10)


def edge_strength_sobel(yf):
    """Mean Sobel gradient magnitude normalized by mean brightness — brightness-agnostic."""
    sx = cv2.Sobel(yf, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(yf, cv2.CV_64F, 0, 1, ksize=3)
    mean_y = np.mean(yf)
    return np.mean(np.sqrt(sx * sx + sy * sy)) / (mean_y + 1e-10)


def blocking_artifact_measure(yf):
    """Detect 8x8 DCT blocking artifacts. Ratio > 1.0 = blocking present."""
    h, w = yf.shape
    h_diffs = np.abs(yf[:, 1:] - yf[:, :-1])
    h_boundary = np.arange(1, w) % 8 == 0
    h_ratio = np.mean(h_diffs[:, h_boundary]) / max(np.mean(h_diffs[:, ~h_boundary]), 1e-10)

    v_diffs = np.abs(yf[1:, :] - yf[:-1, :])
    v_boundary = np.arange(1, h) % 8 == 0
    v_ratio = np.mean(v_diffs[v_boundary, :]) / max(np.mean(v_diffs[~v_boundary, :]), 1e-10)

    return (h_ratio + v_ratio) / 2.0


def detail_texture(yf):
    """Median local coefficient of variation in 16x16 blocks — brightness-agnostic detail."""
    block = 16
    h, w = yf.shape
    nh, nw = h // block, w // block
    if nh == 0 or nw == 0:
        return 0.0
    cropped = yf[:nh * block, :nw * block]
    blocks = cropped.reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(-1, block, block)
    means = blocks.mean(axis=(1, 2))
    stds = blocks.std(axis=(1, 2))
    mask = means > 0.01
    if not mask.any():
        return 0.0
    cvs = stds[mask] / means[mask]
    return float(np.median(cvs))


def detail_tenengrad(yf):
    """Robust Tenengrad detail metric with residual-noise suppression."""
    yd = cv2.GaussianBlur(yf, (0, 0), 1.0)
    gx = cv2.Sobel(yd, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(yd, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    residual = yf - yd
    rx = cv2.Sobel(residual, cv2.CV_64F, 1, 0, ksize=3)
    ry = cv2.Sobel(residual, cv2.CV_64F, 0, 1, ksize=3)
    noise_grad = np.sqrt(rx * rx + ry * ry)
    noise_floor = np.median(noise_grad)

    signal_grad = np.maximum(grad - 1.5 * noise_floor, 0.0)
    return float(np.mean(signal_grad) / (np.mean(yf) + 1e-10))


def detail_sml(yf):
    """Robust sum-modified-Laplacian detail metric (Nayar focus measure family)."""
    yd = cv2.GaussianBlur(yf, (0, 0), 1.0)
    center = yd[1:-1, 1:-1]
    if center.size == 0:
        return 0.0

    ml_x = np.abs(2.0 * center - yd[1:-1, :-2] - yd[1:-1, 2:])
    ml_y = np.abs(2.0 * center - yd[:-2, 1:-1] - yd[2:, 1:-1])
    ml = ml_x + ml_y

    residual = yf - yd
    rc = residual[1:-1, 1:-1]
    ml_nx = np.abs(2.0 * rc - residual[1:-1, :-2] - residual[1:-1, 2:])
    ml_ny = np.abs(2.0 * rc - residual[:-2, 1:-1] - residual[2:, 1:-1])
    noise_floor = np.median(ml_nx + ml_ny)

    signal_ml = np.maximum(ml - 2.0 * noise_floor, 0.0)
    return float(np.mean(signal_ml) / (np.mean(yf) + 1e-10))


def detail_blur_inv(yf, h_size=11):
    """Inverse blur-effect detail metric (Crete et al.) with flat-region noise penalty."""
    if h_size < 3:
        h_size = 3
    if h_size % 2 == 0:
        h_size += 1

    y_smooth = cv2.GaussianBlur(yf, (0, 0), 0.9)
    y_blur = cv2.blur(y_smooth, (h_size, h_size))
    eps = 1e-12
    crop = (slice(2, -1), slice(2, -1))

    axis_blur = []
    for dx, dy in ((1, 0), (0, 1)):
        grad_sharp = np.abs(cv2.Sobel(y_smooth, cv2.CV_64F, dx, dy, ksize=3))
        grad_blur = np.abs(cv2.Sobel(y_blur, cv2.CV_64F, dx, dy, ksize=3))
        t = np.maximum(0.0, grad_sharp - grad_blur)
        m1 = np.sum(np.maximum(grad_sharp[crop], eps))
        m2 = np.sum(t[crop])
        axis_blur.append(np.abs(m1 - m2) / (m1 + eps))

    detail_score = 1.0 - max(axis_blur)

    gx = cv2.Sobel(y_smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_smooth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    flat = grad_mag < np.percentile(grad_mag, 35)
    residual = np.abs(yf - y_smooth)
    noise = np.median(residual[flat]) if flat.any() else np.median(residual)
    noise_penalty = 1.2 * noise / (np.mean(yf) + eps)

    return float(max(0.0, detail_score - noise_penalty))


def texture_quality_measure(yf):
    """Structured detail ratio — how much local variance is structure vs. noise.

    Compares local variance of a mildly smoothed version to the original.
    Structured detail survives smoothing; noise does not.
    Returns ratio close to 1.0 for clean structured detail, lower for noisy detail.
    """
    block = 16
    h, w = yf.shape
    nh, nw = h // block, w // block
    if nh == 0 or nw == 0:
        return 0.5
    yf_smooth = cv2.GaussianBlur(yf, (5, 5), 1.0)
    cropped = yf[:nh * block, :nw * block]
    cropped_s = yf_smooth[:nh * block, :nw * block]
    blocks_o = cropped.reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(-1, block, block)
    blocks_s = cropped_s.reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(-1, block, block)
    var_orig = blocks_o.var(axis=(1, 2))
    var_smooth = blocks_s.var(axis=(1, 2))
    mask = var_orig > 1e-8
    if mask.sum() < 5:
        return 0.5
    ratios = var_smooth[mask] / var_orig[mask]
    return float(np.median(ratios))


def ringing_measure(yf):
    """Detect ringing/haloing — near-edge Laplacian normalized by mean brightness."""
    edges = cv2.Canny((yf * 255).astype(np.uint8), 50, 150)
    if edges.sum() == 0:
        return 0.0
    near_edge = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    near_edge_only = near_edge & ~edges
    if near_edge_only.sum() == 0:
        return 0.0
    mean_y = np.mean(yf)
    return np.mean(np.abs(cv2.Laplacian(yf, cv2.CV_64F))[near_edge_only > 0]) / (mean_y + 1e-10)


def colorfulness_metric(u, v):
    """Hasler & Süsstrunk (2003) colorfulness metric on Cb/Cr planes."""
    cb = u.astype(np.float64) - 512.0
    cr = v.astype(np.float64) - 512.0
    rg = cr - cb
    yb = 0.5 * (cr + cb)
    sigma_rgyb = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mu_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return float(sigma_rgyb + 0.3 * mu_rgyb)


def crushed_blacks(yf):
    """Shadow headroom ratio — fraction of shadow pixels crushed to near-black."""
    SHADOW_CEIL = 0.15
    CRUSH_FLOOR = 0.07
    n_shadow = np.count_nonzero(yf < SHADOW_CEIL)
    if n_shadow == 0:
        return 0.0
    return float(np.count_nonzero(yf < CRUSH_FLOOR)) / float(n_shadow)


def blown_whites(yf):
    """Highlight headroom ratio — fraction of highlight pixels blown to near-white."""
    HIGHLIGHT_FLOOR = 0.85
    BLOW_CEIL = 0.93
    n_highlight = np.count_nonzero(yf > HIGHLIGHT_FLOOR)
    if n_highlight == 0:
        return 0.0
    return float(np.count_nonzero(yf > BLOW_CEIL)) / float(n_highlight)


def naturalness_nss(yf):
    """MSCN kurtosis — natural/film-like signal statistics."""
    C = 1.0 / 1023.0
    mu = cv2.GaussianBlur(yf, (7, 7), 1.166)
    mu_sq = cv2.GaussianBlur(yf * yf, (7, 7), 1.166)
    sigma = np.sqrt(np.maximum(mu_sq - mu * mu, 0.0))
    mscn = (yf - mu) / (sigma + C)
    data = mscn.ravel()
    s = np.std(data)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((data - np.mean(data)) / s) ** 4) - 3.0)


# =====================================================================
# DERIVED METRICS
# =====================================================================

def add_detail_perceptual_metric(all_results, all_perframe):
    """Add derived VHS perceptual detail metric to clip summary and per-frame data.

    detail_perceptual = z(detail_blur_inv) - z(detail_sml) - z(texture_quality)

    Z-normalization is computed across clip means so the three components have
    comparable scale on a given run.
    """
    clip_names = sorted(all_results.keys())
    if not clip_names:
        return False

    for dep in DETAIL_PERCEPTUAL_DEPS:
        missing = [c for c in clip_names if dep not in all_results[c]]
        if missing:
            print(f"WARNING: cannot compute {DETAIL_PERCEPTUAL_KEY}; missing '{dep}' in {len(missing)} clip(s)")
            return False

    mean_blur = np.array([all_results[c]["detail_blur_inv"]["mean"] for c in clip_names], dtype=np.float64)
    mean_sml = np.array([all_results[c]["detail_sml"]["mean"] for c in clip_names], dtype=np.float64)
    mean_tex = np.array([all_results[c]["texture_quality"]["mean"] for c in clip_names], dtype=np.float64)

    mu_b, sd_b = float(np.mean(mean_blur)), float(np.std(mean_blur))
    mu_s, sd_s = float(np.mean(mean_sml)), float(np.std(mean_sml))
    mu_t, sd_t = float(np.mean(mean_tex)), float(np.std(mean_tex))

    z_blur = _zscore_with_params(mean_blur, mu_b, sd_b)
    z_sml = _zscore_with_params(mean_sml, mu_s, sd_s)
    z_tex = _zscore_with_params(mean_tex, mu_t, sd_t)
    mean_scores = z_blur - z_sml - z_tex

    for i, name in enumerate(clip_names):
        arr_blur = np.asarray(all_perframe[name]["detail_blur_inv"], dtype=np.float64)
        arr_sml = np.asarray(all_perframe[name]["detail_sml"], dtype=np.float64)
        arr_tex = np.asarray(all_perframe[name]["texture_quality"], dtype=np.float64)
        n = int(min(len(arr_blur), len(arr_sml), len(arr_tex)))
        if n <= 0:
            frame_scores = np.array([float(mean_scores[i])], dtype=np.float64)
        else:
            frame_scores = (
                _zscore_with_params(arr_blur[:n], mu_b, sd_b)
                - _zscore_with_params(arr_sml[:n], mu_s, sd_s)
                - _zscore_with_params(arr_tex[:n], mu_t, sd_t)
            )

        all_perframe[name][DETAIL_PERCEPTUAL_KEY] = frame_scores
        all_results[name][DETAIL_PERCEPTUAL_KEY] = {
            "mean": float(np.mean(frame_scores)),
            "std": float(np.std(frame_scores)),
        }
    return True


# =====================================================================
# CLIP ANALYSIS
# =====================================================================

def analyze_clip(filepath, width, height, metric_keys, skip_frames=0):
    """Analyze one clip, return (summary_dict, perframe_dict).

    summary_dict has the same format as before: {metric: {mean, std}, n_frames}.
    perframe_dict maps selected metric keys -> numpy array of per-frame values.
    skip_frames: number of initial frames to discard (for timing alignment).
    """
    proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
    accum = {k: [] for k in metric_keys if k != "temporal_stability" and k not in DERIVED_KEYS}
    temporal_diffs = []
    prev_yf = None
    n_frames = 0

    try:
        for _ in range(skip_frames):
            frame = read_frame(proc.stdout, width, height)
            if frame is None:
                break

        while True:
            frame = read_frame(proc.stdout, width, height)
            if frame is None:
                break
            y, u, v = frame
            yf = to_float(y)

            if "sharpness" in accum:
                accum["sharpness"].append(sharpness_laplacian(yf))
            if "edge_strength" in accum:
                accum["edge_strength"].append(edge_strength_sobel(yf))
            if "blocking" in accum:
                accum["blocking"].append(blocking_artifact_measure(yf))
            if "detail" in accum:
                accum["detail"].append(detail_texture(yf))
            if "detail_tenengrad" in accum:
                accum["detail_tenengrad"].append(detail_tenengrad(yf))
            if "detail_sml" in accum:
                accum["detail_sml"].append(detail_sml(yf))
            if "detail_blur_inv" in accum:
                accum["detail_blur_inv"].append(detail_blur_inv(yf))
            if "texture_quality" in accum:
                accum["texture_quality"].append(texture_quality_measure(yf))
            if "ringing" in accum:
                accum["ringing"].append(ringing_measure(yf))
            if "colorfulness" in accum:
                accum["colorfulness"].append(colorfulness_metric(u, v))
            if "naturalness" in accum:
                accum["naturalness"].append(naturalness_nss(yf))
            if "crushed_blacks" in accum:
                accum["crushed_blacks"].append(crushed_blacks(yf))
            if "blown_whites" in accum:
                accum["blown_whites"].append(blown_whites(yf))

            if "temporal_stability" in metric_keys and prev_yf is not None:
                mean_y = 0.5 * (np.mean(yf) + np.mean(prev_yf))
                temporal_diffs.append(np.mean(np.abs(yf - prev_yf)) / (mean_y + 1e-10))
            if "temporal_stability" in metric_keys:
                prev_yf = yf
            n_frames += 1
    finally:
        proc.stdout.close()
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()

    if proc.returncode != 0:
        print(f"  WARNING: ffmpeg exited with code {proc.returncode} for {filepath}")

    if n_frames == 0:
        print(f"  ERROR: No frames decoded for {filepath} (skip={skip_frames})")
        return None, None

    result = {}
    for k, vals in accum.items():
        arr = np.array(vals)
        avg = float(np.median(arr)) if k == "naturalness" else float(np.mean(arr))
        result[k] = {"mean": avg, "std": float(np.std(arr))}

    if "temporal_stability" in metric_keys:
        if temporal_diffs:
            td = np.array(temporal_diffs)
            result["temporal_stability"] = {"mean": float(np.mean(td)), "std": float(np.std(td))}
        else:
            result["temporal_stability"] = {"mean": 0.0, "std": 0.0}

    EPS = 1e-5
    for k in PCT_KEYS:
        if k not in result:
            continue
        if result[k]["mean"] < EPS:
            result[k]["mean"] = 0.0

    result["n_frames"] = n_frames

    perframe = {k: np.array(accum[k]) for k in accum}
    if "temporal_stability" in metric_keys:
        if temporal_diffs:
            perframe["temporal_stability"] = np.array(temporal_diffs)
        else:
            perframe["temporal_stability"] = np.array([0.0])

    return result, perframe
