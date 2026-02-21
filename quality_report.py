#!/usr/bin/env python3
"""No-reference video quality metrics for analog video capture comparison.

Analyzes a directory of video clips (resolution auto-detected via ffprobe)
and produces JSON data, an interactive HTML report with Chart.js
visualizations, and optionally a plain-text report.

Usage:
    python quality_report.py <src_dir> [options]

    src_dir             Directory containing normalized .mov files to analyze

Options:
    --output-dir DIR    Where to write reports (default: same as src_dir)
    --name PREFIX       Report filename prefix (default: "quality_report")
    --pattern GLOB      File glob pattern (default: "*.mov")
    --text              Also generate a plain-text report

Examples:
    python quality_report.py /path/to/normalized/
    python quality_report.py /path/to/normalized/ --output-dir /reports/ --name clip02
    python quality_report.py /path/to/normalized/ --pattern "*_sls.mp4" --name sls_report
    python quality_report.py /path/to/normalized/ --text

Metrics (11 total, all brightness-agnostic):
  Sharpness, edge strength, blocking, detail, texture quality, ringing, temporal stability,
  colorfulness, naturalness, crushed blacks, blown whites

Brightness-sensitive metrics (sharpness, edge strength, detail, ringing, temporal stability) are
normalized by mean Y to eliminate dependence on capture brightness / gain settings.  No upstream
brightness normalization is required.

Dropped metrics: noise (subordinate to detail — correlated r=0.83), contrast and tonal richness
(near-zero discrimination across clips), gradient smoothness (r=0.94 duplicate of texture quality).
Ringing retained despite r=0.94 correlation with sharpness — it measures a distinct analog artifact
(edge overshoot from VCR sharpness circuits / aperture correction) that varies independently of
true edge definition across capture hardware.
"""

import argparse
import subprocess
import os
import sys
import glob
import json
import re
import base64
from datetime import datetime
import numpy as np
import cv2
from common import (ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14,
                    compute_composites, probe_video, decode_command, read_frame)

CMP_ICON_SVG = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<rect x="2" y="3" width="20" height="18" rx="2"/>'
    '<line x1="12" y1="3" x2="12" y2="21"/>'
    '<text x="7" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">A</text>'
    '<text x="17" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">B</text></svg>')


def to_float(y):
    """Convert 10-bit uint16 Y plane to float64 [0, 1]."""
    return y.astype(np.float64) / 1023.0


DETAIL_PERCEPTUAL_KEY = "detail_perceptual"
DETAIL_PERCEPTUAL_DEPS = ("detail_blur_inv", "detail_sml", "texture_quality")
DERIVED_KEYS = {DETAIL_PERCEPTUAL_KEY}
EXTRA_DETAIL_KEYS = ["detail_tenengrad", "detail_sml", "detail_blur_inv", DETAIL_PERCEPTUAL_KEY]
PCT_KEYS = {"crushed_blacks", "blown_whites"}
METRIC_GUIDE_TEXT = {
    "sharpness": "Laplacian variance — blur vs. sharpness",
    "edge_strength": "Sobel gradient — edge definition",
    "blocking": "8x8 DCT boundary ratio",
    "detail": "Local variance — micro-detail",
    "detail_perceptual": "Perceptual detail tuned for SD analog VHS (not validated for HD digital sources)",
    "detail_tenengrad": "Robust Tenengrad — edge detail (noise-suppressed)",
    "detail_sml": "Robust modified Laplacian — fine transition clarity",
    "detail_blur_inv": "Inverse blur effect — directional blur resistance",
    "texture_quality": "Structure/noise ratio — detail quality",
    "ringing": "Edge overshoot / haloing",
    "temporal_stability": "Frame-to-frame diff — flicker",
    "colorfulness": "Hasler-S\u00fcstrunk — color vibrancy",
    "naturalness": "MSCN kurtosis — natural signal statistics",
    "crushed_blacks": "Shadow headroom — clipped near-black fraction",
    "blown_whites": "Highlight headroom — clipped near-white fraction",
}


def parse_metric_csv(csv_text):
    """Parse a comma-separated metric key list."""
    if not csv_text:
        return []
    return [x.strip() for x in csv_text.split(",") if x.strip()]


def short_metric_header(key):
    """Return compact table header text for a metric key."""
    label = METRIC_INFO[key][0]
    compact = re.sub(r"[^A-Za-z0-9]+", "", label)
    return compact[:8] if compact else key[:8]


def fmt_metric_value(key, value):
    """Format one metric value for text/HTML tables."""
    if key in PCT_KEYS:
        return f"{value*100:.1f}%"
    if key == "blocking":
        return f"{value:.4f}"
    if key == "colorfulness":
        return f"{value:.1f}"
    if key == "naturalness":
        return f"{value:.3f}"
    return f"{value:.6f}"


# =====================================================================
# TECHNICAL METRICS
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
    mask = means > 0.01  # skip truly black blocks
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

    # Estimate a noise floor from high-frequency residual gradients.
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

    # Residual ML estimates high-frequency noise floor.
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

    # Penalize residual noise measured on flatter regions.
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
    """Shadow headroom ratio — fraction of shadow pixels crushed to near-black.

    Counts pixels below the crush threshold (Y < ~72 in 10-bit) as a fraction
    of all shadow-region pixels (Y < ~153).  High ratio = shadow detail lost
    to clipping.  Returns 0 if no shadow pixels present.
    """
    SHADOW_CEIL = 0.15   # upper bound of shadow region (~Y=153)
    CRUSH_FLOOR = 0.07   # crush threshold (~Y=72, near legal black 64)
    n_shadow = np.count_nonzero(yf < SHADOW_CEIL)
    if n_shadow == 0:
        return 0.0
    return float(np.count_nonzero(yf < CRUSH_FLOOR)) / float(n_shadow)


def blown_whites(yf):
    """Highlight headroom ratio — fraction of highlight pixels blown to near-white.

    Counts pixels above the blow threshold (Y > ~951 in 10-bit) as a fraction
    of all highlight-region pixels (Y > ~869).  High ratio = highlight detail
    lost to clipping.  Returns 0 if no highlight pixels present.
    """
    HIGHLIGHT_FLOOR = 0.85  # lower bound of highlight region (~Y=869)
    BLOW_CEIL = 0.93        # blow threshold (~Y=951, near legal white 940)
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


def _zscore_with_params(arr, mu, sigma):
    """Return z-scores with guard for near-zero sigma."""
    if sigma < 1e-15:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mu) / sigma


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
    # Strip common suffixes
    for suffix in ["_trimmed_VIEW_ACTION_SAFE_IVTC", "_trimmed_VIEW_ACTION_SAFE_NOIVTC",
                   "_svideo_direct", "_svideo_sdi1", "_svideo_sdi"]:
        name = name.replace(suffix, "")
    # Strip leading "twister_"
    name = name.replace("twister_", "")
    # For bakeoff names with spaces (e.g. "ag1980tbc0 machhd 02"), take first token
    # But for original names with underscores, keep the full name
    if " " in name:
        name = name.split()[0]
    return name


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
        # Skip initial frames if requested
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
        # Use median for naturalness (kurtosis is 4th-power, extremely outlier-sensitive)
        avg = float(np.median(arr)) if k == "naturalness" else float(np.mean(arr))
        result[k] = {"mean": avg, "std": float(np.std(arr))}

    if "temporal_stability" in metric_keys:
        if temporal_diffs:
            td = np.array(temporal_diffs)
            result["temporal_stability"] = {"mean": float(np.mean(td)), "std": float(np.std(td))}
        else:
            result["temporal_stability"] = {"mean": 0.0, "std": 0.0}

    # Suppress floating-point noise on ratio metrics (tiny means from mostly-zero per-frame values)
    EPS = 1e-5
    for k in PCT_KEYS:
        if k not in result:
            continue
        if result[k]["mean"] < EPS:
            result[k]["mean"] = 0.0

    result["n_frames"] = n_frames

    # Per-frame arrays for selected metrics (used for comparison screenshots)
    perframe = {k: np.array(accum[k]) for k in accum}
    if "temporal_stability" in metric_keys:
        if temporal_diffs:
            perframe["temporal_stability"] = np.array(temporal_diffs)
        else:
            perframe["temporal_stability"] = np.array([0.0])

    return result, perframe


def extract_frame_jpeg(filepath, frame_num):
    """Extract a single frame as JPEG bytes using ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", filepath,
         "-vf", f"select=eq(n\\,{frame_num})", "-vframes", "1",
         "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "2", "pipe:1"],
        capture_output=True
    )
    return proc.stdout if proc.returncode == 0 else b""


def find_comparison_frames(clip_paths, clip_names, all_perframe, all_results, skip_offsets, metric_keys):
    """Find a representative frame for each metric and extract screenshots.

    For each metric M:
      a) Find the clip V whose overall M score is best.
      b) From V, find the frame where M is near its 90th percentile — a frame
         where this metric is clearly demonstrated (not just average).
      c) Extract that frame from ALL clips for side-by-side comparison.

    skip_offsets: dict {clip_name: int} — frames skipped at start (added to ffmpeg frame number).
    metric_keys: ordered list of metrics to include in comparison galleries.

    Returns dict: {metric_key: {"frame_num": int, "best_clip": str,
                                 "frames": {clip_name: {"jpeg_b64": str, "value": float}}}}
    """
    comparisons = {}
    for key in metric_keys:
        _, _, higher_better = METRIC_INFO[key]

        # (a) Find the clip with the best overall score for this metric
        best_clip = None
        best_mean = None
        for name in clip_names:
            val = all_results[name][key]["mean"]
            if best_mean is None:
                best_clip, best_mean = name, val
            elif higher_better is True and val > best_mean:
                best_clip, best_mean = name, val
            elif higher_better is False and val < best_mean:
                best_clip, best_mean = name, val
            elif higher_better is None and abs(val - 1.0) < abs(best_mean - 1.0):
                best_clip, best_mean = name, val

        # (b) From the best clip, find the frame closest to p90 of the raw value.
        # This picks frames where the metric is clearly demonstrated, not just average.
        # For "lower is better" metrics (noise, etc.) p90 shows a challenging frame
        # from the best clip — more revealing for quality comparison.
        arr = all_perframe[best_clip][key]
        target_val = float(np.percentile(arr, 90))
        frame_idx = int(np.argmin(np.abs(arr - target_val)))

        # (c) Extract that frame from all clips
        frames = {}
        for name in clip_names:
            path = clip_paths[name]
            # Add back the skip offset so ffmpeg extracts the correct source frame
            actual_frame = frame_idx + skip_offsets.get(name, 0)
            jpeg_data = extract_frame_jpeg(path, actual_frame)
            pf_arr = all_perframe[name][key]
            frame_val = float(pf_arr[frame_idx]) if frame_idx < len(pf_arr) else 0.0
            frames[name] = {
                "jpeg_b64": base64.b64encode(jpeg_data).decode("ascii") if jpeg_data else "",
                "value": frame_val,
            }
        comparisons[key] = {
            "frame_num": frame_idx,
            "best_clip": best_clip,
            "frames": frames,
        }
        print(f"  {METRIC_INFO[key][0]}: frame {frame_idx} (p90 in {best_clip})")

    return comparisons


def extract_sample_screenshots(clip_paths, clip_names, n_screenshots, skip_offsets):
    """Extract N evenly-spaced screenshots from each clip.

    skip_offsets: dict {clip_name: int} — frames skipped at start.
    Returns dict: {clip_name: [{"frame_num": int, "jpeg_b64": str}, ...]}
    """
    if n_screenshots <= 0:
        return {}

    samples = {}
    for name in clip_names:
        path = clip_paths[name]
        skip = skip_offsets.get(name, 0)
        # Get frame count via ffprobe
        cmd = ["ffprobe", "-v", "error", "-count_frames",
               "-select_streams", "v:0",
               "-show_entries", "stream=nb_read_frames",
               "-of", "csv=p=0", path]
        try:
            total = int(subprocess.check_output(cmd, text=True).strip())
        except (ValueError, subprocess.CalledProcessError):
            total = 500  # fallback

        usable = total - skip
        indices = [int((i + 1) * usable / (n_screenshots + 1)) for i in range(n_screenshots)]
        clip_samples = []
        for fi in indices:
            actual_frame = fi + skip
            jpeg_data = extract_frame_jpeg(path, actual_frame)
            clip_samples.append({
                "frame_num": fi,
                "jpeg_b64": base64.b64encode(jpeg_data).decode("ascii") if jpeg_data else "",
            })
        samples[name] = clip_samples
        print(f"  {name}: {len(clip_samples)} screenshots")

    return samples


# =====================================================================
# TEXT REPORT
# =====================================================================

def format_text_report(all_results, src_dir, metric_keys):
    """Format results into a readable text report with auto-sized columns."""
    clip_names = sorted(all_results.keys())
    col_w = max(len(n) for n in clip_names) + 2

    lines = []
    lines.append("=" * 120)
    lines.append("NO-REFERENCE VIDEO QUALITY METRICS REPORT")
    lines.append("=" * 120)
    lines.append(f"Source: {src_dir}")
    lines.append("All metrics are brightness-agnostic (no upstream normalization required).")
    lines.append("")
    lines.append(f"--- Quality Metrics ({len(metric_keys)}) ---")

    headers = [short_metric_header(k) for k in metric_keys]
    header = f"{'Clip':<{col_w}}" + "".join(f" {h:>10}" for h in headers)
    lines.append(header)
    lines.append("-" * len(header))
    for name in clip_names:
        row = f"{name:<{col_w}}"
        for k in metric_keys:
            v = all_results[name][k]["mean"]
            row += f" {fmt_metric_value(k, v):>10}"
        lines.append(row)
    lines.append("")

    # Rankings
    lines.append("=" * 120)
    lines.append("RANKINGS (best to worst per metric)")
    lines.append("=" * 120)

    for key in metric_keys:
        label, _, higher_better = METRIC_INFO[key]
        if higher_better is True:
            dir_str = "(higher = better)"
        elif higher_better is False:
            dir_str = "(lower = better)"
        else:
            dir_str = "(closer to 1.0 = better)"
        lines.append(f"\n  {label} {dir_str}:")

        items = [(n, all_results[n][key]["mean"]) for n in clip_names]
        if higher_better is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif higher_better:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])

        for rank, (name, val) in enumerate(items, 1):
            lines.append(f"    {rank:>2}. {name:<{col_w}} {fmt_metric_value(key, val)}")

    # Composite rankings
    composites = compute_composites(all_results, metric_keys=metric_keys)

    lines.append("")
    lines.append("=" * 120)
    lines.append("OVERALL COMPOSITE RANKING")
    lines.append("=" * 120)
    lines.append("")
    ranked = sorted(clip_names, key=lambda c: composites[c]["overall"])
    for rank, name in enumerate(ranked, 1):
        lines.append(f"  {rank:>2}. {name:<{col_w}} avg rank: {composites[name]['overall']:.1f}")

    lines.append("")
    lines.append("=" * 120)
    return "\n".join(lines)


# =====================================================================
# HTML REPORT
# =====================================================================

HTML_CSS = """
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
    --good: #3fb950; --bad: #f85149; --mid: #d29922;
    --accent2: #d2a8ff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; padding: 24px; max-width: 1400px; margin: 0 auto;
  }
  h1 { font-size: 1.8em; margin-bottom: 8px; }
  h2 { font-size: 1.3em; margin: 40px 0 16px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  h2.alt { color: var(--accent2); }
  h3 { font-size: 1.1em; margin: 20px 0 12px; color: var(--text-dim); }
  .subtitle { color: var(--text-dim); margin-bottom: 24px; font-size: 0.95em; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; }
  .chart-container { position: relative; width: 100%; }
  .chart-wide { height: 520px; }
  .chart-radar { height: 600px; max-width: 900px; margin: 0 auto; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }
  .metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .metric-card canvas { height: 300px !important; }
  .heatmap { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  .heatmap th { background: #21262d; padding: 6px 5px; text-align: center; font-weight: 600;
    border: 1px solid var(--border); white-space: nowrap; position: sticky; top: 0; z-index: 2;
    cursor: pointer; user-select: none; }
  .heatmap th:hover { background: #2d333b; }
  .heatmap th::after { content: ' \\2195'; opacity: 0.3; font-size: 0.8em; }
  .heatmap th.sort-asc::after { content: ' \\2191'; opacity: 0.8; }
  .heatmap th.sort-desc::after { content: ' \\2193'; opacity: 0.8; }
  .heatmap th:first-child { text-align: left; min-width: 140px; }
  .heatmap th { border-bottom: 2px solid var(--accent); }
  .heatmap td { padding: 5px 6px; text-align: center; border: 1px solid var(--border);
    font-variant-numeric: tabular-nums; white-space: nowrap; font-size: 0.92em; }
  .heatmap td:first-child { text-align: left; font-weight: 500; }
  .heatmap tr:hover { outline: 2px solid var(--accent); }
  .rank { color: var(--text-dim); font-size: 0.85em; margin-right: 4px; }
  .legend-note { font-size: 0.85em; color: var(--text-dim); margin-top: 12px; }
  .metric-guide { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px 24px; }
  .metric-guide div { font-size: 0.88em; padding: 4px 0; }
  .metric-guide .mg-label { font-weight: 600; }
  .dir { font-size: 0.78em; padding: 2px 6px; border-radius: 4px; margin-left: 6px; }
  .dir-up { background: #1a3a2a; color: var(--good); }
  .dir-down { background: #3a1a1a; color: var(--bad); }
  .dir-mid { background: #3a2e1a; color: var(--mid); }
  /* Lightbox — CSS-only click-to-enlarge */
  .lb-toggle { display: none; }
  .lb-thumb { cursor: zoom-in; display: block; }
  .lb-thumb img { transition: opacity 0.15s; }
  .lb-thumb:hover img { opacity: 0.85; }
  .lb-overlay { display: none; position: fixed; inset: 0;
    background: #000; z-index: 9999; cursor: zoom-out;
    justify-content: center; align-items: center; flex-direction: column; padding: 20px; }
  .lb-overlay img { max-width: 95vw; max-height: 85vh; object-fit: contain; border-radius: 4px;
    transform-origin: 0 0; user-select: none; -webkit-user-drag: none; }
  .lb-overlay .lb-caption { color: #e6edf3; font-size: 0.9em; margin-top: 12px;
    text-align: center; max-width: 90vw; }
  .lb-toggle:checked + label.lb-overlay { display: flex; }
  .lb-hint { color: #555; font-size: 0.75em; margin-top: 6px; }
  /* A/B Comparison Slider */
  .cmp-card { position: relative; }
  .cmp-btn {
    position: absolute; top: 8px; right: 8px; width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.6); color: #ccc; border-radius: 6px;
    cursor: pointer; z-index: 10; user-select: none; line-height: 0;
    transition: background 0.15s, color 0.15s;
  }
  .cmp-btn:hover { background: rgba(30,90,200,0.8); color: #fff; }
  .cmp-card.cmp-selected { outline: 2px solid var(--accent); }
  .cmp-card.cmp-selected .cmp-btn { background: var(--accent); color: #fff; }
  .cmp-toast {
    display: none; position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #1f6feb; color: #fff; padding: 10px 24px; border-radius: 8px;
    font-size: 0.95em; z-index: 10000; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    pointer-events: none;
  }
  .ab-overlay {
    display: none; position: fixed; inset: 0; background: #000; z-index: 9999;
    flex-direction: column; align-items: center; justify-content: center;
  }
  .ab-viewport {
    position: relative; max-width: 95vw; max-height: 85vh;
    overflow: hidden; cursor: col-resize; touch-action: none;
    transform-origin: 0 0;
  }
  .ab-viewport img { display: block; max-width: 95vw; max-height: 85vh; object-fit: contain;
    user-select: none; -webkit-user-drag: none; }
  .ab-img-b {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    overflow: hidden; clip-path: inset(0 0 0 50%);
  }
  .ab-img-b img { display: block; max-width: 95vw; max-height: 85vh; object-fit: contain; }
  .ab-divider {
    position: absolute; top: 0; left: 50%; width: 3px; height: 100%;
    background: #fff; pointer-events: none; z-index: 2;
  }
  .ab-divider::after {
    content: ''; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%,-50%); width: 28px; height: 28px;
    border: 2px solid #fff; border-radius: 50%; background: rgba(0,0,0,0.5);
  }
  .ab-labels {
    display: flex; gap: 20px; margin-bottom: 8px; font-size: 0.85em;
    color: #aaa; z-index: 2;
  }
  .ab-labels span { padding: 2px 10px; border-radius: 4px; background: rgba(255,255,255,0.1); }
  .ab-close-hint { margin-top: 10px; font-size: 0.8em; color: #666; z-index: 2; }
"""


def generate_html(data, metric_keys, title="Video Quality Report", comparisons=None,
                  samples=None, json_filename=None):
    """Generate a self-contained HTML quality report with Chart.js visualizations."""
    clip_names = sorted(data.keys())
    n = len(clip_names)
    m = len(metric_keys)
    palette = COLORS_7 if n <= 7 else COLORS_14
    colors = {c: palette[i % len(palette)] for i, c in enumerate(clip_names)}

    composites = compute_composites(data, metric_keys=metric_keys)
    ranked_overall = sorted(clip_names, key=lambda c: composites[c]["overall"])

    # Radar: normalize each metric to 0-100
    radar_scores = {}
    for c in clip_names:
        sc = []
        for key in metric_keys:
            vals = [data[d][key]["mean"] for d in clip_names]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx - mn > 1e-15 else 1e-15
            raw = data[c][key]["mean"]
            _, _, hb = METRIC_INFO[key]
            if hb is True:
                sc.append(((raw - mn) / rng) * 100)
            elif hb is False:
                sc.append((1 - (raw - mn) / rng) * 100)
            else:
                max_dist = max(abs(v - 1.0) for v in vals) or 1e-15
                sc.append((1 - abs(raw - 1.0) / max_dist) * 100)
        radar_scores[c] = sc
    radar_labels = [METRIC_INFO[k][0] for k in metric_keys]

    # Heatmap data
    hm_data = []
    for c in ranked_overall:
        cells = [{"raw": data[c][k]["mean"],
                  "raw_fmt": fmt_metric_value(k, data[c][k]["mean"]),
                  "z": composites[c]["zscores"][k], "key": k}
                 for k in metric_keys]
        hm_data.append({"name": c, "overall": composites[c]["overall"], "cells": cells})

    # Per-metric rankings
    per_metric = {}
    for key in metric_keys:
        _, _, hb = METRIC_INFO[key]
        items = [(c, data[c][key]["mean"]) for c in clip_names]
        if hb is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif hb:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])
        is_pct = key in PCT_KEYS
        per_metric[key] = {
            "labels": [c for c, _ in items],
            "values": [v * 100 for _, v in items] if is_pct else [v for _, v in items],
            "colors": [colors[c] for c, _ in items],
            "label": METRIC_INFO[key][0], "unit": METRIC_INFO[key][1],
            "higher_better": hb, "is_pct": is_pct,
        }

    # Radar datasets
    radar_datasets = [{"label": c, "data": radar_scores[c], "borderColor": colors[c],
                       "backgroundColor": colors[c] + "20", "borderWidth": 2, "pointRadius": 3,
                       "hidden": i >= 5} for i, c in enumerate(clip_names)]

    # Chart height scales with clip count
    bar_height = max(320, n * 36 + 80)
    metric_guide_html = []
    for key in metric_keys:
        label, _, hb = METRIC_INFO[key]
        if hb is True:
            dir_class, dir_text = "dir-up", "higher"
        elif hb is False:
            dir_class, dir_text = "dir-down", "lower"
        else:
            dir_class, dir_text = "dir-mid", "~1.0"
        desc = METRIC_GUIDE_TEXT.get(key, "")
        metric_guide_html.append(
            f'<div><span class="mg-label">{label}</span> <span class="dir {dir_class}">{dir_text}</span>'
            f"<br>{desc}</div>"
        )
    heatmap_headers = "".join(f"<th>{short_metric_header(k)}</th>" for k in metric_keys)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>{HTML_CSS}</style>
</head>
<body>

<h1>{title}</h1>
<p class="subtitle">
  {n} analog video captures evaluated on {m} no-reference quality metrics
  (all brightness-agnostic — no upstream normalization required).
</p>

<div class="card">
  <h3>Metric Guide</h3>
  <div class="metric-guide">
    {"".join(metric_guide_html)}
  </div>
</div>

<h2>Overall Composite Ranking</h2>
<div class="card">
  <div class="chart-container" style="height:{bar_height}px;"><canvas id="overallChart"></canvas></div>
  <p class="legend-note">Average rank across all {m} metrics (1 = best per metric). Lower average rank = better overall quality.</p>
</div>

<h2>{m}-Metric Radar Comparison</h2>
<div class="card">
  <div class="chart-container chart-radar"><canvas id="radarChart"></canvas></div>
  <p class="legend-note">All metrics on 0–100 quality scale (higher = better on all axes). Click legend to toggle.{" Top 5 shown by default." if n > 5 else ""}</p>
</div>

<h2>Detailed Heatmap — All Metrics</h2>
<div class="card" style="overflow-x:auto;">
  <table class="heatmap" id="heatmapTable">
    <thead><tr>
      <th>Clip</th><th>Avg Rank</th>
      {heatmap_headers}
    </tr></thead>
    <tbody></tbody>
  </table>
  <p class="legend-note">Ranked by average rank across all {m} metrics (lower = better). Cell color: green = good, red = poor. Click any column header to sort.</p>
</div>

<h2>Individual Metric Rankings</h2>
<div class="metric-grid" id="metricGrid"></div>

<script>
new Chart(document.getElementById('overallChart'), {{
  type:'bar',
  data:{{ labels:{json.dumps([c for c in ranked_overall])},
    datasets:[{{ data:{json.dumps([round(n + 1 - composites[c]["overall"], 2) for c in ranked_overall])},
      backgroundColor:{json.dumps([colors[c] for c in ranked_overall])},
      borderColor:{json.dumps([colors[c] for c in ranked_overall])}, borderWidth:1, borderRadius:4 }}] }},
  options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}},
      tooltip:{{callbacks:{{label:c=>{{const rank={json.dumps({c: round(composites[c]["overall"], 1) for c in ranked_overall})};return 'avg rank: '+rank[c.label]}}}}}} }},
    scales:{{ x:{{min:0,max:{n},grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}},title:{{display:true,text:'Rank Score (higher = better)',color:'#8b949e'}}}}, y:{{grid:{{display:false}},ticks:{{color:'#e6edf3',font:{{size:11}}}}}} }}
  }}
}});

new Chart(document.getElementById('radarChart'),{{
  type:'radar',
  data:{{labels:{json.dumps(radar_labels)},datasets:{json.dumps(radar_datasets)}}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{position:'right',labels:{{color:'#e6edf3',font:{{size:11}},boxWidth:12,padding:6}}}}}},
    scales:{{r:{{min:0,max:100,grid:{{color:'#30363d'}},angleLines:{{color:'#30363d'}},pointLabels:{{color:'#e6edf3',font:{{size:11}}}},ticks:{{display:false}}}}}}
  }}
}});

function escHtml(s){{var d=document.createElement('div');d.textContent=s;return d.innerHTML;}}
const hmData={json.dumps(hm_data)};
const tbody=document.querySelector('#heatmapTable tbody');
hmData.forEach((row,idx)=>{{
  const tr=document.createElement('tr');
  let td=document.createElement('td');
  td.innerHTML='<span class="rank">#'+(idx+1)+'</span> '+escHtml(row.name);
  tr.appendChild(td);
  td=document.createElement('td');td.textContent=row.overall.toFixed(1);
  const mid=({n}+1)/2, ov=Math.max(-1,Math.min(1,(mid-row.overall)/mid));
  td.style.background=ov>0?'rgba(63,185,80,'+(Math.abs(ov)*0.4)+')':'rgba(248,81,73,'+(Math.abs(ov)*0.4)+')';
  td.style.fontWeight='600';tr.appendChild(td);
  row.cells.forEach(cell=>{{
    td=document.createElement('td');
    td.textContent=cell.raw_fmt;
    const z=cell.z,n=Math.max(-1,Math.min(1,z/2.5));
    td.style.background=n>0?'rgba(63,185,80,'+(Math.abs(n)*0.35)+')':'rgba(248,81,73,'+(Math.abs(n)*0.35)+')';
    tr.appendChild(td);
  }});
  tbody.appendChild(tr);
}});

const amd={json.dumps(per_metric)};
const grid=document.getElementById('metricGrid');
Object.entries(amd).forEach(([key,md])=>{{
  const card=document.createElement('div');
  card.className='metric-card';
  const canvas=document.createElement('canvas');card.appendChild(canvas);grid.appendChild(card);
  const dir=md.higher_better===true?'(higher = better)':md.higher_better===false?'(lower = better)':'(closer to 1.0)';
  new Chart(canvas,{{
    type:'bar',data:{{labels:md.labels,datasets:[{{data:md.values,backgroundColor:md.colors.map(c=>c+'cc'),borderColor:md.colors,borderWidth:1,borderRadius:3}}]}},
    options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},title:{{display:true,text:md.label+' '+dir,color:'#e6edf3',font:{{size:14}}}},tooltip:{{callbacks:{{label:c=>md.is_pct?c.parsed.x.toFixed(1)+'%':md.unit+': '+c.parsed.x.toFixed(6)}}}}}},
      scales:{{x:{{grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}}}},y:{{grid:{{display:false}},ticks:{{color:'#e6edf3',font:{{size:10}}}}}}}}
    }}
  }});
}});

document.querySelectorAll('.heatmap th').forEach((th,colIdx)=>{{
  th.addEventListener('click',()=>{{
    const table=th.closest('table'),tbody=table.querySelector('tbody');
    const rows=Array.from(tbody.querySelectorAll('tr'));
    const asc=!th.classList.contains('sort-asc');
    table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(asc?'sort-asc':'sort-desc');
    rows.sort((a,b)=>{{
      const av=a.cells[colIdx].textContent.replace(/^#\\d+\\s*/,'');
      const bv=b.cells[colIdx].textContent.replace(/^#\\d+\\s*/,'');
      const an=parseFloat(av),bn=parseFloat(bv);
      if(!isNaN(an)&&!isNaN(bn)) return asc?an-bn:bn-an;
      return asc?av.localeCompare(bv):bv.localeCompare(av);
    }});
    rows.forEach((r,i)=>{{
      const fc=r.cells[0],name=fc.textContent.replace(/^#\\d+\\s*/,'');
      fc.innerHTML='<span class="rank">#'+(i+1)+'</span> '+escHtml(name);
      tbody.appendChild(r);
    }});
  }});
}});
</script>
"""

    # Collect lightbox overlays to emit at document root level (avoids stacking context clipping)
    lb_id = 0
    lb_overlays = []  # list of (id, img_src, caption) tuples

    # Append metric comparison frames if available
    if comparisons:
        html += """<h2>Visual Metric Comparisons</h2>
<p class="subtitle">For each metric, a strong example frame from the best-scoring clip (near its 90th percentile)
is shown. All clips are shown at the same frame number for direct comparison.
Click any image to enlarge; use the <span style="display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;background:rgba(0,0,0,0.55);border-radius:4px;vertical-align:middle;color:#ccc;">""" + CMP_ICON_SVG.replace('width="20" height="20"', 'width="16" height="16"') + """</span> icon to A/B compare two images with a slider.</p>
"""
        for key in metric_keys:
            if key not in comparisons:
                continue
            comp = comparisons[key]
            label, unit, higher_better = METRIC_INFO[key]
            if higher_better is True:
                direction = "(higher = better)"
            elif higher_better is False:
                direction = "(lower = better)"
            else:
                direction = "(closer to 1.0 = better)"
            html += f"""<h3>{label} {direction} — Frame {comp["frame_num"]} (p90 in {comp["best_clip"]})</h3>
<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:24px;">
"""
            # Sort by metric value (best first)
            if higher_better is True:
                frame_items = sorted(comp["frames"].items(), key=lambda x: -x[1]["value"])
            elif higher_better is False:
                frame_items = sorted(comp["frames"].items(), key=lambda x: x[1]["value"])
            else:
                frame_items = sorted(comp["frames"].items(), key=lambda x: abs(x[1]["value"] - 1.0))
            is_pct = key in ("crushed_blacks", "blown_whites")
            for rank, (name, fdata) in enumerate(frame_items, 1):
                if fdata["jpeg_b64"]:
                    img_src = f"data:image/jpeg;base64,{fdata['jpeg_b64']}"
                    vfmt = f"{fdata['value']*100:.1f}%" if is_pct else f"{fdata['value']:.6f}"
                    caption = f"#{rank} {name} &mdash; {label}: {vfmt} &mdash; Frame {comp['frame_num']}"
                    lb_overlays.append((lb_id, img_src, caption))
                    html += f"""  <div class="card cmp-card" style="flex:1;min-width:300px;max-width:48%;">
    <div class="cmp-btn" data-cmp-caption="#{rank} {name} &mdash; {label}: {vfmt}">{CMP_ICON_SVG}</div>
    <div style="font-weight:600;margin-bottom:4px;"><span class="rank">#{rank}</span> {name}</div>
    <div style="font-size:0.9em;color:var(--text-dim);margin-bottom:8px;">{label}: {vfmt}</div>
    <label for="lb{lb_id}" class="lb-thumb"><img src="{img_src}" style="width:100%;border-radius:4px;"></label>
  </div>
"""
                    lb_id += 1
            html += "</div>\n"

    # Append sample screenshots if available
    if samples:
        html += """<h2>Appendix: Sample Frames</h2>
<p class="subtitle">Evenly-spaced frames from each clip for visual reference and decode verification. Click any image to enlarge.</p>
"""
        for name in sorted(samples.keys()):
            frames = samples[name]
            html += f'<h3>{name}</h3>\n<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">\n'
            for sf in frames:
                if sf["jpeg_b64"]:
                    img_src = f"data:image/jpeg;base64,{sf['jpeg_b64']}"
                    caption = f"{name} &mdash; Frame {sf['frame_num']}"
                    lb_overlays.append((lb_id, img_src, caption))
                    html += f"""  <div style="flex:1;min-width:200px;max-width:32%;">
    <div style="font-size:0.8em;color:var(--text-dim);margin-bottom:4px;">Frame {sf["frame_num"]}</div>
    <label for="lb{lb_id}" class="lb-thumb"><img src="{img_src}" style="width:100%;border-radius:4px;"></label>
  </div>
"""
                    lb_id += 1
            html += "</div>\n"

    # Emit lightbox overlays at document root level (outside all containers)
    if lb_overlays:
        html += '\n<!-- Lightbox overlays (document root to avoid stacking context clipping) -->\n'
        for lid, img_src, caption in lb_overlays:
            html += f'<input type="checkbox" id="lb{lid}" class="lb-toggle">'
            html += f'<label for="lb{lid}" class="lb-overlay"><img src="{img_src}"><div class="lb-caption">{caption}</div><div class="lb-hint">Scroll to zoom &middot; Drag to pan &middot; R to reset &middot; Click to close</div></label>\n'

    # Lightbox zoom/pan script (works for all lightbox images)
    if lb_overlays:
        html += """
<script>
(function(){
  var scale=1,tx=0,ty=0,dragging=false,lastX=0,lastY=0;

  function getActive(){
    var chk=document.querySelector('.lb-toggle:checked');
    if(!chk) return null;
    return chk.nextElementSibling;
  }
  function getImg(){
    var ov=getActive();
    return ov?ov.querySelector('img'):null;
  }
  function apply(img){
    img.style.transform='translate('+tx+'px,'+ty+'px) scale('+scale+')';
    img.style.cursor=scale>1?'grab':'zoom-in';
  }
  function resetZoom(){
    scale=1;tx=0;ty=0;
    var img=getImg();
    if(img){img.style.transform='';img.style.cursor='zoom-in';}
  }

  /* Wheel zoom — zoom toward cursor */
  document.addEventListener('wheel',function(e){
    var img=getImg();
    if(!img) return;
    e.preventDefault();
    var rect=img.getBoundingClientRect();
    var oldS=scale;
    scale=Math.max(1,Math.min(15,scale*(e.deltaY<0?1.15:1/1.15)));
    if(scale===1){resetZoom();return;}
    var ratio=scale/oldS;
    tx=(1-ratio)*(e.clientX-rect.left)+tx;
    ty=(1-ratio)*(e.clientY-rect.top)+ty;
    apply(img);
  },{passive:false});

  /* Drag to pan */
  document.addEventListener('mousedown',function(e){
    var img=getImg();
    if(!img||scale<=1||e.target!==img) return;
    dragging=true;lastX=e.clientX;lastY=e.clientY;
    img.style.cursor='grabbing';
    e.preventDefault();
  });
  document.addEventListener('mousemove',function(e){
    if(!dragging) return;
    var img=getImg();
    if(!img) return;
    tx+=e.clientX-lastX;ty+=e.clientY-lastY;
    lastX=e.clientX;lastY=e.clientY;
    apply(img);
  });
  document.addEventListener('mouseup',function(){
    if(dragging){
      dragging=false;
      var img=getImg();
      if(img&&scale>1) img.style.cursor='grab';
    }
  });

  /* Prevent close when zoomed */
  document.querySelectorAll('.lb-overlay').forEach(function(ov){
    ov.addEventListener('click',function(e){
      if(scale>1){e.preventDefault();e.stopPropagation();}
    });
  });

  /* Prevent native image drag */
  document.querySelectorAll('.lb-overlay img').forEach(function(img){
    img.addEventListener('dragstart',function(e){e.preventDefault();});
  });

  /* Keyboard: r=reset, Escape=close+reset */
  document.addEventListener('keydown',function(e){
    if((e.key==='r'||e.key==='R')&&getImg()){e.preventDefault();resetZoom();}
    if(e.key==='Escape') resetZoom();
  });

  /* Reset zoom when lightbox closes */
  document.querySelectorAll('.lb-toggle').forEach(function(chk){
    chk.addEventListener('change',function(){if(!chk.checked) resetZoom();});
  });

  /* Touch: pinch-to-zoom + drag */
  var lastDist=0,lastMid=null;
  document.addEventListener('touchstart',function(e){
    var img=getImg();if(!img) return;
    if(e.touches.length===2){
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      lastDist=Math.sqrt(dx*dx+dy*dy);
      lastMid={x:(e.touches[0].clientX+e.touches[1].clientX)/2,
               y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      e.preventDefault();
    }else if(e.touches.length===1&&scale>1){
      dragging=true;lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;
      e.preventDefault();
    }
  },{passive:false});
  document.addEventListener('touchmove',function(e){
    var img=getImg();if(!img) return;
    if(e.touches.length===2&&lastDist>0){
      e.preventDefault();
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      var dist=Math.sqrt(dx*dx+dy*dy);
      var mid={x:(e.touches[0].clientX+e.touches[1].clientX)/2,
               y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      var oldS=scale;
      scale=Math.max(1,Math.min(15,scale*(dist/lastDist)));
      if(scale===1){resetZoom();lastDist=dist;return;}
      var rect=img.getBoundingClientRect();
      var ratio=scale/oldS;
      tx=(1-ratio)*(mid.x-rect.left)+tx;
      ty=(1-ratio)*(mid.y-rect.top)+ty;
      tx+=mid.x-lastMid.x;ty+=mid.y-lastMid.y;
      lastDist=dist;lastMid=mid;
      apply(img);
    }else if(e.touches.length===1&&dragging&&scale>1){
      e.preventDefault();
      tx+=e.touches[0].clientX-lastX;ty+=e.touches[0].clientY-lastY;
      lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;
      apply(img);
    }
  },{passive:false});
  document.addEventListener('touchend',function(e){
    if(e.touches.length<2) lastDist=0;
    if(e.touches.length===0) dragging=false;
  });
})();
</script>
"""

    # A/B comparison slider (only when comparison frames exist)
    if comparisons:
        html += """
<!-- A/B Comparison Slider -->
<div class="cmp-toast" id="cmpToast">Select another image to compare</div>
<div class="ab-overlay" id="abOverlay">
  <div class="ab-labels"><span id="abLabelA">A</span><span id="abLabelB">B</span></div>
  <div class="ab-viewport" id="abViewport">
    <img id="abImgA" src="" alt="A">
    <div class="ab-img-b" id="abClipB"><img id="abImgB" src="" alt="B"></div>
    <div class="ab-divider" id="abDivider"></div>
  </div>
  <div class="ab-close-hint">Scroll to zoom &middot; Drag to pan &middot; R to reset &middot; Escape to close</div>
</div>
<script>
(function(){
  var selectedA = null;
  var toast = document.getElementById('cmpToast');
  var overlay = document.getElementById('abOverlay');
  var viewport = document.getElementById('abViewport');
  var imgA = document.getElementById('abImgA');
  var imgB = document.getElementById('abImgB');
  var clipB = document.getElementById('abClipB');
  var divider = document.getElementById('abDivider');
  var labelA = document.getElementById('abLabelA');
  var labelB = document.getElementById('abLabelB');

  /* Zoom state for A/B overlay */
  var zs=1, ztx=0, zty=0, zDrag=false, zLastX=0, zLastY=0, zDidDrag=false;

  function clearSelection() {
    if (selectedA) selectedA.el.classList.remove('cmp-selected');
    selectedA = null;
    toast.style.display = 'none';
  }

  function getSrc(btn) {
    return btn.closest('.cmp-card').querySelector('img').src;
  }

  function updateSlider(pct) {
    clipB.style.clipPath = 'inset(0 0 0 ' + pct + '%)';
    divider.style.left = pct + '%';
  }

  function applyZoom() {
    viewport.style.transform = 'translate('+ztx+'px,'+zty+'px) scale('+zs+')';
    viewport.style.cursor = zs > 1 ? 'grab' : 'col-resize';
  }

  function resetABZoom() {
    zs=1; ztx=0; zty=0;
    viewport.style.transform = '';
    viewport.style.cursor = 'col-resize';
  }

  function openAB(srcA, captA, srcB, captB) {
    imgA.src = srcA; imgB.src = srcB;
    labelA.textContent = 'A: ' + captA;
    labelB.textContent = 'B: ' + captB;
    updateSlider(50);
    resetABZoom();
    overlay.style.display = 'flex';
    clearSelection();
  }

  function closeAB() { resetABZoom(); overlay.style.display = 'none'; }

  document.querySelectorAll('.cmp-btn').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      var src = getSrc(btn);
      var caption = btn.getAttribute('data-cmp-caption');
      if (!selectedA) {
        selectedA = { el: btn.closest('.cmp-card'), src: src, caption: caption };
        selectedA.el.classList.add('cmp-selected');
        toast.style.display = 'block';
      } else if (selectedA.el === btn.closest('.cmp-card')) {
        clearSelection();
      } else {
        openAB(selectedA.src, selectedA.caption, src, caption);
      }
    });
  });

  /* Wheel zoom on A/B viewport */
  overlay.addEventListener('wheel', function(e) {
    if (overlay.style.display !== 'flex') return;
    e.preventDefault();
    var rect = viewport.getBoundingClientRect();
    var oldS = zs;
    zs = Math.max(1, Math.min(15, zs * (e.deltaY < 0 ? 1.15 : 1/1.15)));
    if (zs === 1) { resetABZoom(); return; }
    var ratio = zs / oldS;
    ztx = (1-ratio) * (e.clientX - rect.left) + ztx;
    zty = (1-ratio) * (e.clientY - rect.top) + zty;
    applyZoom();
  }, {passive: false});

  /* Mouse drag to pan when zoomed, slider when not */
  viewport.addEventListener('mousedown', function(e) {
    if (zs > 1) {
      zDrag = true; zDidDrag = false;
      zLastX = e.clientX; zLastY = e.clientY;
      viewport.style.cursor = 'grabbing';
      e.preventDefault();
    }
  });
  viewport.addEventListener('mousemove', function(e) {
    if (zDrag) {
      ztx += e.clientX - zLastX; zty += e.clientY - zLastY;
      zLastX = e.clientX; zLastY = e.clientY;
      zDidDrag = true;
      applyZoom();
      return;
    }
    /* Slider — works at any zoom level */
    var rect = viewport.getBoundingClientRect();
    var pct = ((e.clientX - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    updateSlider(pct);
  });
  document.addEventListener('mouseup', function() {
    if (zDrag) { zDrag = false; if (zs > 1) viewport.style.cursor = 'grab'; }
  });

  /* Touch: single-finger slider or pan; two-finger pinch zoom */
  var abTouchDist=0, abTouchMid=null;
  viewport.addEventListener('touchstart', function(e) {
    if (e.touches.length === 2) {
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      abTouchDist = Math.sqrt(dx*dx+dy*dy);
      abTouchMid = {x:(e.touches[0].clientX+e.touches[1].clientX)/2,
                    y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      e.preventDefault();
    } else if (e.touches.length === 1 && zs > 1) {
      zDrag = true; zDidDrag = false;
      zLastX = e.touches[0].clientX; zLastY = e.touches[0].clientY;
      e.preventDefault();
    }
  }, {passive: false});
  viewport.addEventListener('touchmove', function(e) {
    e.preventDefault();
    if (e.touches.length === 2 && abTouchDist > 0) {
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      var dist=Math.sqrt(dx*dx+dy*dy);
      var mid={x:(e.touches[0].clientX+e.touches[1].clientX)/2,
               y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      var oldS=zs;
      zs=Math.max(1,Math.min(15,zs*(dist/abTouchDist)));
      if(zs===1){resetABZoom();abTouchDist=dist;return;}
      var rect=viewport.getBoundingClientRect();
      var ratio=zs/oldS;
      ztx=(1-ratio)*(mid.x-rect.left)+ztx;
      zty=(1-ratio)*(mid.y-rect.top)+zty;
      ztx+=mid.x-abTouchMid.x; zty+=mid.y-abTouchMid.y;
      abTouchDist=dist; abTouchMid=mid;
      applyZoom();
    } else if (e.touches.length===1 && zDrag && zs>1) {
      ztx+=e.touches[0].clientX-zLastX; zty+=e.touches[0].clientY-zLastY;
      zLastX=e.touches[0].clientX; zLastY=e.touches[0].clientY;
      zDidDrag=true;
      applyZoom();
    } else if (e.touches.length===1 && zs<=1) {
      var rect=viewport.getBoundingClientRect();
      var pct=((e.touches[0].clientX-rect.left)/rect.width)*100;
      pct=Math.max(0,Math.min(100,pct));
      updateSlider(pct);
    }
  }, {passive: false});
  viewport.addEventListener('touchend', function(e) {
    if(e.touches.length<2) abTouchDist=0;
    if(e.touches.length===0){
      zDrag=false;
      if(!zDidDrag && zs<=1) closeAB();
      zDidDrag=false;
    }
  });

  /* Click: close only at 1x zoom and if not after a drag */
  viewport.addEventListener('click', function() {
    if (zDidDrag) { zDidDrag = false; return; }
    if (zs > 1) return;
    closeAB();
  });

  document.addEventListener('keydown', function(e) {
    if (overlay.style.display === 'flex') {
      if (e.key === 'r' || e.key === 'R') { e.preventDefault(); resetABZoom(); return; }
      if (e.key === 'Escape') { closeAB(); return; }
    }
    if (e.key === 'Escape') clearSelection();
  });
})();
</script>
"""

    if json_filename:
        html += f'\n<p style="text-align:center; color:#888; font-size:0.85em; margin-top:2em;">Data source: <code>{json_filename}</code></p>\n'
        html += f'<!-- data-source: {json_filename} -->\n'

    html += "</body>\n</html>"

    return html


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="No-reference video quality metrics for analog video captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("src_dir", help="Directory containing normalized .mov files")
    parser.add_argument("--output-dir", help="Output directory for reports (default: src_dir)")
    parser.add_argument("--name", default="quality_report", help="Report filename prefix (default: quality_report)")
    parser.add_argument("--pattern", default="*.mov", help="File glob pattern (default: *.mov)")
    parser.add_argument("--text", action="store_true", help="Also generate a plain-text report")
    parser.add_argument("--screenshots", type=int, default=0,
                        help="Number of sample screenshots per clip to embed in HTML (default: 0)")
    parser.add_argument("--skip", action="append", default=[],
                        help="Skip N initial frames for a clip: PATTERN:N (e.g. --skip jvctbc1:1). "
                             "PATTERN is matched as substring of clip filename. Can be repeated.")
    parser.add_argument("--extra-detail-metrics", default="",
                        help=("Comma-separated extra detail metrics to compute "
                              f"({', '.join(EXTRA_DETAIL_KEYS)})."))
    parser.add_argument("--report-metrics", default="",
                        help="Comma-separated metric keys to include in HTML/TXT report (default: all computed).")
    args = parser.parse_args()

    # Parse --skip arguments into {pattern: n_frames} dict
    skip_patterns = {}
    for s in args.skip:
        if ":" not in s:
            print(f"ERROR: --skip format must be PATTERN:N, got '{s}'")
            sys.exit(1)
        pat, n = s.rsplit(":", 1)
        try:
            skip_patterns[pat] = int(n)
        except ValueError:
            print(f"ERROR: --skip N must be integer, got '{n}'")
            sys.exit(1)

    extra_detail_metrics = parse_metric_csv(args.extra_detail_metrics)
    unknown_extra = [k for k in extra_detail_metrics if k not in EXTRA_DETAIL_KEYS]
    if unknown_extra:
        print(f"ERROR: Unknown --extra-detail-metrics: {', '.join(unknown_extra)}")
        print(f"       Allowed: {', '.join(EXTRA_DETAIL_KEYS)}")
        sys.exit(1)

    requested_metric_keys = list(ALL_KEYS)
    for key in extra_detail_metrics:
        if key not in requested_metric_keys:
            requested_metric_keys.append(key)

    analysis_metric_keys = list(requested_metric_keys)
    if DETAIL_PERCEPTUAL_KEY in requested_metric_keys:
        # Ensure dependencies are computed even if only the derived key was requested.
        for dep in DETAIL_PERCEPTUAL_DEPS:
            if dep not in analysis_metric_keys:
                analysis_metric_keys.append(dep)

    if args.report_metrics:
        report_metric_keys = parse_metric_csv(args.report_metrics)
        if not report_metric_keys:
            print("ERROR: --report-metrics is empty after parsing")
            sys.exit(1)
        unknown_report = [k for k in report_metric_keys if k not in METRIC_INFO]
        if unknown_report:
            print(f"ERROR: Unknown --report-metrics keys: {', '.join(unknown_report)}")
            sys.exit(1)
        missing_report = [k for k in report_metric_keys if k not in analysis_metric_keys]
        if missing_report:
            print(f"ERROR: --report-metrics keys not computed in this run: {', '.join(missing_report)}")
            print("       Add them via --extra-detail-metrics (or remove from --report-metrics).")
            sys.exit(1)
    else:
        report_metric_keys = list(requested_metric_keys)

    src_dir = args.src_dir.rstrip("/")
    output_dir = args.output_dir or src_dir
    os.makedirs(output_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(src_dir, args.pattern)))
    if not clips:
        print(f"ERROR: No files matching '{args.pattern}' found in {src_dir}")
        sys.exit(1)

    # Probe all clips and validate consistent resolution
    resolutions = {}
    for clip_path in clips:
        resolutions[clip_path] = probe_video(clip_path)
    res_set = set(resolutions.values())
    if len(res_set) > 1:
        print("ERROR: Not all clips have the same resolution:")
        for path, (w, h) in resolutions.items():
            print(f"  {w}x{h}: {os.path.basename(path)}")
        sys.exit(1)
    vid_w, vid_h = res_set.pop()

    print(f"Analyzing {len(clips)} clips in {src_dir} ({vid_w}x{vid_h})...")
    print(f"Computed metrics ({len(analysis_metric_keys)}): {', '.join(analysis_metric_keys)}")
    if report_metric_keys != analysis_metric_keys:
        print(f"Report metrics ({len(report_metric_keys)}): {', '.join(report_metric_keys)}")
    print("")

    all_results = {}
    all_perframe = {}
    clip_paths = {}  # name -> filepath mapping
    skip_offsets = {}  # name -> number of frames skipped
    for i, clip_path in enumerate(clips, 1):
        name = short_name(clip_path)
        clip_paths[name] = clip_path
        # Determine skip for this clip based on --skip patterns
        skip = 0
        basename = os.path.basename(clip_path)
        for pat, n in skip_patterns.items():
            if pat in basename or pat in name:
                skip = n
                break
        skip_offsets[name] = skip

        # Detect name collisions and append suffix if needed
        if name in all_results:
            orig_name = name
            suffix = 2
            while f"{name}_{suffix}" in all_results:
                suffix += 1
            name = f"{name}_{suffix}"
            print(f"  WARNING: Duplicate short name '{orig_name}' from {clip_path}, using '{name}'")
            skip_offsets[name] = skip_offsets.pop(orig_name, skip)

        skip_msg = f", skip {skip}" if skip > 0 else ""
        print(f"[{i:>2}/{len(clips)}] {name}{skip_msg}...", end=" ", flush=True)
        metrics, perframe = analyze_clip(clip_path, vid_w, vid_h, analysis_metric_keys, skip_frames=skip)

        if metrics is None:
            print(f"skipped (no frames decoded)")
            continue

        metrics["_source_file"] = os.path.basename(clip_path)
        all_results[name] = metrics
        all_perframe[name] = perframe
        print(f"done ({metrics['n_frames']} frames)")

    if DETAIL_PERCEPTUAL_KEY in analysis_metric_keys:
        print("\nComputing derived metric: detail_perceptual")
        if not add_detail_perceptual_metric(all_results, all_perframe):
            print("ERROR: failed to compute derived detail_perceptual metric")
            sys.exit(1)

    # JSON output (always) — timestamped filename for provenance
    run_timestamp = datetime.now()
    ts_suffix = run_timestamp.strftime("%Y%m%d_%H%M%S")
    json_output = {
        "_metadata": {
            "schema_version": 3,
            "metrics": list(analysis_metric_keys),
            "n_metrics": len(analysis_metric_keys),
            "report_metrics": list(report_metric_keys),
            "source_dir": os.path.abspath(src_dir),
            "pattern": args.pattern,
            "skip_frames": dict(skip_patterns),
            "timestamp": run_timestamp.isoformat(),
            "json_filename": f"{args.name}_{ts_suffix}.json",
        }
    }
    json_output.update(all_results)
    json_name = f"{args.name}_{ts_suffix}.json"
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON:  {json_path}")

    # Extract comparison frames and sample screenshots for HTML
    clip_names = sorted(all_results.keys())
    samples = None

    print("\nExtracting metric comparison frames...")
    comparisons = find_comparison_frames(clip_paths, clip_names, all_perframe,
                                         all_results, skip_offsets, report_metric_keys)

    if args.screenshots > 0:
        print(f"\nExtracting {args.screenshots} sample screenshots per clip...")
        samples = extract_sample_screenshots(clip_paths, clip_names, args.screenshots,
                                             skip_offsets)

    # HTML output (always)
    display_name = args.name.replace("_", " ").title() if args.name != "quality_report" else os.path.basename(src_dir)
    title = f"Video Quality Report — {display_name}"
    html_path = os.path.join(output_dir, f"{args.name}.html")
    with open(html_path, "w") as f:
        f.write(generate_html(all_results, report_metric_keys, title, comparisons, samples,
                              json_filename=json_name))
    print(f"HTML:  {html_path}")

    # Text output (optional)
    if args.text:
        txt_path = os.path.join(output_dir, f"{args.name}.txt")
        with open(txt_path, "w") as f:
            f.write(format_text_report(all_results, src_dir, report_metric_keys))
        print(f"Text:  {txt_path}")


if __name__ == "__main__":
    main()
