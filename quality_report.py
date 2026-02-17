#!/usr/bin/env python3
"""No-reference video quality metrics for analog video capture comparison.

Analyzes a directory of normalized ProRes 422 HQ clips (576x436, 10-bit
yuv422p10le) and produces JSON data, an interactive HTML report with
Chart.js visualizations, and optionally a plain-text report.

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
import numpy as np
import cv2
from scipy import ndimage

# =====================================================================
# VIDEO PROBING
# =====================================================================

def probe_video(filepath):
    """Get video width and height via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", filepath
    ]
    result = subprocess.check_output(cmd, text=True).strip()
    w, h = result.split(",")
    return int(w), int(h)

# =====================================================================
# METRIC METADATA
# =====================================================================

ALL_KEYS = [
    "sharpness", "edge_strength", "blocking", "detail", "texture_quality",
    "ringing", "temporal_stability", "colorfulness", "naturalness",
    "crushed_blacks", "blown_whites",
]

# Metrics still computed but excluded from scoring (low discrimination or redundant)
_DROPPED_KEYS = ["noise", "contrast", "tonal_richness", "grad_smoothness"]

METRIC_INFO = {
    "sharpness":          ("Sharpness",      "Laplacian CV\u00b2",   True),
    "edge_strength":      ("Edge Strength",  "Sobel norm.",          True),
    "blocking":           ("Blocking",       "8x8 grid ratio",      None),
    "detail":             ("Detail",         "local CV median",      True),
    "texture_quality":    ("Texture Q",      "structure ratio",      True),
    "ringing":            ("Ringing",        "edge overshoot norm.", False),
    "temporal_stability": ("Temporal",       "frame diff norm.",     False),
    "colorfulness":       ("Colorfulness",   "Hasler-S. M",          True),
    "naturalness":        ("Naturalness",    "MSCN kurtosis",        True),
    "crushed_blacks":     ("Crushed Blacks", "shadow headroom",     False),
    "blown_whites":       ("Blown Whites",   "highlight headroom",  False),
}

CMP_ICON_SVG = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<rect x="2" y="3" width="20" height="18" rx="2"/>'
    '<line x1="12" y1="3" x2="12" y2="21"/>'
    '<text x="7" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">A</text>'
    '<text x="17" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">B</text></svg>')

COLORS_7  = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#42d4f4", "#f58231", "#911eb4"]
COLORS_14 = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
             "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#800000"]

# =====================================================================
# FRAME I/O
# =====================================================================

def decode_command(filepath):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", filepath,
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "pipe:1"
    ]


def read_frame(pipe, width, height):
    """Read one YUV422p10le frame from pipe. Returns (Y, U, V) or None at EOF."""
    y_size = width * height
    u_size = (width // 2) * height
    frame_bytes = (y_size + u_size + u_size) * 2  # 16-bit samples
    data = pipe.read(frame_bytes)
    if len(data) < frame_bytes:
        return None
    raw = np.frombuffer(data, dtype=np.uint16)
    y = raw[:y_size].reshape(height, width)
    u = raw[y_size:y_size + u_size].reshape(height, width // 2)
    v = raw[y_size + u_size:].reshape(height, width // 2)
    return y, u, v


def to_float(y):
    """Convert 10-bit uint16 Y plane to float64 [0, 1]."""
    return y.astype(np.float64) / 1023.0


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


def noise_estimate(yf):
    """Noise estimate on flat/smooth regions only.

    Identifies low-gradient patches (flat areas) and measures high-frequency
    energy there. Avoids confusing blur with low noise.
    """
    block = 16
    h, w = yf.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    filtered = ndimage.convolve(yf, kernel)
    sx = cv2.Sobel(yf, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(yf, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx * sx + sy * sy)

    noise_vals = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            if np.mean(grad_mag[r:r+block, c:c+block]) < 0.03:
                patch = filtered[r:r+block, c:c+block]
                noise_vals.append(np.sqrt(np.pi / 2.0) * np.mean(np.abs(patch)) / 6.0)

    if len(noise_vals) < 5:
        block_grads = []
        for r in range(0, h - block + 1, block):
            for c in range(0, w - block + 1, block):
                mg = np.mean(grad_mag[r:r+block, c:c+block])
                patch = filtered[r:r+block, c:c+block]
                sigma = np.sqrt(np.pi / 2.0) * np.mean(np.abs(patch)) / 6.0
                block_grads.append((mg, sigma))
        block_grads.sort(key=lambda x: x[0])
        noise_vals = [s for _, s in block_grads[:max(5, len(block_grads) // 5)]]

    return np.median(noise_vals)


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
    cvs = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            blk = yf[r:r+block, c:c+block]
            mu = np.mean(blk)
            if mu > 0.01:  # skip truly black blocks
                cvs.append(np.std(blk) / mu)
    return float(np.median(cvs)) if cvs else 0.0


def texture_quality_measure(yf):
    """Structured detail ratio — how much local variance is structure vs. noise.

    Compares local variance of a mildly smoothed version to the original.
    Structured detail survives smoothing; noise does not.
    Returns ratio close to 1.0 for clean structured detail, lower for noisy detail.
    """
    block = 16
    h, w = yf.shape
    yf_smooth = cv2.GaussianBlur(yf, (5, 5), 1.0)

    ratios = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            var_orig = np.var(yf[r:r+block, c:c+block])
            var_smooth = np.var(yf_smooth[r:r+block, c:c+block])
            if var_orig > 1e-8:
                ratios.append(var_smooth / var_orig)

    return float(np.median(ratios)) if len(ratios) >= 5 else 0.5


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


# =====================================================================
# PERCEPTUAL METRICS
# =====================================================================

def perceptual_contrast(yf):
    """RMS contrast — correlates with perceived 'punch'."""
    return float(np.std(yf))


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


def tonal_richness(yf):
    """Histogram entropy of Y — measures tonal gradation quality."""
    hist, _ = np.histogram(yf.ravel(), bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


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


def gradient_smoothness(yf):
    """Smoothness of tonal gradients — inverse of banding tendency."""
    block = 32
    h, w = yf.shape
    sx = cv2.Sobel(yf, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(yf, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx * sx + sy * sy)
    gsx = cv2.Sobel(grad_mag, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(grad_mag, cv2.CV_64F, 0, 1, ksize=3)
    grad2_mag = np.sqrt(gsx * gsx + gsy * gsy)

    smoothness_vals = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            mg = np.mean(grad_mag[r:r+block, c:c+block])
            if 0.001 < mg < 0.06:
                smoothness_vals.append(1.0 / (1.0 + np.mean(grad2_mag[r:r+block, c:c+block])))

    return float(np.median(smoothness_vals)) if len(smoothness_vals) >= 3 else 0.5


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

def analyze_clip(filepath, width, height, skip_frames=0):
    """Analyze one clip, return (summary_dict, perframe_dict).

    summary_dict has the same format as before: {metric: {mean, std}, n_frames}.
    perframe_dict maps ALL metric keys -> numpy array of per-frame values.
    skip_frames: number of initial frames to discard (for timing alignment).
    """
    proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
    accum = {k: [] for k in ALL_KEYS if k != "temporal_stability"}
    temporal_diffs = []
    prev_yf = None
    n_frames = 0

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

        accum["sharpness"].append(sharpness_laplacian(yf))
        accum["edge_strength"].append(edge_strength_sobel(yf))
        accum["blocking"].append(blocking_artifact_measure(yf))
        accum["detail"].append(detail_texture(yf))
        accum["texture_quality"].append(texture_quality_measure(yf))
        accum["ringing"].append(ringing_measure(yf))
        accum["colorfulness"].append(colorfulness_metric(u, v))
        accum["naturalness"].append(naturalness_nss(yf))
        accum["crushed_blacks"].append(crushed_blacks(yf))
        accum["blown_whites"].append(blown_whites(yf))

        if prev_yf is not None:
            mean_y = 0.5 * (np.mean(yf) + np.mean(prev_yf))
            temporal_diffs.append(np.mean(np.abs(yf - prev_yf)) / (mean_y + 1e-10))
        prev_yf = yf
        n_frames += 1

    proc.wait()

    result = {}
    for k, vals in accum.items():
        arr = np.array(vals)
        # Use median for naturalness (kurtosis is 4th-power, extremely outlier-sensitive)
        avg = float(np.median(arr)) if k == "naturalness" else float(np.mean(arr))
        result[k] = {"mean": avg, "std": float(np.std(arr))}

    if temporal_diffs:
        td = np.array(temporal_diffs)
        result["temporal_stability"] = {"mean": float(np.mean(td)), "std": float(np.std(td))}
    else:
        result["temporal_stability"] = {"mean": 0.0, "std": 0.0}

    # Suppress floating-point noise on ratio metrics (tiny means from mostly-zero per-frame values)
    EPS = 1e-5
    for k in ("crushed_blacks", "blown_whites"):
        if result[k]["mean"] < EPS:
            result[k]["mean"] = 0.0

    result["n_frames"] = n_frames

    # Per-frame arrays for all metrics (used for comparison screenshots)
    perframe = {k: np.array(accum[k]) for k in accum}
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


def find_comparison_frames(clip_paths, clip_names, all_perframe, all_results, skip_offsets):
    """Find a representative frame for each metric and extract screenshots.

    For each metric M:
      a) Find the clip V whose overall M score is best.
      b) From V, find the frame where M is near its 90th percentile — a frame
         where this metric is clearly demonstrated (not just average).
      c) Extract that frame from ALL clips for side-by-side comparison.

    skip_offsets: dict {clip_name: int} — frames skipped at start (added to ffmpeg frame number).

    Returns dict: {metric_key: {"frame_num": int, "best_clip": str,
                                 "frames": {clip_name: {"jpeg_b64": str, "value": float}}}}
    """
    comparisons = {}
    for key in ALL_KEYS:
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
# COMPOSITE SCORING
# =====================================================================

def compute_composites(data):
    """Compute rank-based overall composite scores.

    For each metric, clips are ranked 1 (best) to N (worst). The overall
    score is the average rank across all 9 metrics — lower is better.
    Z-scores are also computed for heatmap cell coloring (positive = good).
    """
    clip_names = sorted(data.keys())
    n = len(clip_names)

    # Z-scores for heatmap cell coloring (sign-flipped so positive = good)
    zscores = {}
    for key in ALL_KEYS:
        arr = np.array([data[c][key]["mean"] for c in clip_names])
        mu, sigma = np.mean(arr), np.std(arr)
        zs = (arr - mu) / sigma if sigma > 1e-15 else np.zeros(n)
        _, _, hb = METRIC_INFO[key]
        if hb is False:
            zs = -zs
        elif hb is None:
            dist = np.abs(arr - 1.0)
            d_mu, d_sigma = np.mean(dist), np.std(dist)
            zs = -(dist - d_mu) / d_sigma if d_sigma > 1e-15 else np.zeros(n)
        zscores[key] = zs

    # Ranks per metric (1 = best, N = worst)
    ranks = {}
    for key in ALL_KEYS:
        arr = np.array([data[c][key]["mean"] for c in clip_names])
        _, _, hb = METRIC_INFO[key]
        if hb is True:
            order = np.argsort(-arr)
        elif hb is False:
            order = np.argsort(arr)
        else:
            order = np.argsort(np.abs(arr - 1.0))
        rank_arr = np.empty(n, dtype=float)
        for r, idx in enumerate(order, 1):
            rank_arr[idx] = r
        ranks[key] = rank_arr

    avg_ranks = np.mean([ranks[k] for k in ALL_KEYS], axis=0)

    return {c: {"overall": float(avg_ranks[i]),
                "ranks": {k: int(ranks[k][i]) for k in ALL_KEYS},
                "zscores": {k: float(zscores[k][i]) for k in ALL_KEYS}}
            for i, c in enumerate(clip_names)}


# =====================================================================
# TEXT REPORT
# =====================================================================

def format_text_report(all_results, src_dir):
    """Format results into a readable text report with auto-sized columns."""
    clip_names = sorted(all_results.keys())
    col_w = max(len(n) for n in clip_names) + 2

    lines = []
    lines.append("=" * 120)
    lines.append(f"NO-REFERENCE VIDEO QUALITY METRICS REPORT")
    lines.append("=" * 120)
    lines.append(f"Source: {src_dir}")
    lines.append("All clips normalized to common brightness/saturation reference before measurement.")
    lines.append("")

    for label, keys, headers in [
        ("Quality Metrics", ALL_KEYS,
         ["Sharpness", "Edge Str", "Blocking", "Detail", "TexQual", "Ringing", "Temporal", "Colorful", "Natural", "CrBlack", "BlnWhite"]),
    ]:
        lines.append(f"--- {label} ---")
        header = f"{'Clip':<{col_w}}" + "".join(f" {h:>10}" for h in headers)
        lines.append(header)
        lines.append("-" * len(header))
        for name in clip_names:
            row = f"{name:<{col_w}}"
            for k in keys:
                v = all_results[name][k]["mean"]
                if k in ("crushed_blacks", "blown_whites"):
                    row += f" {v*100:>9.1f}%"
                elif k == "blocking":
                    row += f" {v:>10.4f}"
                elif k == "colorfulness":
                    row += f" {v:>10.2f}"
                elif k == "naturalness":
                    row += f" {v:>10.4f}"
                else:
                    row += f" {v:>10.6f}"
            lines.append(row)
        lines.append("")

    # Rankings
    lines.append("=" * 120)
    lines.append("RANKINGS (best to worst per metric)")
    lines.append("=" * 120)

    for key in ALL_KEYS:
        label, unit, higher_better = METRIC_INFO[key]
        if higher_better is True:
            dir_str = "(higher = better)"
        elif higher_better is False:
            dir_str = "(lower = better)"
        else:
            dir_str = "(closer to 1.0 = better)"
        label = f"{label} {dir_str}"
        higher_better = higher_better  # used below
        lines.append(f"\n  {label}:")
        items = [(n, all_results[n][key]["mean"]) for n in clip_names]
        if higher_better is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif higher_better:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])
        for rank, (name, val) in enumerate(items, 1):
            if key in ("crushed_blacks", "blown_whites"):
                lines.append(f"    {rank:>2}. {name:<{col_w}} {val*100:.1f}%")
            else:
                fmt = ".4f" if key == "blocking" else ".2f" if key == "colorfulness" else ".4f" if key == "naturalness" else ".6f"
                lines.append(f"    {rank:>2}. {name:<{col_w}} {val:{fmt}}")

    # Composite rankings
    composites = compute_composites(all_results)

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


def generate_html(data, title="Video Quality Report", comparisons=None, samples=None):
    """Generate a self-contained HTML quality report with Chart.js visualizations."""
    clip_names = sorted(data.keys())
    n = len(clip_names)
    palette = COLORS_7 if n <= 7 else COLORS_14
    colors = {c: palette[i % len(palette)] for i, c in enumerate(clip_names)}

    composites = compute_composites(data)
    ranked_overall = sorted(clip_names, key=lambda c: composites[c]["overall"])

    # Radar: normalize each metric to 0-100
    radar_scores = {}
    for c in clip_names:
        sc = []
        for key in ALL_KEYS:
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
    radar_labels = [METRIC_INFO[k][0] for k in ALL_KEYS]

    # Heatmap data
    hm_data = []
    for c in ranked_overall:
        cells = [{"raw": data[c][k]["mean"], "z": composites[c]["zscores"][k], "key": k} for k in ALL_KEYS]
        hm_data.append({"name": c, "overall": composites[c]["overall"], "cells": cells})

    # Per-metric rankings
    per_metric = {}
    for key in ALL_KEYS:
        _, _, hb = METRIC_INFO[key]
        items = [(c, data[c][key]["mean"]) for c in clip_names]
        if hb is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif hb:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])
        is_pct = key in ("crushed_blacks", "blown_whites")
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
  {n} analog video captures evaluated on 11 no-reference quality metrics
  (all brightness-agnostic — no upstream normalization required).
</p>

<div class="card">
  <h3>Metric Guide</h3>
  <div class="metric-guide">
    <div><span class="mg-label">Sharpness</span> <span class="dir dir-up">higher</span><br>Laplacian variance — blur vs. sharpness</div>
    <div><span class="mg-label">Edge Strength</span> <span class="dir dir-up">higher</span><br>Sobel gradient — edge definition</div>
    <div><span class="mg-label">Blocking</span> <span class="dir dir-mid">~1.0</span><br>8x8 DCT boundary ratio</div>
    <div><span class="mg-label">Detail</span> <span class="dir dir-up">higher</span><br>Local variance — micro-detail</div>
    <div><span class="mg-label">Texture Q</span> <span class="dir dir-up">higher</span><br>Structure/noise ratio — detail quality</div>
    <div><span class="mg-label">Ringing</span> <span class="dir dir-down">lower</span><br>Edge overshoot / haloing — retained despite high sharpness correlation; reflects a distinct analog artifact (VCR sharpness circuits, aperture correction) that varies independently across hardware</div>
    <div><span class="mg-label">Temporal</span> <span class="dir dir-down">lower</span><br>Frame-to-frame diff — flicker</div>
    <div><span class="mg-label">Colorfulness</span> <span class="dir dir-up">higher</span><br>Hasler-S&uuml;sstrunk — color vibrancy</div>
    <div><span class="mg-label">Naturalness</span> <span class="dir dir-up">higher</span><br>MSCN kurtosis — natural signal statistics</div>
    <div><span class="mg-label">Crushed Blacks</span> <span class="dir dir-down">lower %</span><br>Shadow headroom — fraction of shadow pixels clipped to near-black</div>
    <div><span class="mg-label">Blown Whites</span> <span class="dir dir-down">lower %</span><br>Highlight headroom — fraction of highlight pixels clipped to near-white</div>
  </div>
</div>

<h2>Overall Composite Ranking</h2>
<div class="card">
  <div class="chart-container" style="height:{bar_height}px;"><canvas id="overallChart"></canvas></div>
  <p class="legend-note">Average rank across all 11 metrics (1 = best per metric). Lower average rank = better overall quality.</p>
</div>

<h2>11-Metric Radar Comparison</h2>
<div class="card">
  <div class="chart-container chart-radar"><canvas id="radarChart"></canvas></div>
  <p class="legend-note">All metrics on 0–100 quality scale (higher = better on all axes). Click legend to toggle.{" Top 5 shown by default." if n > 5 else ""}</p>
</div>

<h2>Detailed Heatmap — All Metrics</h2>
<div class="card" style="overflow-x:auto;">
  <table class="heatmap" id="heatmapTable">
    <thead><tr>
      <th>Clip</th><th>Avg Rank</th>
      <th>Sharp</th><th>Edge</th><th>Block</th><th>Detail</th><th>TexQ</th><th>Ring</th><th>Temp</th><th>Color</th><th>Natur</th><th>CrBlk</th><th>BlnWh</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  <p class="legend-note">Ranked by average rank across all 11 metrics (lower = better). Cell color: green = good, red = poor. Raw metric values shown; CrBlk/BlnWh as percentages. Click any column header to sort.</p>
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

const hmData={json.dumps(hm_data)};
const tbody=document.querySelector('#heatmapTable tbody');
hmData.forEach((row,idx)=>{{
  const tr=document.createElement('tr');
  let td=document.createElement('td');
  td.innerHTML='<span class="rank">#'+(idx+1)+'</span> '+row.name;
  tr.appendChild(td);
  td=document.createElement('td');td.textContent=row.overall.toFixed(1);
  const mid=({n}+1)/2, ov=Math.max(-1,Math.min(1,(mid-row.overall)/mid));
  td.style.background=ov>0?'rgba(63,185,80,'+(Math.abs(ov)*0.4)+')':'rgba(248,81,73,'+(Math.abs(ov)*0.4)+')';
  td.style.fontWeight='600';tr.appendChild(td);
  row.cells.forEach(cell=>{{
    td=document.createElement('td');
    td.textContent=['crushed_blacks','blown_whites'].includes(cell.key)?(cell.raw*100).toFixed(1)+'%':['blocking'].includes(cell.key)?cell.raw.toFixed(4):['colorfulness'].includes(cell.key)?cell.raw.toFixed(1):['naturalness'].includes(cell.key)?cell.raw.toFixed(3):cell.raw.toFixed(5);
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
      fc.innerHTML='<span class="rank">#'+(i+1)+'</span> '+name;
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
        for key in ALL_KEYS:
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

    print(f"Analyzing {len(clips)} clips in {src_dir} ({vid_w}x{vid_h})...\n")

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
        skip_msg = f", skip {skip}" if skip > 0 else ""
        print(f"[{i:>2}/{len(clips)}] {name}{skip_msg}...", end=" ", flush=True)
        metrics, perframe = analyze_clip(clip_path, vid_w, vid_h, skip_frames=skip)
        all_results[name] = metrics
        all_perframe[name] = perframe
        print(f"done ({metrics['n_frames']} frames)")

    # JSON output (always)
    json_path = os.path.join(output_dir, f"{args.name}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON:  {json_path}")

    # Extract comparison frames and sample screenshots for HTML
    clip_names = sorted(all_results.keys())
    comparisons = None
    samples = None

    if args.screenshots > 0 or True:  # always extract comparison frames
        print("\nExtracting metric comparison frames...")
        comparisons = find_comparison_frames(clip_paths, clip_names, all_perframe,
                                             all_results, skip_offsets)

    if args.screenshots > 0:
        print(f"\nExtracting {args.screenshots} sample screenshots per clip...")
        samples = extract_sample_screenshots(clip_paths, clip_names, args.screenshots,
                                             skip_offsets)

    # HTML output (always)
    display_name = args.name.replace("_", " ").title() if args.name != "quality_report" else os.path.basename(src_dir)
    title = f"Video Quality Report — {display_name}"
    html_path = os.path.join(output_dir, f"{args.name}.html")
    with open(html_path, "w") as f:
        f.write(generate_html(all_results, title, comparisons, samples))
    print(f"HTML:  {html_path}")

    # Text output (optional)
    if args.text:
        txt_path = os.path.join(output_dir, f"{args.name}.txt")
        with open(txt_path, "w") as f:
            f.write(format_text_report(all_results, src_dir))
        print(f"Text:  {txt_path}")


if __name__ == "__main__":
    main()
