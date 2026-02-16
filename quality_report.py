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

Metrics (13 total):
  Technical (8): sharpness, edge strength, noise, blocking, detail, texture quality, ringing, temporal
  Perceptual (5): contrast, colorfulness, tonal richness, naturalness, gradient smoothness
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

TECH_KEYS = ["sharpness", "edge_strength", "noise", "blocking", "detail", "texture_quality", "ringing", "temporal_stability"]
PERC_KEYS = ["contrast", "colorfulness", "tonal_richness", "naturalness", "grad_smoothness"]
ALL_KEYS = TECH_KEYS + PERC_KEYS

METRIC_INFO = {
    "sharpness":          ("Sharpness",      "Laplacian var",       True,  "tech"),
    "edge_strength":      ("Edge Strength",  "Sobel mean",          True,  "tech"),
    "noise":              ("Noise",          "flat-region est.",     False, "tech"),
    "blocking":           ("Blocking",       "8x8 grid ratio",      None,  "tech"),
    "detail":             ("Detail",         "local var median",     True,  "tech"),
    "texture_quality":    ("Texture Q",      "structure ratio",      True,  "tech"),
    "ringing":            ("Ringing",        "edge overshoot",       False, "tech"),
    "temporal_stability": ("Temporal",       "frame diff mean",      False, "tech"),
    "contrast":           ("Contrast",       "RMS contrast",         True,  "perc"),
    "colorfulness":       ("Colorfulness",   "Hasler-S. M",          True,  "perc"),
    "tonal_richness":     ("Tonal Richness", "Y entropy (bits)",     True,  "perc"),
    "naturalness":        ("Naturalness",    "MSCN kurtosis",        True,  "perc"),
    "grad_smoothness":    ("Grad Smooth",    "gradient smoothness",  True,  "perc"),
}

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
    """Laplacian variance — standard no-ref sharpness metric."""
    return np.var(cv2.Laplacian(yf, cv2.CV_64F))


def edge_strength_sobel(yf):
    """Mean Sobel gradient magnitude — edge definition quality."""
    sx = cv2.Sobel(yf, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(yf, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(sx * sx + sy * sy))


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
    """Median local variance in 16x16 blocks — detail/texture preservation."""
    block = 16
    h, w = yf.shape
    variances = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            variances.append(np.var(yf[r:r+block, c:c+block]))
    return np.median(variances)


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
    """Detect ringing/haloing around edges via near-edge Laplacian energy."""
    edges = cv2.Canny((yf * 255).astype(np.uint8), 50, 150)
    if edges.sum() == 0:
        return 0.0
    near_edge = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    near_edge_only = near_edge & ~edges
    if near_edge_only.sum() == 0:
        return 0.0
    return np.mean(np.abs(cv2.Laplacian(yf, cv2.CV_64F))[near_edge_only > 0])


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
        accum["noise"].append(noise_estimate(yf))
        accum["blocking"].append(blocking_artifact_measure(yf))
        accum["detail"].append(detail_texture(yf))
        accum["texture_quality"].append(texture_quality_measure(yf))
        accum["ringing"].append(ringing_measure(yf))
        accum["contrast"].append(perceptual_contrast(yf))
        accum["colorfulness"].append(colorfulness_metric(u, v))
        accum["tonal_richness"].append(tonal_richness(yf))
        accum["naturalness"].append(naturalness_nss(yf))
        accum["grad_smoothness"].append(gradient_smoothness(yf))

        if prev_yf is not None:
            temporal_diffs.append(np.mean(np.abs(yf - prev_yf)))
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
        _, _, higher_better, _ = METRIC_INFO[key]

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
    """Compute discrimination-weighted technical, perceptual, and overall composite z-scores.

    Each metric's z-score is weighted by its coefficient of variation (relative spread
    across clips). Metrics where all clips score similarly contribute less to rankings.
    Weights are normalized within tech/perc groups so total weight = number of metrics.
    """
    clip_names = sorted(data.keys())
    n = len(clip_names)

    zscores = {}
    raw_weights = {}
    for key in ALL_KEYS:
        arr = np.array([data[c][key]["mean"] for c in clip_names])
        mu, sigma = np.mean(arr), np.std(arr)
        zscores[key] = (arr - mu) / sigma if sigma > 1e-15 else np.zeros(n)
        # Discrimination weight: CV (relative spread). Use max(|mu|, sigma) as
        # denominator to handle near-zero means gracefully.
        abs_mu = max(abs(float(mu)), float(sigma), 1e-10)
        raw_weights[key] = float(sigma) / abs_mu

    # Normalize weights within each group so total weight = number of metrics in group
    weights = dict(raw_weights)
    for group_keys in [TECH_KEYS, PERC_KEYS]:
        total_w = sum(weights[k] for k in group_keys)
        if total_w > 0:
            scale = len(group_keys) / total_w
            for k in group_keys:
                weights[k] *= scale

    tech = np.zeros(n)
    for k in ["sharpness", "edge_strength", "detail", "texture_quality"]:
        tech += weights[k] * zscores[k]
    for k in ["noise", "ringing", "temporal_stability"]:
        tech -= weights[k] * zscores[k]
    blocking_dist = np.abs(np.array([data[c]["blocking"]["mean"] for c in clip_names]) - 1.0)
    bd_mu, bd_sigma = np.mean(blocking_dist), np.std(blocking_dist)
    if bd_sigma > 1e-15:
        tech -= weights["blocking"] * (blocking_dist - bd_mu) / bd_sigma

    perc = sum(weights[k] * zscores[k] for k in PERC_KEYS)

    tc_mu, tc_sig = np.mean(tech), np.std(tech)
    pc_mu, pc_sig = np.mean(perc), np.std(perc)
    tech_norm = (tech - tc_mu) / tc_sig if tc_sig > 1e-15 else np.zeros(n)
    perc_norm = (perc - pc_mu) / pc_sig if pc_sig > 1e-15 else np.zeros(n)
    overall = tech_norm + perc_norm

    return {c: {"tech": float(tech[i]), "perc": float(perc[i]), "overall": float(overall[i]),
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
        ("Technical Metrics", TECH_KEYS,
         ["Sharpness", "Edge Str", "Noise", "Blocking", "Detail", "TexQual", "Ringing", "Temporal"]),
        ("Perceptual / Cinematic Metrics", PERC_KEYS,
         ["Contrast", "Colorful", "TonalRich", "Natural", "GradSmooth"]),
    ]:
        lines.append(f"--- {label} ---")
        header = f"{'Clip':<{col_w}}" + "".join(f" {h:>10}" for h in headers)
        lines.append(header)
        lines.append("-" * len(header))
        for name in clip_names:
            row = f"{name:<{col_w}}"
            for k in keys:
                v = all_results[name][k]["mean"]
                if k == "blocking":
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

    for key, label, higher_better in [
        ("sharpness", "Sharpness (higher = better)", True),
        ("edge_strength", "Edge Strength (higher = better)", True),
        ("noise", "Noise Level (lower = better)", False),
        ("blocking", "Blocking Artifacts (closer to 1.0 = better)", None),
        ("detail", "Detail Preservation (higher = better)", True),
        ("texture_quality", "Texture Quality / Structure Ratio (higher = better)", True),
        ("ringing", "Ringing/Haloing (lower = better)", False),
        ("temporal_stability", "Temporal Stability (lower = better)", False),
        ("contrast", "Perceptual Contrast (higher = better)", True),
        ("colorfulness", "Colorfulness (higher = better)", True),
        ("tonal_richness", "Tonal Richness (higher = better)", True),
        ("naturalness", "Naturalness / MSCN Kurtosis (higher = better)", True),
        ("grad_smoothness", "Gradient Smoothness (higher = better)", True),
    ]:
        lines.append(f"\n  {label}:")
        items = [(n, all_results[n][key]["mean"]) for n in clip_names]
        if higher_better is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif higher_better:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])
        for rank, (name, val) in enumerate(items, 1):
            fmt = ".4f" if key == "blocking" else ".2f" if key == "colorfulness" else ".4f" if key == "naturalness" else ".6f"
            lines.append(f"    {rank:>2}. {name:<{col_w}} {val:{fmt}}")

    # Composite rankings
    composites = compute_composites(all_results)

    for title, score_key in [
        ("TECHNICAL QUALITY RANKING", "tech"),
        ("PERCEPTUAL / CINEMATIC RANKING", "perc"),
        ("OVERALL COMPOSITE RANKING (Technical + Perceptual)", "overall"),
    ]:
        lines.append("")
        lines.append("=" * 120)
        lines.append(title)
        lines.append("=" * 120)
        lines.append("")
        ranked = sorted(clip_names, key=lambda c: -composites[c][score_key])
        for rank, name in enumerate(ranked, 1):
            lines.append(f"  {rank:>2}. {name:<{col_w}} {score_key} z-score: {composites[name][score_key]:>+.3f}")

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
    --perc-accent: #d2a8ff; --tech-accent: #58a6ff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; padding: 24px; max-width: 1400px; margin: 0 auto;
  }
  h1 { font-size: 1.8em; margin-bottom: 8px; }
  h2 { font-size: 1.3em; margin: 40px 0 16px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  h2.perc { color: var(--perc-accent); }
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
  .metric-card.perc-card { border-color: #3d2b5e; }
  .heatmap { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  .heatmap th { background: #21262d; padding: 6px 5px; text-align: center; font-weight: 600;
    border: 1px solid var(--border); white-space: nowrap; position: sticky; top: 0; z-index: 2;
    cursor: pointer; user-select: none; }
  .heatmap th:hover { background: #2d333b; }
  .heatmap th::after { content: ' \\2195'; opacity: 0.3; font-size: 0.8em; }
  .heatmap th.sort-asc::after { content: ' \\2191'; opacity: 0.8; }
  .heatmap th.sort-desc::after { content: ' \\2193'; opacity: 0.8; }
  .heatmap th:first-child { text-align: left; min-width: 140px; }
  .heatmap .th-tech { border-bottom: 2px solid var(--tech-accent); }
  .heatmap .th-perc { border-bottom: 2px solid var(--perc-accent); }
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
  .section-label { display: inline-block; font-size: 0.75em; padding: 2px 8px; border-radius: 4px;
    margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px; }
  .section-label.tech { background: #1a2a3a; color: var(--tech-accent); }
  .section-label.perc { background: #2a1a3a; color: var(--perc-accent); }
  /* Lightbox — CSS-only click-to-enlarge */
  .lb-toggle { display: none; }
  .lb-thumb { cursor: zoom-in; display: block; }
  .lb-thumb img { transition: opacity 0.15s; }
  .lb-thumb:hover img { opacity: 0.85; }
  .lb-overlay { display: none; position: fixed; inset: 0;
    background: #000; z-index: 9999; cursor: zoom-out;
    justify-content: center; align-items: center; flex-direction: column; padding: 20px; }
  .lb-overlay img { max-width: 95vw; max-height: 85vh; object-fit: contain; border-radius: 4px; }
  .lb-overlay .lb-caption { color: #e6edf3; font-size: 0.9em; margin-top: 12px;
    text-align: center; max-width: 90vw; }
  .lb-toggle:checked + label.lb-overlay { display: flex; }
"""


def generate_html(data, title="Video Quality Report", comparisons=None, samples=None):
    """Generate a self-contained HTML quality report with Chart.js visualizations."""
    clip_names = sorted(data.keys())
    n = len(clip_names)
    palette = COLORS_7 if n <= 7 else COLORS_14
    colors = {c: palette[i % len(palette)] for i, c in enumerate(clip_names)}

    composites = compute_composites(data)
    ranked_overall = sorted(clip_names, key=lambda c: -composites[c]["overall"])
    ranked_tech = sorted(clip_names, key=lambda c: -composites[c]["tech"])
    ranked_perc = sorted(clip_names, key=lambda c: -composites[c]["perc"])

    # Radar: normalize each metric to 0-100
    radar_scores = {}
    for c in clip_names:
        sc = []
        for key in ALL_KEYS:
            vals = [data[d][key]["mean"] for d in clip_names]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx - mn > 1e-15 else 1e-15
            raw = data[c][key]["mean"]
            _, _, hb, _ = METRIC_INFO[key]
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
        hm_data.append({"name": c, "tech": composites[c]["tech"],
                        "perc": composites[c]["perc"], "overall": composites[c]["overall"], "cells": cells})

    # Per-metric rankings
    per_metric = {}
    for key in ALL_KEYS:
        _, _, hb, group = METRIC_INFO[key]
        items = [(c, data[c][key]["mean"]) for c in clip_names]
        if hb is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif hb:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])
        per_metric[key] = {
            "labels": [c for c, _ in items], "values": [v for _, v in items],
            "colors": [colors[c] for c, _ in items],
            "label": METRIC_INFO[key][0], "unit": METRIC_INFO[key][1],
            "higher_better": hb, "group": group
        }

    # Scatter points
    scatter_points = [{"x": composites[c]["tech"], "y": composites[c]["perc"],
                       "label": c, "color": colors[c]} for c in clip_names]

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
  {n} analog video captures — normalized to common brightness &amp; saturation —
  evaluated on 8 technical + 5 perceptual no-reference metrics.
</p>

<div class="card">
  <h3>Metric Guide</h3>
  <span class="section-label tech">TECHNICAL / ARTIFACT</span>
  <div class="metric-guide" style="margin-bottom:16px;">
    <div><span class="mg-label">Sharpness</span> <span class="dir dir-up">higher</span><br>Laplacian variance — blur vs. sharpness</div>
    <div><span class="mg-label">Edge Strength</span> <span class="dir dir-up">higher</span><br>Sobel gradient — edge definition</div>
    <div><span class="mg-label">Noise</span> <span class="dir dir-down">lower</span><br>High-freq energy in flat patches</div>
    <div><span class="mg-label">Blocking</span> <span class="dir dir-mid">~1.0</span><br>8x8 DCT boundary ratio</div>
    <div><span class="mg-label">Detail</span> <span class="dir dir-up">higher</span><br>Local variance — micro-detail</div>
    <div><span class="mg-label">Texture Q</span> <span class="dir dir-up">higher</span><br>Structure/noise ratio — detail quality</div>
    <div><span class="mg-label">Ringing</span> <span class="dir dir-down">lower</span><br>Edge overshoot / haloing</div>
    <div><span class="mg-label">Temporal</span> <span class="dir dir-down">lower</span><br>Frame-to-frame diff — flicker</div>
  </div>
  <span class="section-label perc">PERCEPTUAL / CINEMATIC</span>
  <div class="metric-guide">
    <div><span class="mg-label">Contrast</span> <span class="dir dir-up">higher</span><br>RMS contrast — tonal punch</div>
    <div><span class="mg-label">Colorfulness</span> <span class="dir dir-up">higher</span><br>Hasler-S&uuml;sstrunk — color vibrancy</div>
    <div><span class="mg-label">Tonal Richness</span> <span class="dir dir-up">higher</span><br>Histogram entropy — smooth gradations</div>
    <div><span class="mg-label">Naturalness</span> <span class="dir dir-up">higher</span><br>MSCN kurtosis — natural signal statistics</div>
    <div><span class="mg-label">Grad Smoothness</span> <span class="dir dir-up">higher</span><br>Gradient continuity — anti-banding</div>
  </div>
</div>

<h2>Overall Composite Ranking (Technical + Perceptual)</h2>
<div class="card">
  <div class="chart-container" style="height:{bar_height}px;"><canvas id="overallChart"></canvas></div>
  <p class="legend-note">Discrimination-weighted combination of technical and perceptual sub-scores. Metrics with more spread across devices get more influence.</p>
</div>

<h2>Technical vs. Perceptual Quality</h2>
<div class="card">
  <div class="chart-container" style="height:500px;"><canvas id="scatterChart"></canvas></div>
  <p class="legend-note">Upper-right = best on both dimensions. Each dot is one capture pipeline.</p>
</div>

<h2>Technical &amp; Perceptual Rankings</h2>
<div class="two-col">
  <div class="card">
    <span class="section-label tech">TECHNICAL</span>
    <div class="chart-container" style="height:{bar_height}px;"><canvas id="techChart"></canvas></div>
  </div>
  <div class="card">
    <span class="section-label perc">PERCEPTUAL</span>
    <div class="chart-container" style="height:{bar_height}px;"><canvas id="percChart"></canvas></div>
  </div>
</div>

<h2>13-Metric Radar Comparison</h2>
<div class="card">
  <div class="chart-container chart-radar"><canvas id="radarChart"></canvas></div>
  <p class="legend-note">All metrics on 0–100 quality scale (higher = better on all axes). Click legend to toggle.{" Top 5 shown by default." if n > 5 else ""}</p>
</div>

<h2>Detailed Heatmap — All Metrics</h2>
<div class="card" style="overflow-x:auto;">
  <table class="heatmap" id="heatmapTable">
    <thead><tr>
      <th>Clip</th><th>Overall</th><th>Tech</th><th>Perc</th>
      <th class="th-tech">Sharp</th><th class="th-tech">Edge</th><th class="th-tech">Noise</th><th class="th-tech">Block</th><th class="th-tech">Detail</th><th class="th-tech">TexQ</th><th class="th-tech">Ring</th><th class="th-tech">Temp</th>
      <th class="th-perc">Contr</th><th class="th-perc">Color</th><th class="th-perc">Tonal</th><th class="th-perc">Natur</th><th class="th-perc">Grad</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  <p class="legend-note">Ranked by overall composite. Cell color: green = good, red = poor (z-score). Raw metric values shown. Click any column header to sort.</p>
</div>

<h2>Technical Metric Rankings</h2>
<div class="metric-grid" id="techMetricGrid"></div>
<h2 class="perc">Perceptual / Cinematic Metric Rankings</h2>
<div class="metric-grid" id="percMetricGrid"></div>

<script>
function hbar(id, labels, scores, clrs, xlabel) {{
  new Chart(document.getElementById(id), {{
    type:'bar', data:{{ labels, datasets:[{{ data:scores, backgroundColor:clrs, borderColor:clrs, borderWidth:1, borderRadius:4 }}] }},
    options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>'z-score: '+c.parsed.x.toFixed(3)}}}} }},
      scales:{{ x:{{grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}},title:{{display:!!xlabel,text:xlabel||'',color:'#8b949e'}}}}, y:{{grid:{{display:false}},ticks:{{color:'#e6edf3',font:{{size:11}}}}}} }}
    }}
  }});
}}

hbar('overallChart',{json.dumps([c for c in ranked_overall])},{json.dumps([composites[c]["overall"] for c in ranked_overall])},{json.dumps([colors[c] for c in ranked_overall])},'Overall Z-Score (higher = better)');
hbar('techChart',{json.dumps([c for c in ranked_tech])},{json.dumps([composites[c]["tech"] for c in ranked_tech])},{json.dumps([colors[c] for c in ranked_tech])},'Technical Z-Score');
hbar('percChart',{json.dumps([c for c in ranked_perc])},{json.dumps([composites[c]["perc"] for c in ranked_perc])},{json.dumps([colors[c] for c in ranked_perc])},'Perceptual Z-Score');

new Chart(document.getElementById('scatterChart'),{{
  type:'scatter',
  data:{{datasets:{json.dumps(scatter_points)}.map(p=>({{label:p.label,data:[{{x:p.x,y:p.y}}],backgroundColor:p.color,borderColor:p.color,pointRadius:8,pointHoverRadius:11}}))}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.dataset.label+' (tech:'+c.parsed.x.toFixed(2)+', perc:'+c.parsed.y.toFixed(2)+')'}}}}}},
    scales:{{x:{{grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}},title:{{display:true,text:'Technical Quality Z-Score →',color:'#58a6ff'}}}},y:{{grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}},title:{{display:true,text:'Perceptual Quality Z-Score →',color:'#d2a8ff'}}}}}}
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
  [row.overall,row.tech,row.perc].forEach(val=>{{
    td=document.createElement('td');td.textContent=val.toFixed(2);
    const n=Math.max(-1,Math.min(1,val/5));
    td.style.background=n>0?'rgba(63,185,80,'+(Math.abs(n)*0.4)+')':'rgba(248,81,73,'+(Math.abs(n)*0.4)+')';
    td.style.fontWeight='600';tr.appendChild(td);
  }});
  row.cells.forEach(cell=>{{
    td=document.createElement('td');
    td.textContent=['blocking'].includes(cell.key)?cell.raw.toFixed(4):['colorfulness'].includes(cell.key)?cell.raw.toFixed(1):['naturalness'].includes(cell.key)?cell.raw.toFixed(3):cell.raw.toFixed(5);
    const z=cell.z,n=Math.max(-1,Math.min(1,z/2.5));
    td.style.background=n>0?'rgba(63,185,80,'+(Math.abs(n)*0.35)+')':'rgba(248,81,73,'+(Math.abs(n)*0.35)+')';
    tr.appendChild(td);
  }});
  tbody.appendChild(tr);
}});

const amd={json.dumps(per_metric)};
['techMetricGrid','percMetricGrid'].forEach(gridId=>{{
  const grid=document.getElementById(gridId);
  const group=gridId==='techMetricGrid'?'tech':'perc';
  Object.entries(amd).filter(([,md])=>md.group===group).forEach(([key,md])=>{{
    const card=document.createElement('div');
    card.className=group==='perc'?'metric-card perc-card':'metric-card';
    const canvas=document.createElement('canvas');card.appendChild(canvas);grid.appendChild(card);
    const dir=md.higher_better===true?'(higher = better)':md.higher_better===false?'(lower = better)':'(closer to 1.0)';
    new Chart(canvas,{{
      type:'bar',data:{{labels:md.labels,datasets:[{{data:md.values,backgroundColor:md.colors.map(c=>c+'cc'),borderColor:md.colors,borderWidth:1,borderRadius:3}}]}},
      options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}},title:{{display:true,text:md.label+' '+dir,color:group==='perc'?'#d2a8ff':'#e6edf3',font:{{size:14}}}},tooltip:{{callbacks:{{label:c=>md.unit+': '+c.parsed.x.toFixed(6)}}}}}},
        scales:{{x:{{grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}}}},y:{{grid:{{display:false}},ticks:{{color:'#e6edf3',font:{{size:10}}}}}}}}
      }}
    }});
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
is shown. All clips are shown at the same frame number for direct comparison. Click any image to enlarge.</p>
"""
        for key in ALL_KEYS:
            if key not in comparisons:
                continue
            comp = comparisons[key]
            label, unit, higher_better, group = METRIC_INFO[key]
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
            for rank, (name, fdata) in enumerate(frame_items, 1):
                if fdata["jpeg_b64"]:
                    img_src = f"data:image/jpeg;base64,{fdata['jpeg_b64']}"
                    caption = f"#{rank} {name} &mdash; {label}: {fdata['value']:.6f} &mdash; Frame {comp['frame_num']}"
                    lb_overlays.append((lb_id, img_src, caption))
                    html += f"""  <div class="card" style="flex:1;min-width:300px;max-width:48%;">
    <div style="font-weight:600;margin-bottom:4px;"><span class="rank">#{rank}</span> {name}</div>
    <div style="font-size:0.9em;color:var(--text-dim);margin-bottom:8px;">{label}: {fdata["value"]:.6f}</div>
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
            html += f'<label for="lb{lid}" class="lb-overlay"><img src="{img_src}"><div class="lb-caption">{caption}</div></label>\n'

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
    title = f"Video Quality Report — {os.path.basename(src_dir) or args.name}"
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
