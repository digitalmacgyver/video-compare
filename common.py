"""Shared constants, metric metadata, and utility functions for video quality analysis.

Used by quality_report.py, cross_clip_report.py, normalize.py, and normalize_linear.py.
"""

import subprocess
import os
import json
import numpy as np
from scipy.stats import rankdata

# =====================================================================
# METRIC METADATA
# =====================================================================

ALL_KEYS = [
    "sharpness", "edge_strength", "blocking", "detail", "texture_quality",
    "ringing", "temporal_stability", "colorfulness", "naturalness",
    "crushed_blacks", "blown_whites",
]

METRIC_INFO = {
    "sharpness":          ("Sharpness",      "Laplacian CV\u00b2",   True),
    "edge_strength":      ("Edge Strength",  "Sobel norm.",          True),
    "blocking":           ("Blocking",       "8x8 grid ratio",      None),
    "detail":             ("Detail",         "local CV median",      True),
    "detail_perceptual":  ("Detail VHS-SD",  "z(blur)-z(sml)-z(tex)", True),
    "detail_tenengrad":   ("Detail Teneng",  "robust Sobel",         True),
    "detail_sml":         ("Detail SML",     "robust mod Lap.",      True),
    "detail_blur_inv":    ("Detail BlurInv", "1 - blur effect",      True),
    "texture_quality":    ("Texture Q",      "structure ratio",      True),
    "ringing":            ("Ringing",        "edge overshoot norm.", False),
    "temporal_stability": ("Temporal",       "frame diff norm.",     False),
    "colorfulness":       ("Colorfulness",   "Hasler-S. M",          True),
    "naturalness":        ("Naturalness",    "MSCN kurtosis",        True),
    "crushed_blacks":     ("Crushed Blacks", "shadow headroom",     False),
    "blown_whites":       ("Blown Whites",   "highlight headroom",  False),
}

COLORS_7  = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#42d4f4", "#f58231", "#911eb4"]
COLORS_14 = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
             "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#800000"]

# =====================================================================
# COMPOSITE SCORING
# =====================================================================

NON_DISC_CV_THRESHOLD = 0.01  # Metrics with CV below 1% are excluded from composite


def compute_composites(data, metric_keys=None):
    """Compute rank-based overall composite scores.

    For each metric, clips are ranked 1 (best) to N (worst) using tie-aware
    fractional ranks.  The overall score is the average rank across all
    discriminating metrics — lower is better.  Metrics with coefficient of
    variation below NON_DISC_CV_THRESHOLD are excluded from the composite
    but still reported in per-metric ranks.
    Z-scores are also computed for heatmap cell coloring (positive = good).
    """
    clip_names = sorted(k for k in data.keys() if not k.startswith("_"))
    n = len(clip_names)
    key_order = list(metric_keys) if metric_keys is not None else list(ALL_KEYS)
    # Only use metrics present in all clips (backward compat with older JSONs)
    available = set(key_order)
    for c in clip_names:
        available &= set(data[c].keys())
    keys = [k for k in key_order if k in available]

    if not keys:
        return {c: {"overall": 0.0, "ranks": {}, "zscores": {}, "non_discriminating": []}
                for c in clip_names}

    # Identify non-discriminating metrics (CV below threshold)
    non_discriminating = []
    for key in keys:
        arr = np.array([data[c][key]["mean"] for c in clip_names])
        mu = np.mean(arr)
        cv = np.std(arr) / (np.abs(mu) + 1e-10)
        if cv < NON_DISC_CV_THRESHOLD:
            non_discriminating.append(key)
    active_keys = [k for k in keys if k not in non_discriminating]

    zscores = {}
    for key in keys:
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

    # Ranks per metric (1 = best, N = worst), tie-aware — computed for all keys
    ranks = {}
    for key in keys:
        arr = np.array([data[c][key]["mean"] for c in clip_names])
        _, _, hb = METRIC_INFO[key]
        if hb is True:
            ranks[key] = rankdata(-arr, method='average')
        elif hb is False:
            ranks[key] = rankdata(arr, method='average')
        else:
            ranks[key] = rankdata(np.abs(arr - 1.0), method='average')

    # Overall composite uses only active (discriminating) keys
    if active_keys:
        avg_ranks = np.mean([ranks[k] for k in active_keys], axis=0)
    else:
        avg_ranks = np.mean([ranks[k] for k in keys], axis=0)

    return {c: {"overall": float(avg_ranks[i]),
                "ranks": {k: float(ranks[k][i]) for k in keys},
                "zscores": {k: float(zscores[k][i]) for k in keys},
                "non_discriminating": list(non_discriminating)}
            for i, c in enumerate(clip_names)}

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


def detect_frame_rate(filepath):
    """Auto-detect frame rate from a video file using ffprobe.

    Returns frame rate as a string fraction (e.g. '24000/1001' or '60000/1001').
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: ffprobe failed for {filepath}, defaulting to 24000/1001")
        return "24000/1001"

    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if streams:
        return streams[0].get("r_frame_rate", "24000/1001")
    return "24000/1001"

# =====================================================================
# FRAME I/O
# =====================================================================

def decode_command(filepath):
    """Build ffmpeg decode command that outputs raw yuv422p10le to pipe."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", filepath,
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "pipe:1"
    ]


def encode_command(filepath, width, height, frame_rate):
    """Build ffmpeg encode command that reads raw yuv422p10le from pipe."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "-s", f"{width}x{height}",
        "-r", frame_rate,
        "-i", "pipe:0",
        "-c:v", "prores_ks", "-profile:v", "3",  # ProRes 422 HQ
        "-pix_fmt", "yuv422p10le",
        "-vendor", "apl0",
        filepath
    ]


def read_frame(pipe, width, height):
    """Read one raw YUV422p10le frame from pipe. Returns (Y, U, V) or None at EOF."""
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


def write_frame(pipe, y, u, v):
    """Write one raw YUV422p10le frame to pipe."""
    pipe.write(y.tobytes())
    pipe.write(u.tobytes())
    pipe.write(v.tobytes())
