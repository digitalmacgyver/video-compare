# Video Compare Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 1832-line `quality_report.py` into focused modules, deduplicate shared CSS, fix stale docstrings and a double-wait bug, and remove repository cruft — all without changing observable CLI behavior or any metric value.

**Architecture:** Three-layer module split: `common.py` (constants, ffmpeg helpers, composite ranking) ← `metrics.py` (NEW: pure metric functions, `analyze_clip`, `short_name`) ← `html_report.py` (NEW: HTML/CSS/text report rendering). Top-level CLIs (`quality_report.py`, `quality_metrics.py`, `cross_clip_report.py`) become thin orchestration. The synthetic test suite (`test_cases/test_metrics.py`) is the regression net — it must pass with byte-identical output before and after every task.

**Tech Stack:** Python 3.8+, NumPy, OpenCV (headless), SciPy, ffmpeg/ffprobe subprocess. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-09-codebase-cleanup-design.md`

**Baseline:** `python test_cases/test_metrics.py` reports `76/76 tests passed.` and `5/5 tests passed.` Capture this output before starting and check after every task.

---

## Conventions used in this plan

- Project root: `/home/viblio/coding_projects/video_compare/`. All paths below are relative to this root.
- Activate the venv before running anything: `source venv/bin/activate`. The venv may already be active in your shell — running `which python` should show `<root>/venv/bin/python`.
- After every task that touches Python code, run `python test_cases/test_metrics.py` and confirm both `76/76` and `5/5` pass blocks. Stop and investigate if any test fails or the count differs.
- Commits use Conventional Commits prefixes: `refactor:`, `fix:`, `docs:`, `chore:`.
- Each task ends with a commit. Frequent commits are intentional — they let `git bisect` localize any regression to one task.

---

## Task 1: Capture test baseline + extract `parse_skip_args` helper

**Why first:** Smallest possible refactor — moves a 10-line block of duplicated arg parsing into a shared helper. Validates the test loop works as a regression net before doing anything bigger.

**Files:**
- Modify: `common.py` — append a new function near the bottom.
- Modify: `quality_report.py:1657-1668` — replace inline `--skip` parse with helper call.
- Modify: `quality_metrics.py:67-78` — replace inline `--skip` parse with helper call.

- [ ] **Step 1: Run baseline tests and capture output**

```bash
cd /home/viblio/coding_projects/video_compare
source venv/bin/activate
python test_cases/test_metrics.py > /tmp/test_baseline.txt 2>&1
tail -5 /tmp/test_baseline.txt
```

Expected output ends with:
```
5/5 tests passed.
  All bug tests pass — Phase 2 fixes are in place.
```

If anything else, stop and surface the issue — the test suite is broken before we even start.

- [ ] **Step 2: Add `parse_skip_args` to `common.py`**

Append at the end of `common.py` (after `write_frame`):

```python
# =====================================================================
# CLI HELPERS
# =====================================================================

def parse_skip_args(skip_list):
    """Parse a list of '--skip PATTERN:N' strings into {pattern: n_frames}.

    Exits with status 1 on bad format. Used by quality_report.py and
    quality_metrics.py to ensure consistent --skip parsing.
    """
    import sys
    skip_patterns = {}
    for s in skip_list:
        if ":" not in s:
            print(f"ERROR: --skip format must be PATTERN:N, got '{s}'")
            sys.exit(1)
        pat, n = s.rsplit(":", 1)
        try:
            skip_patterns[pat] = int(n)
        except ValueError:
            print(f"ERROR: --skip N must be integer, got '{n}'")
            sys.exit(1)
    return skip_patterns
```

- [ ] **Step 3: Use helper in `quality_report.py`**

Replace lines 1657-1668 (the block starting `# Parse --skip arguments into {pattern: n_frames} dict`) with:

```python
    # Parse --skip arguments into {pattern: n_frames} dict
    skip_patterns = parse_skip_args(args.skip)
```

Update the import at the top of `quality_report.py` (around line 51) to include `parse_skip_args`:

```python
from common import (ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14,
                    compute_composites, probe_video, decode_command, read_frame,
                    parse_skip_args)
```

- [ ] **Step 4: Use helper in `quality_metrics.py`**

Replace lines 67-78 (the `# Parse --skip arguments` block) with:

```python
    skip_patterns = parse_skip_args(args.skip)
```

Update imports in `quality_metrics.py` (around line 38) to include `parse_skip_args`:

```python
from common import ALL_KEYS, METRIC_INFO, probe_video, parse_skip_args
```

- [ ] **Step 5: Run tests**

```bash
python test_cases/test_metrics.py 2>&1 | tail -5
```

Expected: `5/5 tests passed.` (same as baseline). If different, revert and investigate.

- [ ] **Step 6: Smoke-test CLI help**

```bash
python quality_report.py --help 2>&1 | grep -A1 -- "--skip"
python quality_metrics.py --help 2>&1 | grep -A1 -- "--skip"
```

Expected: both show their `--skip` help text. Just confirms imports resolve and argparse still loads.

- [ ] **Step 7: Commit**

```bash
git add common.py quality_report.py quality_metrics.py
git commit -m "refactor: extract --skip parsing into common.parse_skip_args

Removes byte-for-byte duplication between quality_report.py and
quality_metrics.py."
```

---

## Task 2: Create `metrics.py` and move metric functions

**Why:** This is the core architectural decoupling. After this task, `quality_metrics.py` no longer imports from `quality_report.py`. `test_cases/test_metrics.py` imports from `metrics` instead of `quality_report`.

**Files:**
- Create: `metrics.py`
- Modify: `quality_report.py` — delete moved code, import from `metrics`.
- Modify: `quality_metrics.py` — import from `metrics` instead of `quality_report`.
- Modify: `test_cases/test_metrics.py:18` — import from `metrics` instead of `quality_report`.

What moves to `metrics.py`:
- Constants: `DETAIL_PERCEPTUAL_KEY`, `DETAIL_PERCEPTUAL_DEPS`, `DERIVED_KEYS`, `EXTRA_DETAIL_KEYS`, `PCT_KEYS`
- Helpers: `to_float`, `parse_metric_csv`, `_zscore_with_params`, `short_name`
- All metric functions: `sharpness_laplacian`, `edge_strength_sobel`, `blocking_artifact_measure`, `detail_texture`, `detail_tenengrad`, `detail_sml`, `detail_blur_inv`, `texture_quality_measure`, `ringing_measure`, `colorfulness_metric`, `crushed_blacks`, `blown_whites`, `naturalness_nss`
- `add_detail_perceptual_metric`
- `analyze_clip`

What stays in `quality_report.py`:
- `CMP_ICON_SVG` (presentation — stays for now, will move to `html_report.py` in Task 3)
- `METRIC_GUIDE_TEXT` (presentation — stays for now, will move in Task 3)
- `fmt_metric_value`, `short_metric_header` (formatting — Task 3)
- `extract_frame_jpeg`, `find_comparison_frames`, `extract_sample_screenshots`
- `format_text_report`, `HTML_CSS`, `generate_html`
- `load_and_merge_jsons`, `main`

- [ ] **Step 1: Create `metrics.py` with all moved code**

Write `/home/viblio/coding_projects/video_compare/metrics.py`:

```python
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

from common import METRIC_INFO, decode_command, read_frame


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
```

- [ ] **Step 2: Delete the moved code from `quality_report.py`**

Open `quality_report.py` and delete:
- Lines 64-67 (the `to_float` function — wait, those line numbers shift. Use exact text matching.)

Use `Edit` to remove these blocks one at a time. Each block has unique enough text to match without needing line numbers:

1. The `to_float` function (4 lines starting `def to_float`):

```python
def to_float(y):
    """Convert 10-bit uint16 Y plane to float64 [0, 1]."""
    return y.astype(np.float64) / 1023.0
```

2. The constants block:

```python
DETAIL_PERCEPTUAL_KEY = "detail_perceptual"
DETAIL_PERCEPTUAL_DEPS = ("detail_blur_inv", "detail_sml", "texture_quality")
DERIVED_KEYS = {DETAIL_PERCEPTUAL_KEY}
EXTRA_DETAIL_KEYS = ["detail_tenengrad", "detail_sml", "detail_blur_inv", DETAIL_PERCEPTUAL_KEY]
PCT_KEYS = {"crushed_blacks", "blown_whites"}
```

3. `parse_metric_csv` (5 lines).

4. The entire `# TECHNICAL METRICS` section (starts at `# =====` comment block + `def sharpness_laplacian`) through and including `def naturalness_nss(yf)` and the `def _zscore_with_params` and `def add_detail_perceptual_metric` definitions.

5. The `# CLIP NAMING` comment block and `def short_name`.

6. The `# CLIP ANALYSIS` comment block and `def analyze_clip`.

After deletion `quality_report.py` should still contain: `CMP_ICON_SVG`, `METRIC_GUIDE_TEXT`, `fmt_metric_value`, `short_metric_header`, `extract_frame_jpeg`, `find_comparison_frames`, `extract_sample_screenshots`, `format_text_report`, `HTML_CSS`, `generate_html`, `load_and_merge_jsons`, `main`.

- [ ] **Step 3: Update `quality_report.py` imports**

At the top of `quality_report.py`, replace the existing `from common import` line with these two:

```python
from common import (ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14,
                    compute_composites, probe_video, parse_skip_args)
from metrics import (
    DETAIL_PERCEPTUAL_KEY, DETAIL_PERCEPTUAL_DEPS, DERIVED_KEYS,
    EXTRA_DETAIL_KEYS, PCT_KEYS,
    parse_metric_csv, short_name, analyze_clip, add_detail_perceptual_metric,
)
```

(Note: `decode_command` and `read_frame` were imported from `common` for use inside `analyze_clip`. They're now used only in `metrics.py`, so remove them from the `quality_report.py` import.)

Also remove the now-unused `cv2` and `numpy as np` imports if they aren't used by anything still in the file. Search the remaining file for `cv2.` and `np.` usage to decide. Likely both are still used in `find_comparison_frames` / `format_text_report` etc. — check before deleting.

- [ ] **Step 4: Update `quality_metrics.py` imports**

Replace the import block (lines ~38-43) currently importing from `quality_report`:

```python
from common import ALL_KEYS, METRIC_INFO, probe_video, parse_skip_args
from metrics import (
    EXTRA_DETAIL_KEYS, DETAIL_PERCEPTUAL_KEY, DETAIL_PERCEPTUAL_DEPS, DERIVED_KEYS,
    parse_metric_csv, short_name, analyze_clip, add_detail_perceptual_metric,
)
```

Note: `quality_metrics.py` no longer imports anything from `quality_report`.

- [ ] **Step 5: Update `test_cases/test_metrics.py` import**

Edit `test_cases/test_metrics.py` line 18 (the `from quality_report import ...` block). Change `quality_report` to `metrics`:

```python
from metrics import (
    sharpness_laplacian,
    edge_strength_sobel,
    blocking_artifact_measure,
    detail_texture,
    detail_tenengrad,
    detail_sml,
    detail_blur_inv,
    ...
)
```

(Preserve every name in the original import list — only the module name changes.)

- [ ] **Step 6: Run tests**

```bash
python test_cases/test_metrics.py 2>&1 | tail -5
```

Expected: `5/5 tests passed.` (same as baseline).

If any test fails, the most likely cause is a missing name in `metrics.py` — check the test file's import list against `metrics.py`.

- [ ] **Step 7: Smoke-test all CLIs still load**

```bash
python quality_report.py --help > /dev/null && echo "quality_report OK"
python quality_metrics.py --help > /dev/null && echo "quality_metrics OK"
python cross_clip_report.py --help > /dev/null && echo "cross_clip_report OK"
```

Expected: three OK lines. Any failure means an import broke.

- [ ] **Step 8: Verify the CLI behavior actually works on a tiny synthetic clip**

```bash
python -c "
from metrics import analyze_clip, short_name, sharpness_laplacian
from common import ALL_KEYS, METRIC_INFO
import numpy as np
y = (np.random.rand(64, 64) * 1023).astype(np.uint16)
print('sharpness_laplacian on synthetic Y:', sharpness_laplacian(y.astype(np.float64) / 1023.0))
print('short_name:', short_name('/tmp/twister_foo_trimmed_VIEW_ACTION_SAFE_IVTC_normalized.mov'))
"
```

Expected: prints a small float and `foo`.

- [ ] **Step 9: Commit**

```bash
git add metrics.py quality_report.py quality_metrics.py test_cases/test_metrics.py
git commit -m "refactor: extract metric functions into metrics.py

Pure metric functions, analyze_clip, short_name, and related
constants now live in metrics.py. quality_metrics.py no longer
imports from quality_report.py — both top-level scripts now
share the metric layer cleanly via metrics.py.

test_cases/test_metrics.py imports updated. Behavior unchanged:
synthetic test suite passes 76/76 + 5/5 with same numbers."
```

---

## Task 3: Create `html_report.py` and move HTML/text rendering

**Why:** Reduces `quality_report.py` from ~1100 lines to ~400. The HTML template lives in one place and `cross_clip_report.py` can import the shared CSS in Task 4.

**Files:**
- Create: `html_report.py`
- Modify: `quality_report.py` — delete moved code, import from `html_report`.

What moves to `html_report.py`:
- `CMP_ICON_SVG`, `METRIC_GUIDE_TEXT`, `HTML_CSS`
- `fmt_metric_value`, `short_metric_header`
- `extract_frame_jpeg`, `find_comparison_frames`, `extract_sample_screenshots`
- `format_text_report`
- `generate_html`

`fmt_metric_value` references `PCT_KEYS` from `metrics.py`. Import it.

- [ ] **Step 1: Create `html_report.py` with the moved code**

Write `/home/viblio/coding_projects/video_compare/html_report.py`. Copy the moved chunks from `quality_report.py` verbatim and add an appropriate header and imports:

```python
"""HTML, text, and screenshot rendering for the quality report.

Generates the self-contained HTML report (with embedded CSS, Chart.js
references, comparison frames, and sample screenshots) and the
optional plain-text summary. Frame extraction via ffmpeg lives here
because it's only used for HTML output.

Pure presentation — no metric algorithms.
"""

import base64
import json
import re
import subprocess
import numpy as np

from common import ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14, compute_composites
from metrics import PCT_KEYS


CMP_ICON_SVG = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<rect x="2" y="3" width="20" height="18" rx="2"/>'
    '<line x1="12" y1="3" x2="12" y2="21"/>'
    '<text x="7" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">A</text>'
    '<text x="17" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">B</text></svg>')


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
    "colorfulness": "Hasler-Süstrunk — color vibrancy",
    "naturalness": "MSCN kurtosis — natural signal statistics",
    "crushed_blacks": "Shadow headroom — clipped near-black fraction",
    "blown_whites": "Highlight headroom — clipped near-white fraction",
}


# =====================================================================
# FORMATTING
# =====================================================================

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
# FRAME EXTRACTION (for HTML comparison galleries)
# =====================================================================

# [Move extract_frame_jpeg, find_comparison_frames, extract_sample_screenshots
#  verbatim from quality_report.py here]


# =====================================================================
# TEXT REPORT
# =====================================================================

# [Move format_text_report verbatim from quality_report.py here]


# =====================================================================
# HTML REPORT
# =====================================================================

# HTML_CSS = """ ... """ — moved verbatim from quality_report.py

# [Move HTML_CSS and generate_html verbatim here]
```

**Don't actually leave the `[Move ... verbatim]` comments — copy the actual code.** Use the existing `quality_report.py` content as the source. After this step, `html_report.py` should contain:

- The header and imports above
- `CMP_ICON_SVG`, `METRIC_GUIDE_TEXT` (verbatim)
- `short_metric_header`, `fmt_metric_value` (verbatim — they reference `PCT_KEYS` which is imported from `metrics`)
- `extract_frame_jpeg`, `find_comparison_frames`, `extract_sample_screenshots` (verbatim)
- `format_text_report` (verbatim)
- `HTML_CSS` (the multi-line string verbatim)
- `generate_html` (verbatim)

- [ ] **Step 2: Delete moved code from `quality_report.py`**

Remove from `quality_report.py`:
- `CMP_ICON_SVG`
- `METRIC_GUIDE_TEXT`
- `short_metric_header`
- `fmt_metric_value`
- `extract_frame_jpeg`
- `find_comparison_frames`
- `extract_sample_screenshots`
- `format_text_report`
- `HTML_CSS`
- `generate_html`

Best done with `Edit` blocks matched on the `def name(...)` line plus enough surrounding context to be unique. Section comment headers (`# === HTML REPORT ===` etc.) can be deleted with their sections.

What remains in `quality_report.py` after this step:
- Module docstring and shebang
- Imports
- `load_and_merge_jsons`
- `main`
- The `if __name__ == "__main__"` guard

This should be ~250-350 lines.

- [ ] **Step 3: Update `quality_report.py` imports**

Add to the top of `quality_report.py`:

```python
from html_report import generate_html, format_text_report, find_comparison_frames, extract_sample_screenshots
```

Remove imports that were only used by the moved functions:
- `import base64` (only used in `extract_frame_jpeg`)
- `import re` (only used in `short_metric_header`)
- `COLORS_7`, `COLORS_14` (only used in `generate_html`)
- `import cv2`, `import numpy as np` — check; if no remaining code in `quality_report.py` uses them, remove. (`load_and_merge_jsons` doesn't; `main` might via passthrough — check.)

Actual final import block at top of `quality_report.py`:

```python
import argparse
import os
import sys
import glob
import json
from datetime import datetime

from common import ALL_KEYS, METRIC_INFO, parse_skip_args
from metrics import (
    DETAIL_PERCEPTUAL_KEY, DETAIL_PERCEPTUAL_DEPS, EXTRA_DETAIL_KEYS,
    parse_metric_csv, short_name, analyze_clip, add_detail_perceptual_metric,
)
from html_report import generate_html, format_text_report, find_comparison_frames, extract_sample_screenshots
```

Note `compute_composites` may still be referenced in `format_text_report` (yes — it is). That import moves with the function to `html_report.py`. So `quality_report.py` doesn't import `compute_composites` directly anymore.

- [ ] **Step 4: Run tests**

```bash
python test_cases/test_metrics.py 2>&1 | tail -5
```

Expected: `5/5 tests passed.`

- [ ] **Step 5: Smoke-test CLIs**

```bash
python quality_report.py --help > /dev/null && echo "quality_report OK"
python quality_metrics.py --help > /dev/null && echo "quality_metrics OK"
python cross_clip_report.py --help > /dev/null && echo "cross_clip_report OK"
```

Expected: three OK lines.

- [ ] **Step 6: End-to-end smoke test on the existing example_report JSON**

This actually exercises `generate_html` against real data — there is a 19 KB pre-computed JSON we can feed through `--from-json` mode without needing video files.

```bash
python quality_report.py --from-json example_report/index.json --output-dir /tmp --name html_smoke 2>&1 | tail -3
```

Expected: prints `JSON: /tmp/html_smoke_<timestamp>.json` and `HTML: /tmp/html_smoke.html`. Then verify the HTML opens:

```bash
ls -la /tmp/html_smoke.html
head -c 500 /tmp/html_smoke.html
```

Expected: a non-empty HTML file beginning with `<!DOCTYPE html>`.

- [ ] **Step 7: Line count sanity check**

```bash
wc -l common.py metrics.py html_report.py quality_report.py quality_metrics.py cross_clip_report.py
```

Expected approximate counts: `common.py ~225`, `metrics.py ~370`, `html_report.py ~850`, `quality_report.py ~350`, `quality_metrics.py ~190`, `cross_clip_report.py ~370` (unchanged).

- [ ] **Step 8: Commit**

```bash
git add html_report.py quality_report.py
git commit -m "refactor: extract HTML/text rendering into html_report.py

quality_report.py is now ~350 lines of CLI orchestration.
HTML_CSS, generate_html, format_text_report, comparison-frame
extraction, and the screenshot logic live in html_report.py.

Behavior unchanged: synthetic test suite passes 76/76 + 5/5
and end-to-end --from-json mode produces a non-empty HTML file."
```

---

## Task 4: Deduplicate `HTML_CSS` between `cross_clip_report.py` and `html_report.py`

**Why:** `cross_clip_report.py` carries an 80-line near-clone of `HTML_CSS`. Replace it with an import.

**Files:**
- Modify: `cross_clip_report.py:31-70` — delete local `HTML_CSS`, import from `html_report`.

Note: the two CSS strings are not byte-identical — `cross_clip_report.py`'s copy is a subset (no lightbox / cmp-card / ab-overlay rules because it doesn't render those components). However, importing the larger `html_report.HTML_CSS` is harmless: unused selectors are dead CSS and won't affect rendering. The byte size of the cross-clip HTML grows by ~3 KB, which is negligible compared to the page itself.

- [ ] **Step 1: Update `cross_clip_report.py` import block**

At the top of `cross_clip_report.py` (around line 21), change the import to also pull `HTML_CSS`:

```python
from common import ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14, compute_composites
from html_report import HTML_CSS
```

- [ ] **Step 2: Delete the local `HTML_CSS` definition**

Delete lines 27-70 of `cross_clip_report.py` (the comment header and the multi-line `HTML_CSS = """..."""` block that reads:

```python
# =====================================================================
# CSS (shared dark theme)
# =====================================================================

HTML_CSS = """
  :root {
  ...
  .clip-tag { ... }
"""
```

The remaining `cross_clip_report.py` still references `HTML_CSS` in the `.format(css=HTML_CSS, ...)` call near line 296 — that now uses the imported one.

- [ ] **Step 3: Run tests**

```bash
python test_cases/test_metrics.py 2>&1 | tail -5
```

Expected: `5/5 tests passed.` (`cross_clip_report.py` isn't exercised by the test suite, but importing it triggers parsing.)

- [ ] **Step 4: End-to-end smoke test of cross_clip_report**

We need two JSON files. Synthesize a quick second one from the existing example by duplicating it:

```bash
cp example_report/index.json /tmp/clip_a.json
cp example_report/index.json /tmp/clip_b.json
python cross_clip_report.py /tmp/clip_a.json /tmp/clip_b.json --output /tmp/crossclip_smoke.html 2>&1 | tail -3
```

Expected: prints `Output: /tmp/crossclip_smoke.html`. Verify:

```bash
ls -la /tmp/crossclip_smoke.html
head -c 200 /tmp/crossclip_smoke.html
grep -c "var(--bg)" /tmp/crossclip_smoke.html
```

Expected: non-empty file, starts with `<!DOCTYPE html>`, and `grep -c` returns ≥1 (proves the imported CSS is embedded).

- [ ] **Step 5: Commit**

```bash
git add cross_clip_report.py
git commit -m "refactor: import HTML_CSS from html_report instead of duplicating

cross_clip_report.py used a near-identical 80-line copy of the
HTML_CSS block. Now imports the canonical version. Unused
component selectors (lightbox, A/B slider) are dead CSS — harmless."
```

---

## Task 5: Fix double `proc.wait()` in `normalize.py`

**Why:** Two functions in `normalize.py` call `proc.wait()` once inside the success path and once in the `finally` block after `proc.kill()`. The success-path call is redundant and the wrong pattern; the `finally` block handles both clean and exceptional shutdown.

**Files:**
- Modify: `normalize.py:63` — remove inner `proc.wait()` from `compute_reference_stats`.
- Modify: `normalize.py:125` — remove inner `proc_r.wait()` from `normalize_clip` (first pass).

Note: the second-pass section in `normalize_clip` (lines 171-173) has `proc_dec.wait()` and `proc_enc.wait()` inside `try` — these are needed because the `finally` block calls `kill` on both, and we need the encoder to finish writing before we close stdin. The pattern here is different (writer/reader pair) so we leave it alone. **Only remove the two single-pass `wait()` calls.**

- [ ] **Step 1: Remove redundant wait in `compute_reference_stats`**

Edit `normalize.py`. The function currently looks like (around line 39):

```python
def compute_reference_stats(ref_path, width, height):
    ...
    try:
        ...
        while True:
            ...
            n_frames += 1

        proc.wait()           # <-- LINE 63: REMOVE THIS
    finally:
        proc.stdout.close()
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()
```

Replace the snippet around line 63 (the two lines `        proc.wait()` and the blank line before `    finally:`) with just a blank line before `    finally:`:

```python
            n_frames += 1
    finally:
```

Be careful — there are *two* lines like `        proc.wait()` in the file (the redundant one and the correct one in `finally`). The redundant one is followed by `    finally:` after a blank line. The `finally` one is preceded by `pass` and followed by the function's last expression.

- [ ] **Step 2: Remove redundant wait in `normalize_clip` first pass**

Same pattern, around line 125 in the first-pass `compute_mean_y`-style block:

```python
        proc_r.wait()         # <-- REMOVE THIS
    finally:
        proc_r.stdout.close()
        try:
            proc_r.kill()
        except OSError:
            pass
        proc_r.wait()
```

Becomes:

```python
            n_frames += 1
    finally:
        proc_r.stdout.close()
```

- [ ] **Step 3: Verify `normalize.py` still parses**

```bash
python -c "import normalize; print('normalize imports OK')"
```

Expected: `normalize imports OK`.

- [ ] **Step 4: Run tests**

```bash
python test_cases/test_metrics.py 2>&1 | tail -5
```

Expected: `5/5 tests passed.` (Tests don't exercise normalize, but ensure nothing else broke.)

- [ ] **Step 5: Commit**

```bash
git add normalize.py
git commit -m "fix: remove redundant proc.wait() calls in normalize.py

The success-path wait() is unnecessary — the finally block already
calls kill() and wait(). Pattern matches the existing approach in
analyze_clip()."
```

---

## Task 6: Documentation fixes (11→14 metrics, EXTRA_DETAIL_KEYS comment)

**Why:** Stale docstrings claim "11 metrics" — actual default is 14.

**Files:**
- Modify: `quality_report.py` — module docstring (top of file).
- Modify: `test_cases/README.md:3` — "11 quality metrics" → "11 core quality metrics".
- Modify: `metrics.py` — add a comment near `EXTRA_DETAIL_KEYS`.

- [ ] **Step 1: Update `quality_report.py` module docstring**

The current docstring lines (around line 25-29) say:

```
Metrics (11 total, all brightness-agnostic):
  Sharpness, edge strength, blocking, detail, texture quality, ringing, temporal stability,
  colorfulness, naturalness, crushed blacks, blown whites
```

Replace with:

```
Metrics (14 by default — 11 core + 3 extra detail; all brightness-agnostic):
  Core: sharpness, edge strength, blocking, detail, texture quality, ringing,
        temporal stability, colorfulness, naturalness, crushed blacks, blown whites
  Extra detail (default on): detail_perceptual (VHS-SD), detail_blur_inv, detail_sml
  Extra detail (opt-in via --extra-detail-metrics): detail_tenengrad
```

- [ ] **Step 2: Update `test_cases/README.md`**

Line 3 currently reads:

> Synthetic test images that verify each of the 11 quality metrics responds correctly to its target artifact, plus brightness invariance checks.

Change "11 quality metrics" to "11 core quality metrics + 3 extra detail metrics" and update the surrounding sentence so it scans cleanly:

> Synthetic test images that verify each of the 11 core quality metrics (and the perceptual detail derived metric) responds correctly to its target artifact, plus brightness invariance checks.

- [ ] **Step 3: Add clarifying comment near `EXTRA_DETAIL_KEYS` in `metrics.py`**

The comment block already exists from Task 2:

```python
# Available extra-detail metric keys. Only the subset listed in
# --extra-detail-metrics (default: detail_perceptual,detail_blur_inv,detail_sml)
# is computed by default; detail_tenengrad is available but opt-in.
EXTRA_DETAIL_KEYS = ["detail_tenengrad", "detail_sml", "detail_blur_inv", DETAIL_PERCEPTUAL_KEY]
```

If Task 2 was followed verbatim this is already in place — nothing to do. Otherwise, add it.

- [ ] **Step 4: Verify**

```bash
grep -n "metrics" quality_report.py | head -3
head -3 test_cases/README.md
```

Expected: docstring shows "14 by default" and README says "11 core".

- [ ] **Step 5: Commit**

```bash
git add quality_report.py test_cases/README.md metrics.py
git commit -m "docs: update metric counts to reflect 11 core + 3 extra detail default

quality_report.py module docstring and test_cases/README.md both
claimed '11 metrics'. The default is now 14 (11 core + 3 extra
detail metrics computed by default, plus detail_tenengrad opt-in)."
```

---

## Task 7: Move `review2/` to `docs/archive/review2/`

**Files:**
- Move: `review2/` → `docs/archive/review2/`

- [ ] **Step 1: Move with `git mv` to preserve history**

```bash
mkdir -p docs/archive
git mv review2 docs/archive/review2
```

- [ ] **Step 2: Verify**

```bash
ls docs/archive/review2/ | head
test ! -d review2 && echo "review2/ removed from root"
```

Expected: file listing for the moved dir, then "review2/ removed from root".

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: archive prior code-review docs under docs/archive/

review2/ contained design artifacts from the previous code-review
cycle (architecture.md, code_review.md, implementation_plan.md,
metric_implementation.md, metric_selection.md, project_plan.md,
plus baselines/). Decisions are already implemented in the
codebase. Moved to docs/archive/ to keep the repo root focused."
```

---

## Task 8: Remove `example_report/index.html` from main branch

**Files:**
- Delete: `example_report/index.html` (27 MB)
- Modify: `.gitignore` — add `example_report/*.html`

The README's GitHub Pages link (`digitalmacgyver.github.io/video-compare/example_report/`) is served from a separate `gh-pages` branch (or similar) and continues to work. We only stop committing the bloated file to the working tree.

**Sanity check first:** is there actually a separate Pages branch?

- [ ] **Step 1: Check Pages-branch setup**

```bash
git branch -a | grep -i pages
git remote -v
```

If no `gh-pages` branch exists, the README's link must be served from this repo's main branch. **In that case, stop and ask the user before deleting** — the live demo would break.

If a `gh-pages` (or `pages`) branch exists, proceed. Typical output: `remotes/origin/gh-pages`.

- [ ] **Step 2: Remove the file from the working tree**

```bash
git rm example_report/index.html
ls example_report/
```

Expected: only `index.json` remains in `example_report/`.

- [ ] **Step 3: Update `.gitignore`**

Append to `.gitignore`:

```
example_report/*.html
```

(`index.json` stays committed — it's small and useful as a sample input for `--from-json` smoke tests.)

- [ ] **Step 4: Verify**

```bash
cat .gitignore
git status
```

Expected: `.gitignore` shows the new line; `git status` shows `index.html` deleted and `.gitignore` modified.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: remove 27 MB example_report/index.html from main branch

The HTML file (base64-embedded screenshots) bloats clones and
git history. The live demo is served from the gh-pages branch;
the README link is unaffected. index.json is kept for use as a
sample --from-json input."
```

---

## Task 9: Tighten `.gitignore` and remove `.playwright-mcp/` cruft

**Files:**
- Modify: `.gitignore` — broaden cache rule, add generated outputs.
- Delete: `.playwright-mcp/` (local only — already gitignored)

- [ ] **Step 1: Replace `.gitignore` content**

Current `.gitignore`:

```
venv/
__pycache__/
*.pyc
.claude/
.playwright-mcp/
test_cases/assets/
example_report/*.html
```

Replace entirely with:

```
# Python
venv/
**/__pycache__/
*.pyc

# Editor / tool state
.claude/
.playwright-mcp/

# Generated test assets
test_cases/assets/

# Generated reports — kept out of the working tree
example_report/*.html
*_quality_report*.html
*_quality_report*.json
*_quality_metrics*.json
comparison.html
```

- [ ] **Step 2: Delete the stale `.playwright-mcp/` directory locally**

```bash
rm -rf .playwright-mcp
git status
```

Expected: `git status` shows `.gitignore` modified; `.playwright-mcp/` is not listed (it was already ignored).

- [ ] **Step 3: Run tests one last time**

```bash
python test_cases/test_metrics.py 2>&1 | tail -5
```

Expected: `5/5 tests passed.`

- [ ] **Step 4: Final repo overview**

```bash
ls -la
wc -l common.py metrics.py html_report.py quality_report.py quality_metrics.py cross_clip_report.py normalize.py
git status
git log --oneline -10
```

Expected: clean working tree (or only `.gitignore` modified pending commit), Python files at the expected sizes, recent commits show the refactor history.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: broaden .gitignore for caches and generated reports

- **/__pycache__/ to cover test_cases subdir
- ignore generated *_quality_report*.{html,json} outputs
- ignore *_quality_metrics*.json incremental outputs
- ignore comparison.html (cross_clip_report default name)"
```

---

## Final verification

After Task 9 commit:

- [ ] Run the synthetic test suite one more time:
  ```bash
  python test_cases/test_metrics.py 2>&1 | tail -10
  ```
  Expected: matches the baseline captured in Task 1, Step 1.

- [ ] Run all three CLIs' help:
  ```bash
  python quality_report.py --help > /dev/null && echo OK
  python quality_metrics.py --help > /dev/null && echo OK
  python cross_clip_report.py --help > /dev/null && echo OK
  python normalize.py --help > /dev/null && echo OK
  python normalize_linear.py --help > /dev/null && echo OK
  ```
  Expected: five OK lines.

- [ ] End-to-end report generation from JSON:
  ```bash
  python quality_report.py --from-json example_report/index.json --output-dir /tmp --name final_smoke 2>&1 | tail -3
  ls /tmp/final_smoke*.{html,json}
  ```
  Expected: HTML and JSON output files.

- [ ] `git log --oneline` should show 9 new commits, one per task.

If anything fails: do not hand-fix — `git bisect` to the offending task and revert just that commit.
