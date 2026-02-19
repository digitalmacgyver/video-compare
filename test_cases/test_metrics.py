#!/usr/bin/env python3
"""Synthetic metric validation — generates test images and verifies each metric
responds correctly to its target artifact.

Usage:
    python test_cases/test_metrics.py

Exit code 0 if all tests pass, 1 otherwise.
"""

import sys, os
import math
import numpy as np
import cv2

# Add project root to path so we can import from quality_report
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quality_report import (
    sharpness_laplacian,
    edge_strength_sobel,
    blocking_artifact_measure,
    detail_texture,
    texture_quality_measure,
    ringing_measure,
    colorfulness_metric,
    naturalness_nss,
    crushed_blacks,
    blown_whites,
    compute_composites,
    short_name,
    ALL_KEYS,
    METRIC_INFO,
)

SIZE = 512
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def save_png(name, img_float):
    """Save a [0,1] float image as 8-bit PNG for visual inspection."""
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    cv2.imwrite(path, np.clip(img_float * 255, 0, 255).astype(np.uint8))


def save_color_png(name, bgr_float):
    """Save a [0,1] float BGR image as 8-bit color PNG."""
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    cv2.imwrite(path, np.clip(bgr_float * 255, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Test image generators
# ---------------------------------------------------------------------------

def gen_sharp_scene():
    """Composite scene: step edges, diagonal line, checkerboard, zone plate."""
    img = np.zeros((SIZE, SIZE), dtype=np.float64)
    # Quadrant 1 (top-left): vertical + horizontal step edges
    img[:256, 64:128] = 0.8
    img[:256, 192:256] = 0.8
    img[64:128, :256] = np.maximum(img[64:128, :256], 0.6)
    # Quadrant 2 (top-right): diagonal lines
    for i in range(256):
        j = i
        if j < 256:
            img[i, 256 + j] = 0.9
            if j + 1 < 256:
                img[i, 256 + j + 1] = 0.9
    # Quadrant 3 (bottom-left): fine checkerboard
    checker = np.indices((256, 256)).sum(axis=0) % 2
    img[256:, :256] = checker.astype(np.float64) * 0.7 + 0.15
    # Quadrant 4 (bottom-right): zone plate (concentric rings)
    y, x = np.mgrid[:256, :256].astype(np.float64)
    r2 = (x - 128) ** 2 + (y - 128) ** 2
    img[256:, 256:] = 0.5 + 0.4 * np.cos(r2 / 80.0)
    return img


def gen_ringing_edges():
    """Clean bars with smoothed transitions for ringing test.

    Edges are softened with sigma=1 blur so the clean version has low
    near-edge Laplacian energy.  The ringing version adds explicit damped
    cosine oscillations near edges via distance transform.
    """
    img = np.full((SIZE, SIZE), 0.25, dtype=np.float64)
    img[:, 160:352] = 0.75
    img[160:352, :] = np.maximum(img[160:352, :], 0.75)
    # Smooth the step transitions so clean edges have low Laplacian
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def add_ringing_artifacts(clean):
    """Add explicit damped-cosine ringing near edges."""
    edges = cv2.Canny((clean * 255).astype(np.uint8), 50, 150)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    # Damped cosine — period/amplitude chosen so oscillation gradients
    # stay below Canny lower threshold (won't create false edges)
    period = 6.0
    decay_len = 10.0
    amplitude = 0.10
    osc = amplitude * np.cos(2 * np.pi * dist / period) * np.exp(-dist / decay_len)
    mask = (dist > 0) & (dist < 20)
    out = clean.copy()
    out[mask] += osc[mask]
    return np.clip(out, 0, 1)


def gen_smooth_gradient():
    """Smooth diagonal gradient (x+y)/(2*SIZE)."""
    y, x = np.mgrid[:SIZE, :SIZE].astype(np.float64)
    return (x + y) / (2.0 * SIZE)


def gen_texture_gratings():
    """Overlapping sinusoidal gratings — structured texture."""
    y, x = np.mgrid[:SIZE, :SIZE].astype(np.float64)
    g1 = np.sin(2 * np.pi * x / 12.0)
    g2 = np.sin(2 * np.pi * y / 16.0)
    g3 = np.sin(2 * np.pi * (x + y) / 20.0)
    img = 0.5 + 0.15 * g1 + 0.15 * g2 + 0.1 * g3
    return np.clip(img, 0, 1)


def gen_color_bars():
    """8-bar full-gamut color pattern. Returns BGR float [0,1]."""
    colors_rgb = [
        [1, 1, 1],  # white
        [1, 1, 0],  # yellow
        [0, 1, 1],  # cyan
        [0, 1, 0],  # green
        [1, 0, 1],  # magenta
        [1, 0, 0],  # red
        [0, 0, 1],  # blue
        [0, 0, 0],  # black
    ]
    bar_w = SIZE // 8
    bgr = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    for i, rgb in enumerate(colors_rgb):
        c0 = i * bar_w
        c1 = (i + 1) * bar_w if i < 7 else SIZE
        bgr[:, c0:c1, 0] = rgb[2]  # B
        bgr[:, c0:c1, 1] = rgb[1]  # G
        bgr[:, c0:c1, 2] = rgb[0]  # R
    return bgr


def rgb_to_ycbcr_planes(bgr_float):
    """Convert BGR float [0,1] to 10-bit Y, Cb, Cr uint16 planes.

    Uses BT.601 matrix (same as our pipeline for SD content).
    Cb/Cr centered at 512 (10-bit midpoint).
    Returns Y (full res), Cb (half-width), Cr (half-width) as uint16.
    """
    R = bgr_float[:, :, 2]
    G = bgr_float[:, :, 1]
    B = bgr_float[:, :, 0]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
    # Scale to 10-bit
    Y10 = np.clip(Y * 1023, 0, 1023).astype(np.uint16)
    Cb10 = np.clip(Cb * 1023 + 512, 0, 1023).astype(np.uint16)
    Cr10 = np.clip(Cr * 1023 + 512, 0, 1023).astype(np.uint16)
    # Subsample chroma to 4:2:2 (half width)
    Cb_sub = ((Cb10[:, 0::2].astype(np.float64) + Cb10[:, 1::2].astype(np.float64)) / 2).astype(np.uint16)
    Cr_sub = ((Cr10[:, 0::2].astype(np.float64) + Cr10[:, 1::2].astype(np.float64)) / 2).astype(np.uint16)
    return Y10, Cb_sub, Cr_sub


def gen_natural_scene():
    """Multi-scale random rectangles — sparse structure gives leptokurtic MSCN.

    Natural images have MSCN kurtosis > 0 because of sparse edges against
    flat/smooth regions.  This synthetic scene mimics that: random rectangles
    at varying scales create isolated edges with flat interiors.
    """
    rng = np.random.RandomState(42)
    img = np.full((SIZE, SIZE), 0.35, dtype=np.float64)
    # Large rectangles (background structure)
    for _ in range(15):
        w, h = rng.randint(80, 200, size=2)
        x, y = rng.randint(0, SIZE - w), rng.randint(0, SIZE - h)
        img[y:y+h, x:x+w] = rng.uniform(0.15, 0.85)
    # Medium rectangles (objects)
    for _ in range(30):
        w, h = rng.randint(20, 80, size=2)
        x, y = rng.randint(0, SIZE - w), rng.randint(0, SIZE - h)
        img[y:y+h, x:x+w] = rng.uniform(0.1, 0.9)
    # Small rectangles (fine detail)
    for _ in range(60):
        w, h = rng.randint(5, 25, size=2)
        x, y = rng.randint(0, SIZE - w), rng.randint(0, SIZE - h)
        img[y:y+h, x:x+w] = rng.uniform(0.1, 0.9)
    # Mild blur to soften pixel-perfect edges slightly
    return cv2.GaussianBlur(img, (0, 0), 0.8)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests():
    results = []

    # --- Test 1: Sharp vs. Blur → sharpness, edge_strength, detail ---
    sharp = gen_sharp_scene()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4.0)
    save_png("edges_sharp.png", sharp)
    save_png("edges_blurred.png", blurred)

    for metric_name, func in [("sharpness", sharpness_laplacian),
                               ("edge_strength", edge_strength_sobel),
                               ("detail", detail_texture)]:
        v_good = func(sharp)
        v_bad = func(blurred)
        passed = v_good > v_bad
        results.append(("Sharp vs. Blur", metric_name, v_good, v_bad, "higher", passed))

    # --- Test 2: Ringing ---
    clean_edges = gen_ringing_edges()
    ringing_edges = add_ringing_artifacts(clean_edges)
    save_png("edge_clean.png", clean_edges)
    save_png("edge_ringing.png", ringing_edges)

    v_good = ringing_measure(clean_edges)
    v_bad = ringing_measure(ringing_edges)
    passed = v_good < v_bad
    results.append(("Ringing", "ringing", v_good, v_bad, "lower", passed))

    # --- Test 3: Blocking ---
    gradient = gen_smooth_gradient()
    save_png("gradient_clean.png", gradient)
    # JPEG roundtrip at Q=5
    enc_ok, buf = cv2.imencode(".jpg", (gradient * 255).astype(np.uint8),
                                [cv2.IMWRITE_JPEG_QUALITY, 5])
    blocked = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    save_png("gradient_blocked.png", blocked)

    v_good = blocking_artifact_measure(gradient)
    v_bad = blocking_artifact_measure(blocked)
    passed = abs(v_good - 1.0) < abs(v_bad - 1.0)
    results.append(("Blocking", "blocking", v_good, v_bad, "~1.0", passed))

    # --- Test 4: Texture Quality ---
    tex_clean = gen_texture_gratings()
    rng = np.random.RandomState(99)
    tex_noisy = np.clip(tex_clean + rng.normal(0, 0.08, tex_clean.shape), 0, 1)
    save_png("texture_clean.png", tex_clean)
    save_png("texture_noisy.png", tex_noisy)

    v_good = texture_quality_measure(tex_clean)
    v_bad = texture_quality_measure(tex_noisy)
    passed = v_good > v_bad
    results.append(("Texture Quality", "texture_quality", v_good, v_bad, "higher", passed))

    # --- Test 5: Colorfulness ---
    color_bars = gen_color_bars()
    save_color_png("scene_colorful.png", color_bars)
    # Desaturated version: scale chroma to 25%
    _, cb_full, cr_full = rgb_to_ycbcr_planes(color_bars)
    _, cb_desat, cr_desat = rgb_to_ycbcr_planes(color_bars)
    cb_desat = (0.25 * (cb_desat.astype(np.float64) - 512) + 512).astype(np.uint16)
    cr_desat = (0.25 * (cr_desat.astype(np.float64) - 512) + 512).astype(np.uint16)
    # Save a visual reference of desaturated version
    # Reconstruct BGR from desaturated YCbCr for the PNG
    Y10, _, _ = rgb_to_ycbcr_planes(color_bars)
    Yf = Y10.astype(np.float64) / 1023.0
    Cb_up = np.repeat(cb_desat.astype(np.float64), 2, axis=1)[:, :SIZE]
    Cr_up = np.repeat(cr_desat.astype(np.float64), 2, axis=1)[:, :SIZE]
    Cbf = (Cb_up - 512) / 1023.0
    Crf = (Cr_up - 512) / 1023.0
    R = Yf + 1.402 * Crf
    G = Yf - 0.344136 * Cbf - 0.714136 * Crf
    B = Yf + 1.772 * Cbf
    desat_vis = np.stack([np.clip(B, 0, 1), np.clip(G, 0, 1), np.clip(R, 0, 1)], axis=-1)
    save_color_png("scene_desaturated.png", desat_vis)

    v_good = colorfulness_metric(cb_full, cr_full)
    v_bad = colorfulness_metric(cb_desat, cr_desat)
    passed = v_good > v_bad
    results.append(("Colorfulness", "colorfulness", v_good, v_bad, "higher", passed))

    # --- Test 6: Naturalness ---
    # Sparse structure → leptokurtic MSCN (kurtosis > 0).
    # Heavy Gaussian noise destroys the sparse structure, pushing MSCN
    # toward Gaussian (kurtosis → 0), i.e. lower value.
    natural = gen_natural_scene()
    save_png("natural_scene.png", natural)
    rng_nat = np.random.RandomState(77)
    noisy = np.clip(natural + rng_nat.normal(0, 0.20, natural.shape), 0, 1)
    save_png("natural_noisy.png", noisy)

    v_good = naturalness_nss(natural)
    v_bad = naturalness_nss(noisy)
    passed = v_good > v_bad
    results.append(("Naturalness", "naturalness", v_good, v_bad, "higher", passed))

    # --- Test 7: Temporal Stability ---
    # Uses brightness-normalized frame diffs (matching analyze_clip)
    rng = np.random.RandomState(7)
    base_pattern = gen_texture_gratings()  # reuse sinusoidal pattern
    n_frames = 30

    # Stable: tiny noise only
    stable_diffs = []
    prev = None
    for i in range(n_frames):
        frame = base_pattern + rng.normal(0, 0.001, base_pattern.shape)
        frame = np.clip(frame, 0, 1)
        if prev is not None:
            mean_y = 0.5 * (np.mean(frame) + np.mean(prev))
            stable_diffs.append(np.mean(np.abs(frame - prev)) / (mean_y + 1e-10))
        prev = frame

    # Flickering: sinusoidal brightness oscillation
    flicker_diffs = []
    prev = None
    for i in range(n_frames):
        brightness = 0.05 * np.sin(2 * np.pi * i / 5.0)
        frame = np.clip(base_pattern + brightness, 0, 1)
        if prev is not None:
            mean_y = 0.5 * (np.mean(frame) + np.mean(prev))
            flicker_diffs.append(np.mean(np.abs(frame - prev)) / (mean_y + 1e-10))
        prev = frame

    v_good = float(np.mean(stable_diffs))
    v_bad = float(np.mean(flicker_diffs))
    passed = v_good < v_bad
    results.append(("Temporal Stability", "temporal_stability", v_good, v_bad, "lower", passed))

    # --- Test 8: Crushed Blacks ---
    # Good: smooth gradient 0→1 (shadow pixels spread across full range)
    # Bad: same but all pixels below 0.10 clamped to 0.03 (piled at floor)
    gradient = gen_smooth_gradient()
    crushed_img = gradient.copy()
    crushed_img[crushed_img < 0.10] = 0.03
    save_png("shadows_spread.png", gradient)
    save_png("shadows_crushed.png", crushed_img)

    v_good = crushed_blacks(gradient)
    v_bad = crushed_blacks(crushed_img)
    passed = v_good < v_bad
    results.append(("Crushed Blacks", "crushed_blacks", v_good, v_bad, "lower", passed))

    # --- Test 9: Blown Whites ---
    # Good: smooth gradient 0→1 (highlight pixels spread across full range)
    # Bad: same but all pixels above 0.90 clamped to 0.97 (piled at ceiling)
    blown_img = gradient.copy()
    blown_img[blown_img > 0.90] = 0.97
    save_png("highlights_spread.png", gradient)
    save_png("highlights_blown.png", blown_img)

    v_good = blown_whites(gradient)
    v_bad = blown_whites(blown_img)
    passed = v_good < v_bad
    results.append(("Blown Whites", "blown_whites", v_good, v_bad, "lower", passed))

    # --- Test 10: Brightness Invariance ---
    # Verify brightness-agnostic metrics give the same result at different
    # brightness levels.  The sharp scene at 1.0x vs 0.3x should produce
    # nearly identical values for detail, sharpness, and edge_strength.
    dim = np.clip(sharp * 0.3, 0, 1)
    for metric_name, func in [("detail", detail_texture),
                               ("sharpness", sharpness_laplacian),
                               ("edge_strength", edge_strength_sobel)]:
        v_normal = func(sharp)
        v_dim = func(dim)
        # Allow 5% relative tolerance
        rel_diff = abs(v_normal - v_dim) / (abs(v_normal) + 1e-10)
        passed = rel_diff < 0.05
        results.append(("Brightness Inv.", metric_name, v_normal, v_dim, "~equal", passed))

    return results


# ---------------------------------------------------------------------------
# Phase 1.3: Baseline regression tests (should always pass)
# ---------------------------------------------------------------------------

# Luma metrics that take a single yf frame
LUMA_METRICS = [
    ("sharpness",       sharpness_laplacian),
    ("edge_strength",   edge_strength_sobel),
    ("blocking",        blocking_artifact_measure),
    ("detail",          detail_texture),
    ("texture_quality", texture_quality_measure),
    ("ringing",         ringing_measure),
    ("naturalness",     naturalness_nss),
    ("crushed_blacks",  crushed_blacks),
    ("blown_whites",    blown_whites),
]


def run_edge_case_tests():
    """Test all metrics with degenerate inputs: all-black, all-white, constant gray.

    Verifies: no exceptions, no NaN, no Inf for any metric.
    """
    results = []
    frames = {
        "all_black":     np.zeros((64, 64), dtype=np.float64),
        "all_white":     np.ones((64, 64), dtype=np.float64),
        "constant_gray": np.full((64, 64), 0.5, dtype=np.float64),
    }

    for frame_name, frame in frames.items():
        for metric_name, func in LUMA_METRICS:
            try:
                val = func(frame)
                ok = math.isfinite(val)
            except Exception as e:
                val = float('nan')
                ok = False
            results.append(("Edge: " + frame_name, metric_name, val, 0.0, "finite", ok))

        # Colorfulness with neutral chroma (512 = midpoint)
        cb_neutral = np.full((64, 32), 512, dtype=np.uint16)
        cr_neutral = np.full((64, 32), 512, dtype=np.uint16)
        try:
            val = colorfulness_metric(cb_neutral, cr_neutral)
            ok = math.isfinite(val)
        except Exception as e:
            val = float('nan')
            ok = False
        results.append(("Edge: " + frame_name, "colorfulness", val, 0.0, "finite", ok))

    return results


def run_brightness_invariance_tests():
    """Test brightness invariance for metrics not covered by the original suite.

    Original suite tests: detail, sharpness, edge_strength (multiplicative 0.3x).
    This adds: blocking, texture_quality.
    Ringing is excluded -- its Canny edge detection has known brightness dependence
    (8-bit quantization, documented in metric_implementation.md §6).
    """
    results = []

    # Blocking: use a textured scene with mild JPEG blocking (Q=30).
    # Q=5 is too extreme -- within-block diffs go to zero, breaking the
    # ratio metric's epsilon guard.  Q=30 has visible blocking but
    # non-zero within-block gradients, so the ratio cancels properly.
    tex_scene = gen_texture_gratings()
    enc_ok, buf = cv2.imencode(".jpg", (tex_scene * 255).astype(np.uint8),
                                [cv2.IMWRITE_JPEG_QUALITY, 30])
    blocked = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    blocked_dim = np.clip(blocked * 0.5, 0, 1)

    v_normal = blocking_artifact_measure(blocked)
    v_dim = blocking_artifact_measure(blocked_dim)
    rel_diff = abs(v_normal - v_dim) / (abs(v_normal) + 1e-10)
    # Blocking is a ratio metric -- inherently scale-invariant.
    # Use 5% tolerance.
    passed = rel_diff < 0.05
    results.append(("Bright. Inv.", "blocking", v_normal, v_dim, "~equal", passed))

    # Texture quality: structured gratings at two brightness levels
    tex = gen_texture_gratings()
    tex_dim = np.clip(tex * 0.5, 0, 1)
    v_normal = texture_quality_measure(tex)
    v_dim = texture_quality_measure(tex_dim)
    rel_diff = abs(v_normal - v_dim) / (abs(v_normal) + 1e-10)
    passed = rel_diff < 0.05
    results.append(("Bright. Inv.", "texture_quality", v_normal, v_dim, "~equal", passed))

    return results


def run_short_name_tests():
    """Test short_name() extracts device names correctly."""
    results = []

    cases = [
        # (input_path, expected_output)
        ("twister_jvctbc1_svideo_direct.mov",    "jvctbc1"),
        ("twister_ag1980p_svideo_sdi1.mov",      "ag1980p"),
        ("twister_es15_svideo_sdi.mov",           "es15"),
        ("device_trimmed_VIEW_ACTION_SAFE_IVTC.mov",  "device"),
        ("device_trimmed_VIEW_ACTION_SAFE_NOIVTC.mov", "device"),
        ("ag1980tbc0 machhd 02.mov",              "ag1980tbc0"),
        ("simple.mov",                             "simple"),
        ("twister_device_normalized.mov",          "device"),
    ]

    for input_path, expected in cases:
        actual = short_name(input_path)
        passed = actual == expected
        # Use the result tuple format: (test_name, metric_name, v_good, v_bad, expect, passed)
        # Repurpose fields: v_good=actual, v_bad=0 (unused), expect=expected
        results.append(("short_name", f"'{input_path[:25]}'", 0.0, 0.0, expected, passed))
        if not passed:
            print(f"  short_name('{input_path}') = '{actual}', expected '{expected}'")

    return results


# ---------------------------------------------------------------------------
# Phase 1.4: Bug-demonstrating tests (expected to FAIL before Phase 2 fixes)
# ---------------------------------------------------------------------------

def make_composite_data(clip_metrics):
    """Build a data dict suitable for compute_composites().

    clip_metrics: dict of {clip_name: {metric_key: value}}
    Returns dict of {clip_name: {metric_key: {"mean": value, "std": 0.0}}}
    """
    data = {}
    for clip_name, metrics in clip_metrics.items():
        data[clip_name] = {}
        for key in ALL_KEYS:
            val = metrics.get(key, 0.5)
            data[clip_name][key] = {"mean": val, "std": 0.0}
    return data


def run_known_bug_tests():
    """Tests for known bugs. These are expected to FAIL before Phase 2 fixes.

    After Phase 2, they should all PASS.
    """
    results = []

    # --- Bug test 1: Tied ranks should be equal ---
    # Two clips with identical sharpness should get the same rank.
    clip_metrics = {
        "clip_a": {"sharpness": 10.0, "edge_strength": 0.5, "blocking": 1.0,
                    "detail": 0.3, "texture_quality": 0.8, "ringing": 0.1,
                    "temporal_stability": 0.02, "colorfulness": 100.0,
                    "naturalness": 2.0, "crushed_blacks": 0.05, "blown_whites": 0.0},
        "clip_b": {"sharpness": 10.0, "edge_strength": 0.6, "blocking": 1.1,
                    "detail": 0.4, "texture_quality": 0.7, "ringing": 0.15,
                    "temporal_stability": 0.03, "colorfulness": 110.0,
                    "naturalness": 1.5, "crushed_blacks": 0.03, "blown_whites": 0.0},
        "clip_c": {"sharpness": 5.0, "edge_strength": 0.4, "blocking": 1.2,
                    "detail": 0.2, "texture_quality": 0.6, "ringing": 0.2,
                    "temporal_stability": 0.05, "colorfulness": 80.0,
                    "naturalness": 1.0, "crushed_blacks": 0.08, "blown_whites": 0.01},
    }
    data = make_composite_data(clip_metrics)
    composites = compute_composites(data)

    # clip_a and clip_b have identical sharpness (10.0) — they should get equal rank
    rank_a = composites["clip_a"]["ranks"]["sharpness"]
    rank_b = composites["clip_b"]["ranks"]["sharpness"]
    passed = rank_a == rank_b
    results.append(("Tied Ranks", "sharpness equal",
                    float(rank_a), float(rank_b), "equal", passed))

    # Both also have identical blown_whites (0.0) — should get equal rank
    rank_a_bw = composites["clip_a"]["ranks"]["blown_whites"]
    rank_b_bw = composites["clip_b"]["ranks"]["blown_whites"]
    passed = rank_a_bw == rank_b_bw
    results.append(("Tied Ranks", "blown_whites eq",
                    float(rank_a_bw), float(rank_b_bw), "equal", passed))

    # --- Bug test 2: All-equal values should all get same rank ---
    all_same = {
        "clip_x": {k: 5.0 for k in ALL_KEYS},
        "clip_y": {k: 5.0 for k in ALL_KEYS},
        "clip_z": {k: 5.0 for k in ALL_KEYS},
    }
    data_eq = make_composite_data(all_same)
    comp_eq = compute_composites(data_eq)

    # All three clips should have identical overall rank
    overall_x = comp_eq["clip_x"]["overall"]
    overall_y = comp_eq["clip_y"]["overall"]
    overall_z = comp_eq["clip_z"]["overall"]
    passed = (overall_x == overall_y == overall_z)
    results.append(("All-Equal", "overall rank",
                    overall_x, overall_y, "all same", passed))

    # Each clip should have rank 2.0 on every metric (average of 1,2,3)
    rank_x_sharp = comp_eq["clip_x"]["ranks"]["sharpness"]
    passed = (rank_x_sharp == 2.0)
    results.append(("All-Equal", "sharpness=2.0",
                    float(rank_x_sharp), 2.0, "=2.0", passed))

    # --- Bug test 3: Discrimination gating ---
    # Make a dataset where blown_whites is constant (0.0) for all clips
    # but other metrics vary.  blown_whites should be in non_discriminating.
    disc_metrics = {
        "clip_p": {"sharpness": 15.0, "edge_strength": 0.8, "blocking": 1.0,
                    "detail": 0.5, "texture_quality": 0.9, "ringing": 0.05,
                    "temporal_stability": 0.01, "colorfulness": 120.0,
                    "naturalness": 3.0, "crushed_blacks": 0.03, "blown_whites": 0.0},
        "clip_q": {"sharpness": 8.0, "edge_strength": 0.5, "blocking": 1.2,
                    "detail": 0.3, "texture_quality": 0.7, "ringing": 0.1,
                    "temporal_stability": 0.03, "colorfulness": 90.0,
                    "naturalness": 1.5, "crushed_blacks": 0.06, "blown_whites": 0.0},
        "clip_r": {"sharpness": 5.0, "edge_strength": 0.3, "blocking": 1.5,
                    "detail": 0.2, "texture_quality": 0.5, "ringing": 0.15,
                    "temporal_stability": 0.06, "colorfulness": 60.0,
                    "naturalness": 0.8, "crushed_blacks": 0.10, "blown_whites": 0.0},
    }
    disc_data = make_composite_data(disc_metrics)
    disc_comp = compute_composites(disc_data)
    non_disc = disc_comp["clip_p"]["non_discriminating"]
    passed = "blown_whites" in non_disc
    results.append(("Disc. Gating", "blown_whites excluded",
                    float(len(non_disc)), 1.0, "in list", passed))

    return results


# ---------------------------------------------------------------------------
# Output formatting and main
# ---------------------------------------------------------------------------

def fmt_val(v):
    """Format a metric value for the results table."""
    if abs(v) >= 1000:
        return f"{v:>14.2f}"
    return f"{v:>14.6f}"


def print_results(results, title="METRIC VALIDATION RESULTS"):
    w = 95
    print("=" * w)
    print(title)
    print("=" * w)
    hdr = f"{'Test':<23}{'Metric':<20}{'Good':>14}{'Bad':>14}   {'Expect':<8}{'Result'}"
    print(hdr)
    print("-" * w)
    n_pass = 0
    for test_name, metric, v_good, v_bad, expect, passed in results:
        tag = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        print(f"{test_name:<23}{metric:<20}{fmt_val(v_good)}{fmt_val(v_bad)}   {expect:<8}{tag}")
    print("-" * w)
    print(f"{n_pass}/{len(results)} tests passed.")
    return n_pass == len(results)


if __name__ == "__main__":
    # --- Main tests (must all pass) ---
    results = run_tests()
    results += run_edge_case_tests()
    results += run_brightness_invariance_tests()
    results += run_short_name_tests()
    all_pass = print_results(results)
    print(f"\nTest assets saved to: {ASSETS}")

    # --- Known-bug tests (expected to fail before Phase 2 fixes) ---
    print()
    bug_results = run_known_bug_tests()
    print_results(bug_results, title="KNOWN BUG TESTS (expected to fail before Phase 2)")
    bug_pass = sum(1 for r in bug_results if r[5])
    if bug_pass == len(bug_results):
        print("  All bug tests pass — Phase 2 fixes are in place.")
    else:
        print(f"  {len(bug_results) - bug_pass} expected failures — fixes pending (Phase 2).")

    # Exit code based only on main tests
    sys.exit(0 if all_pass else 1)
