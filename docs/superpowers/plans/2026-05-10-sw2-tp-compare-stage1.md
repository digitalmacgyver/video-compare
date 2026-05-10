# SW2 Test Pattern Comparison — Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 1 vertical slice of the SW2 NTSC test-pattern comparison tool: synthesize an ideal frame, register a captured frame against it via grid-intersection fiducials, sample top-left tartan + 4-step gray patches, and produce a per-capture JSON plus a multi-capture HTML comparison report.

**Architecture:** Three new sibling scripts at the `video_compare/` repo root (`tp_synthesize.py`, `tp_measure.py`, `tp_compare.py`) plus a shared constants/helpers module (`tp_chart.py`). Mirrors the existing `quality_metrics.py` → `cross_clip_report.py` pattern. Reuses `common.py` for ffmpeg/probe helpers. The synthesized ideal frame is YUV422p10le, BT.601 limited range, at 720×486; smaller captures are padded with grey. Registration uses 6–8 grid-line intersections on the grey background as fiducials with RANSAC affine fit.

**Tech Stack:** Python 3, NumPy, OpenCV (headless), ffmpeg subprocess for I/O, plain-Python test scripts matching the existing `test_cases/test_metrics.py` style.

**Reference materials:**
- Spec: `docs/superpowers/specs/2026-05-10-sw2-tp-compare-design.md`
- Codex starter (donor math + coordinate references): `/home/viblio/coding_projects/sw2_analysis/codex/analyze_tartan.py`
- Sample capture for end-to-end smoke: `/wintmp/analog_video/tp_compare/sample/sw2_dvdrip_sample.mov` (720×480 ProRes)
- Existing infra: `common.py` (read_frame, decode_command, probe_video)

**Conventions:**
- Tests are plain-Python scripts in `test_cases/`, runnable via `python test_cases/test_<name>.py`. Each script exits 0 on success, 1 on any test failure. Match the style of `test_cases/test_metrics.py`.
- All code uses NumPy `uint16` for 10-bit YUV422p planes (Y shape `(H, W)`, U and V shape `(H, W//2)`).
- Frequent commits — at the end of every task.

---

## File Structure

| Path | Purpose |
|---|---|
| `tp_chart.py` | SW2 constants: BT.601 math, ideal YUV values, region table, registration landmark catalog, version constant |
| `tp_synthesize.py` | Library + CLI: synthesize the ideal yuv422p10le frame |
| `tp_register.py` | Library: landmark detector + RANSAC affine fit + top-level `register()` |
| `tp_measure.py` | CLI: per-capture frame extraction, padding, registration, region sampling, JSON writer |
| `tp_compare.py` | CLI: multi-JSON → comparison HTML |
| `test_cases/test_tp_chart.py` | Tests for tp_chart constants and math |
| `test_cases/test_tp_synthesize.py` | Tests for the synthesizer |
| `test_cases/test_tp_register.py` | Tests for landmark detection + affine fit + register |
| `test_cases/test_tp_measure.py` | Tests for tp_measure end-to-end |
| `test_cases/test_tp_compare.py` | Tests for tp_compare HTML rendering |

`common.py`, `quality_report.py`, `quality_metrics.py`, `cross_clip_report.py`, `metrics.py`, `test_cases/test_metrics.py` are NOT modified.

---

## Task 1: `tp_chart.py` — BT.601 math + base constants

**Files:**
- Create: `tp_chart.py`
- Test: `test_cases/test_tp_chart.py`

- [ ] **Step 1: Write the failing test for math + constants**

Create `test_cases/test_tp_chart.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_chart constants and YUV math."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tp_chart


def approx(actual, expected, tol):
    return abs(actual - expected) <= tol


def test_constants_exist():
    assert tp_chart.TP_CHART_VERSION == 1
    assert tp_chart.BLACK_Y10 == 64
    assert tp_chart.WHITE_Y10 == 940
    assert tp_chart.CHROMA_CENTER == 512
    # Grey background = 50% IRE in BT.601 limited range = 64 + 0.5*876 = 502
    assert tp_chart.GREY_BACKGROUND_Y10 == 502
    # 4-step grays at 20/40/60/80% IRE
    assert tp_chart.GRAY_IDEAL_Y10 == (239.2, 414.4, 589.6, 764.8)


def test_rgb_norm_to_yuv10_black():
    y, u, v = tp_chart.rgb_norm_to_yuv10(0.0, 0.0, 0.0)
    assert approx(y, 64.0, 0.1)
    assert approx(u, 512.0, 0.1)
    assert approx(v, 512.0, 0.1)


def test_rgb_norm_to_yuv10_white():
    y, u, v = tp_chart.rgb_norm_to_yuv10(1.0, 1.0, 1.0)
    assert approx(y, 940.0, 0.1)
    assert approx(u, 512.0, 0.1)
    assert approx(v, 512.0, 0.1)


def test_rgb_norm_to_yuv10_75_yellow():
    # R=G=0.75, B=0 -> Y = 0.299*0.75 + 0.587*0.75 = 0.6645
    # Y10 = 64 + 876 * 0.6645 = 646.1
    y, u, v = tp_chart.rgb_norm_to_yuv10(0.75, 0.75, 0.0)
    assert approx(y, 646.1, 0.5)
    assert approx(u, 176.0, 1.0)   # 512 + 896 * 0.5 * (-0.6645)/0.886
    assert approx(v, 566.7, 1.0)   # 512 + 896 * 0.5 * (0.75 - 0.6645)/0.701


def test_yuv10_to_rgb8_round_trip():
    y, u, v = tp_chart.rgb_norm_to_yuv10(0.5, 0.5, 0.5)
    r8, g8, b8 = tp_chart.yuv10_to_rgb8(y, u, v)
    assert approx(r8, 128, 1)
    assert approx(g8, 128, 1)
    assert approx(b8, 128, 1)


TESTS = [
    test_constants_exist,
    test_rgb_norm_to_yuv10_black,
    test_rgb_norm_to_yuv10_white,
    test_rgb_norm_to_yuv10_75_yellow,
    test_yuv10_to_rgb8_round_trip,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(TESTS)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_chart.py`

Expected: `ModuleNotFoundError: No module named 'tp_chart'`

- [ ] **Step 3: Implement `tp_chart.py` with constants + math**

Create `tp_chart.py`:

```python
"""Snell & Wilcox Test Chart #2 (NTSC) — constants, math, region table.

Used by tp_synthesize, tp_measure, tp_compare. Treats the ideal frame as
yuv422p10le, BT.601 limited range, at the SDI active raster (720x486).

References:
- TPG20/21 manual (PAL, page 4.25); NTSC differences per page 4.30 + the
  "What is it?" PDF.
- Existing codex starter: /home/viblio/coding_projects/sw2_analysis/codex
"""

from __future__ import annotations
from typing import Tuple

TP_CHART_VERSION = 1

# BT.601 limited-range 10-bit
BLACK_Y10 = 64
WHITE_Y10 = 940
CHROMA_CENTER = 512
Y_RANGE = WHITE_Y10 - BLACK_Y10  # 876

# Grey background level on the SW2 chart: 50% IRE per spec.
GREY_BACKGROUND_Y10 = 502  # 64 + 0.5 * 876 = 502.0, rounded to 10-bit code

# 4-step grayscale below the tartan: 20%, 40%, 60%, 80% IRE.
GRAY_IDEAL_Y10: Tuple[float, float, float, float] = (
    BLACK_Y10 + 0.20 * Y_RANGE,  # 239.2
    BLACK_Y10 + 0.40 * Y_RANGE,  # 414.4
    BLACK_Y10 + 0.60 * Y_RANGE,  # 589.6
    BLACK_Y10 + 0.80 * Y_RANGE,  # 764.8
)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def rgb_norm_to_yuv10(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Rec.601 limited-range R'G'B' (each in [0,1]) -> 10-bit YUV422 codes."""
    kr = 0.299
    kb = 0.114
    kg = 1.0 - kr - kb  # 0.587
    y = kr * r + kg * g + kb * b
    cb = 0.5 * (b - y) / (1.0 - kb)
    cr = 0.5 * (r - y) / (1.0 - kr)
    return (
        BLACK_Y10 + Y_RANGE * y,
        CHROMA_CENTER + 896.0 * cb,
        CHROMA_CENTER + 896.0 * cr,
    )


def yuv10_to_rgb8(y10: float, u10: float, v10: float) -> Tuple[int, int, int]:
    """10-bit BT.601 limited-range YUV -> 8-bit sRGB-ish (for swatch display)."""
    y = (y10 - BLACK_Y10) / Y_RANGE
    cb = (u10 - CHROMA_CENTER) / 896.0
    cr = (v10 - CHROMA_CENTER) / 896.0
    r = _clamp01(y + 1.402 * cr)
    g = _clamp01(y - 0.344136 * cb - 0.714136 * cr)
    b = _clamp01(y + 1.772 * cb)
    return (round(r * 255.0), round(g * 255.0), round(b * 255.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_chart.py`

Expected:
```
PASS  test_constants_exist
PASS  test_rgb_norm_to_yuv10_black
PASS  test_rgb_norm_to_yuv10_white
PASS  test_rgb_norm_to_yuv10_75_yellow
PASS  test_yuv10_to_rgb8_round_trip

All 5 tests passed
```

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(sw2): tp_chart base constants and BT.601 YUV math"
```

---

## Task 2: `tp_chart.py` — Region table + landmark catalog

**Files:**
- Modify: `tp_chart.py`
- Modify: `test_cases/test_tp_chart.py`

- [ ] **Step 1: Add failing tests for region table + landmark catalog**

Append to `test_cases/test_tp_chart.py` (insert before the `TESTS = [...]` line):

```python
def test_tartan_regions_count_and_ids():
    ids = [r["id"] for r in tp_chart.TARTAN_REGIONS]
    assert len(ids) == 8
    assert set(ids) == {"YEL", "CYN", "BLU", "RED",
                        "MAG_L", "GRN_L", "RED_L", "CYN_L"}


def test_tartan_region_record_shape():
    for r in tp_chart.TARTAN_REGIONS:
        assert "id" in r and "name" in r and "kind" in r
        assert r["kind"] == "tartan_rect"
        x, y, w, h = r["ideal_box"]
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= 720
        assert 0 <= y and y + h <= 486
        assert "expected" in r
        assert "y10" in r["expected"] and "u10" in r["expected"] and "v10" in r["expected"]
        assert r["sample"] == {"kind": "center_window", "size_frac": 0.2}


def test_tartan_75_yellow_expected():
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    e = yel["expected"]
    assert approx(e["y10"], 646.1, 0.5)
    assert approx(e["u10"], 176.0, 1.0)
    assert approx(e["v10"], 566.7, 1.0)


def test_gray_regions():
    assert len(tp_chart.GRAY_REGIONS) == 4
    ids = [r["id"] for r in tp_chart.GRAY_REGIONS]
    assert ids == ["G1", "G2", "G3", "G4"]
    for r, expected_y in zip(tp_chart.GRAY_REGIONS, tp_chart.GRAY_IDEAL_Y10):
        assert r["kind"] == "gray_step"
        assert approx(r["expected"]["y10"], expected_y, 0.05)
        assert r["expected"]["u10"] == 512
        assert r["expected"]["v10"] == 512


def test_grid_landmarks():
    lms = tp_chart.GRID_LANDMARKS
    assert len(lms) >= 6
    for lm in lms:
        assert "id" in lm
        assert 0 < lm["ideal_x"] < 720
        assert 0 < lm["ideal_y"] < 486
        assert lm["search_window_px"] >= 16


def test_grid_landmark_distribution():
    # At least one landmark in each vertical third of the picture
    ys = [lm["ideal_y"] for lm in tp_chart.GRID_LANDMARKS]
    assert any(y < 162 for y in ys), "need landmark in upper third"
    assert any(162 <= y < 324 for y in ys), "need landmark in middle third"
    assert any(y >= 324 for y in ys), "need landmark in lower third"
```

Then update the `TESTS = [...]` list to include the new test functions.

- [ ] **Step 2: Run test to verify failures**

Run: `python test_cases/test_tp_chart.py`

Expected: 6 new tests fail with `AttributeError` on `TARTAN_REGIONS` / `GRAY_REGIONS` / `GRID_LANDMARKS`.

- [ ] **Step 3: Implement region table + landmark catalog**

Append to `tp_chart.py`:

```python
# =====================================================================
# IDEAL FRAME LAYOUT (720x486 active raster)
# =====================================================================
#
# Top-left tartan + gray region (Stage 1 scope):
#
#   x: [0, 30) [30, 60) [60, 90) [90, 120)
#   y: [0, 27)   YEL    CYN     BLU     RED       <- 75% top tartan row
#   y: [27, 54)  MAG_L  GRN_L   RED_L   CYN_L     <- low-sat bottom tartan row
#   y: [54, 81)  G1     G2      G3      G4        <- 20/40/60/80% gray strip
#   y: [81, 108) (boundary triangle cell, Stage 2)
#
# Coordinates are inclusive of x, exclusive of x+w (NumPy slicing convention).
# Sample windows are 20% of the box, centered.
#
# These coordinates may need calibration against real captures (the existing
# codex starter measured a 30x27 fine cell with row centers at y=13 and y=40
# for tartan, which agrees with this layout). Initial smoke testing on real
# sources will confirm or refine.

TARTAN_BOX_W = 30
TARTAN_BOX_H = 27
GRAY_BOX_W = 30
GRAY_BOX_H = 27

_TARTAN_TOP_ROW_Y = 0
_TARTAN_BOT_ROW_Y = 27
_GRAY_ROW_Y = 54

_TARTAN_TOP_COLORS = [
    ("YEL",   "yellow_75",   0.75, 0.75, 0.00),
    ("CYN",   "cyan_75",     0.00, 0.75, 0.75),
    ("BLU",   "blue_75",     0.00, 0.00, 0.75),
    ("RED",   "red_75",      0.75, 0.00, 0.00),
]

# Bottom row: low-saturation companions, codex-named.  Targets are taken from
# the codex sample measurements (the TPG generator values, transcribed from
# captures known to be close to ideal). Not a 25%-of-something formula.
_TARTAN_BOT_EXPECTED = [
    ("MAG_L",  "magenta_low",  267.0, 587.0, 607.0),
    ("GRN_L",  "green_low",    420.0, 436.0, 416.0),
    ("RED_L",  "red_low",      473.0, 474.0, 626.0),
    ("CYN_L",  "cyan_low",     675.0, 549.0, 397.0),
]

_SAMPLE = {"kind": "center_window", "size_frac": 0.2}


def _box(col: int, row_y: int, w: int = TARTAN_BOX_W, h: int = TARTAN_BOX_H):
    return (col * TARTAN_BOX_W, row_y, w, h)


def _build_tartan_regions():
    regions = []
    for col, (rid, name, r, g, b) in enumerate(_TARTAN_TOP_COLORS):
        y10, u10, v10 = rgb_norm_to_yuv10(r, g, b)
        regions.append({
            "id": rid,
            "name": name,
            "kind": "tartan_rect",
            "ideal_box": _box(col, _TARTAN_TOP_ROW_Y),
            "expected": {"y10": y10, "u10": u10, "v10": v10},
            "sample": _SAMPLE,
        })
    for col, (rid, name, y10, u10, v10) in enumerate(_TARTAN_BOT_EXPECTED):
        regions.append({
            "id": rid,
            "name": name,
            "kind": "tartan_rect",
            "ideal_box": _box(col, _TARTAN_BOT_ROW_Y),
            "expected": {"y10": y10, "u10": u10, "v10": v10},
            "sample": _SAMPLE,
        })
    return regions


def _build_gray_regions():
    return [
        {
            "id": f"G{i+1}",
            "name": f"gray_step_{int((i + 1) * 20)}",
            "kind": "gray_step",
            "ideal_box": _box(i, _GRAY_ROW_Y, GRAY_BOX_W, GRAY_BOX_H),
            "expected": {"y10": GRAY_IDEAL_Y10[i], "u10": 512, "v10": 512},
            "sample": _SAMPLE,
        }
        for i in range(4)
    ]


TARTAN_REGIONS = _build_tartan_regions()
GRAY_REGIONS = _build_gray_regions()


# =====================================================================
# REGISTRATION LANDMARK CATALOG
# =====================================================================
#
# Black grid intersections on grey background. Avoid the tartan/gray strip
# (top-left 120x108 region), the chart border, and the burst columns.
# 12x9 main grid implies intersections at multiples of 60 (x) and 54 (y).
# We bias coverage to give a stable affine fit (anchors in upper, middle,
# and lower thirds; columns spread across the picture).

_LANDMARK_GRID = [
    # (id, ideal_x, ideal_y) -- chosen on flat grey background, away from
    # tartan/gray (x>=180 in upper rows) and burst regions (mid-x avoided).
    ("L1", 180,  54),
    ("L2", 300,  54),
    ("L3", 540,  54),
    ("L4", 180, 162),
    ("L5", 540, 162),
    ("L6",  60, 270),
    ("L7", 660, 270),
    ("L8", 360, 378),
]

GRID_LANDMARKS = [
    {"id": rid, "ideal_x": x, "ideal_y": y, "search_window_px": 24}
    for rid, x, y in _LANDMARK_GRID
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_chart.py`

Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(sw2): tp_chart region table and grid-landmark catalog"
```

---

## Task 3: `tp_synthesize.py` — Base frame (grey + grid)

**Files:**
- Create: `tp_synthesize.py`
- Create: `test_cases/test_tp_synthesize.py`

- [ ] **Step 1: Write the failing test for the base frame**

Create `test_cases/test_tp_synthesize.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_synthesize."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize


def test_synthesize_shapes():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    assert Y.shape == (486, 720)
    assert U.shape == (486, 360)
    assert V.shape == (486, 360)
    assert Y.dtype == np.uint16


def test_synthesize_grey_background_dominates():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Most pixels (>= 50%) should be grey-background; we overlay tartan,
    # gray strip, and grid lines on top.
    grey_count = int((Y == tp_chart.GREY_BACKGROUND_Y10).sum())
    assert grey_count > Y.size * 0.5, (
        f"only {grey_count}/{Y.size} pixels at grey level"
    )


def test_synthesize_chroma_centred_off_color_regions():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # In the lower-right quadrant (no tartan/gray/triangle in stage 1),
    # chroma should be the centre value 512 everywhere except where grid
    # lines fall.
    u_mid = U[300:400, 250:300]
    v_mid = V[300:400, 250:300]
    assert (u_mid == tp_chart.CHROMA_CENTER).all()
    assert (v_mid == tp_chart.CHROMA_CENTER).all()


def test_synthesize_grid_intersections_dark():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for lm in tp_chart.GRID_LANDMARKS:
        x, y = lm["ideal_x"], lm["ideal_y"]
        # The 3x3 patch around a landmark should contain dark pixels
        patch = Y[y - 1:y + 2, x - 1:x + 2]
        assert int(patch.min()) < 100, (
            f"landmark {lm['id']} at ({x},{y}): no dark pixel "
            f"(min Y10={int(patch.min())})"
        )


TESTS = [
    test_synthesize_shapes,
    test_synthesize_grey_background_dominates,
    test_synthesize_chroma_centred_off_color_regions,
    test_synthesize_grid_intersections_dark,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(TESTS)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_synthesize.py`

Expected: `ModuleNotFoundError: No module named 'tp_synthesize'`.

- [ ] **Step 3: Implement the base synthesizer (grey + grid)**

Create `tp_synthesize.py`:

```python
#!/usr/bin/env python3
"""Synthesize the ideal SW2 NTSC test pattern frame at 720x486.

Stage 1 scope: grey background, black main grid, top-left tartan (4x2),
4-step gray strip below it, and a placeholder boundary-triangle cell.

Library entry point:
    Y, U, V = tp_synthesize.synthesize(720, 486)
        # uint16 yuv422p10le planes, BT.601 limited range

CLI (added in a later task):
    python tp_synthesize.py --raster 720x486 --output ideal.png
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

import tp_chart

# Main grid: 12 columns x 9 rows on 720x486 -> 60 x 54 px cells.
_GRID_COL_PX = 60
_GRID_ROW_PX = 54
# Black grid line width: 168 ns at 13.5 MHz ~= 2.27 samples. Use 3 px so
# the line is centred on integer-coord landmarks (an even width would
# straddle a sub-pixel boundary and bake a 0.5 px offset into detection).
_GRID_LINE_W = 3


def _make_grey_planes(width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = np.full((height, width), tp_chart.GREY_BACKGROUND_Y10, dtype=np.uint16)
    U = np.full((height, width // 2), tp_chart.CHROMA_CENTER, dtype=np.uint16)
    V = np.full((height, width // 2), tp_chart.CHROMA_CENTER, dtype=np.uint16)
    return Y, U, V


def _draw_grid(Y: np.ndarray) -> None:
    """Overlay the black main grid on the Y plane in place.

    With _GRID_LINE_W=3 (odd) and half=1, each line spans [center-1, center+2),
    i.e. 3 columns/rows centred exactly on the integer landmark coordinate.
    """
    h, w = Y.shape
    half = _GRID_LINE_W // 2  # 1 for width 3
    for x in range(0, w + 1, _GRID_COL_PX):
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        if x1 > x0:
            Y[:, x0:x1] = tp_chart.BLACK_Y10
    for y in range(0, h + 1, _GRID_ROW_PX):
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        if y1 > y0:
            Y[y0:y1, :] = tp_chart.BLACK_Y10


def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ideal SW2 frame as (Y, U, V) uint16 planes (yuv422p10le)."""
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    # Tartan + gray + boundary triangle added in subsequent tasks.
    return Y, U, V
```

Note on landmark placement: the GRID_LANDMARKS x-coords (60, 180, 300, 360, 540, 660) are all multiples of 60 and y-coords (54, 162, 270, 378) are all multiples of 54, so each lands exactly on a grid intersection.

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_synthesize.py`

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(sw2): tp_synthesize base frame (grey + grid)"
```

---

## Task 4: `tp_synthesize.py` — Tartan blocks

**Files:**
- Modify: `tp_synthesize.py`
- Modify: `test_cases/test_tp_synthesize.py`

- [ ] **Step 1: Add failing test for tartan rendering**

Append to `test_cases/test_tp_synthesize.py` (before the `TESTS = [...]` list):

```python
def test_synthesize_tartan_centers():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for r in tp_chart.TARTAN_REGIONS:
        x, y, w, h = r["ideal_box"]
        cx, cy = x + w // 2, y + h // 2
        # Sample a 3x3 patch at the box centre to avoid grid edges.
        y_sample = float(Y[cy - 1:cy + 2, cx - 1:cx + 2].mean())
        u_sample = float(U[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        v_sample = float(V[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        assert abs(y_sample - r["expected"]["y10"]) < 2.0, (
            f"{r['id']} Y10: got {y_sample:.1f}, want {r['expected']['y10']:.1f}"
        )
        assert abs(u_sample - r["expected"]["u10"]) < 2.0, (
            f"{r['id']} U10: got {u_sample:.1f}, want {r['expected']['u10']:.1f}"
        )
        assert abs(v_sample - r["expected"]["v10"]) < 2.0, (
            f"{r['id']} V10: got {v_sample:.1f}, want {r['expected']['v10']:.1f}"
        )
```

Add `test_synthesize_tartan_centers` to the `TESTS` list.

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_synthesize.py`

Expected: `test_synthesize_tartan_centers` fails because the tartan region centers still show grey-background Y10=502 instead of the per-color expected values.

- [ ] **Step 3: Implement tartan rendering**

In `tp_synthesize.py`, add the tartan renderer and call it from `synthesize`:

```python
def _fill_box_yuv422(
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    box: Tuple[int, int, int, int],
    y10: float,
    u10: float,
    v10: float,
) -> None:
    """Fill a rectangular box in all three planes with the given YUV codes."""
    x, y, w, h = box
    Y[y:y + h, x:x + w] = int(round(y10))
    # 4:2:2 chroma: x is at 2x luma resolution.
    cx0, cx1 = x // 2, (x + w) // 2
    U[y:y + h, cx0:cx1] = int(round(u10))
    V[y:y + h, cx0:cx1] = int(round(v10))


def _draw_tartan(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    for r in tp_chart.TARTAN_REGIONS:
        e = r["expected"]
        _fill_box_yuv422(Y, U, V, r["ideal_box"], e["y10"], e["u10"], e["v10"])
```

Then update `synthesize` to call `_draw_tartan` after `_draw_grid`:

```python
def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    _draw_tartan(Y, U, V)
    return Y, U, V
```

Note: tartan boxes are drawn AFTER the grid, so the colored boxes overwrite any grid lines that fall inside them. This is correct — the chart layout has the grid drawn underneath the colored regions, but the colored regions are opaque.

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_synthesize.py`

Expected: all tests pass (5 total now).

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(sw2): tp_synthesize tartan 4x2 block"
```

---

## Task 5: `tp_synthesize.py` — Gray strip

**Files:**
- Modify: `tp_synthesize.py`
- Modify: `test_cases/test_tp_synthesize.py`

- [ ] **Step 1: Add failing test for gray strip**

Append to `test_cases/test_tp_synthesize.py` (before the `TESTS` list):

```python
def test_synthesize_gray_strip_centers():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for r in tp_chart.GRAY_REGIONS:
        x, y, w, h = r["ideal_box"]
        cx, cy = x + w // 2, y + h // 2
        y_sample = float(Y[cy - 1:cy + 2, cx - 1:cx + 2].mean())
        u_sample = float(U[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        v_sample = float(V[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        assert abs(y_sample - r["expected"]["y10"]) < 1.0, (
            f"{r['id']} Y10: got {y_sample:.1f}, want {r['expected']['y10']:.1f}"
        )
        assert int(round(u_sample)) == 512
        assert int(round(v_sample)) == 512
```

Add `test_synthesize_gray_strip_centers` to `TESTS`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_synthesize.py`

Expected: `test_synthesize_gray_strip_centers` fails (centers are still grey-background, not the per-step values).

- [ ] **Step 3: Implement gray strip rendering**

Add to `tp_synthesize.py`:

```python
def _draw_gray_strip(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    for r in tp_chart.GRAY_REGIONS:
        e = r["expected"]
        _fill_box_yuv422(Y, U, V, r["ideal_box"], e["y10"], e["u10"], e["v10"])
```

Update `synthesize`:

```python
def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    _draw_tartan(Y, U, V)
    _draw_gray_strip(Y, U, V)
    return Y, U, V
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_synthesize.py`

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(sw2): tp_synthesize 4-step gray strip"
```

---

## Task 6: `tp_synthesize.py` — Boundary triangle + CLI

**Files:**
- Modify: `tp_synthesize.py`
- Modify: `test_cases/test_tp_synthesize.py`

- [ ] **Step 1: Add failing test for boundary triangle and CLI smoke**

Append to `test_cases/test_tp_synthesize.py` (before `TESTS`):

```python
def test_synthesize_boundary_triangle_present():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Boundary-triangle cell is the 30x27 box at x in [0,30), y in [81,108).
    cell = Y[81:108, 0:30]
    # Should contain a meaningful chunk of black pixels (the triangle interior).
    black_count = int((cell == tp_chart.BLACK_Y10).sum())
    assert black_count > 50, (
        f"boundary-triangle cell has only {black_count} black pixels "
        f"(out of {cell.size})"
    )


def test_cli_writes_png(tmp_dir):
    import subprocess
    out = os.path.join(tmp_dir, "ideal.png")
    cmd = ["python", "tp_synthesize.py", "--raster", "720x486", "--output", out]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists(out)
    # PNG should be 720x486
    import cv2
    img = cv2.imread(out)
    assert img.shape == (486, 720, 3)


def test_cli_writes_yuv(tmp_dir):
    import subprocess
    out = os.path.join(tmp_dir, "ideal.yuv")
    cmd = ["python", "tp_synthesize.py", "--raster", "720x486", "--output", out]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists(out)
    # yuv422p10le: (720*486 + 2 * 360*486) * 2 bytes
    expected_size = (720 * 486 + 2 * 360 * 486) * 2
    assert os.path.getsize(out) == expected_size
```

Replace the test runner so `tmp_dir` is created and passed:

```python
import tempfile
import shutil

TESTS_NO_TMPDIR = [
    test_synthesize_shapes,
    test_synthesize_grey_background_dominates,
    test_synthesize_chroma_centred_off_color_regions,
    test_synthesize_grid_intersections_dark,
    test_synthesize_tartan_centers,
    test_synthesize_gray_strip_centers,
    test_synthesize_boundary_triangle_present,
]
TESTS_TMPDIR = [
    test_cli_writes_png,
    test_cli_writes_yuv,
]


def main():
    failed = 0
    for t in TESTS_NO_TMPDIR:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    tmp = tempfile.mkdtemp(prefix="tp_synth_test_")
    try:
        for t in TESTS_TMPDIR:
            try:
                t(tmp)
                print(f"PASS  {t.__name__}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {t.__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(TESTS_NO_TMPDIR) + len(TESTS_TMPDIR)
    if failed:
        print(f"\n{failed}/{total} tests failed")
        sys.exit(1)
    print(f"\nAll {total} tests passed")
```

(Replace the previous `TESTS` list and `main()` definition.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_synthesize.py`

Expected: `test_synthesize_boundary_triangle_present` fails (no triangle yet) and CLI tests fail (`tp_synthesize.py` has no `__main__` argparse handler).

- [ ] **Step 3: Implement boundary triangle + CLI**

Append to `tp_synthesize.py`:

```python
def _draw_boundary_triangle_upper_left(Y: np.ndarray) -> None:
    """Black filled triangle inside the (0..30, 81..108) cell, pointing right.

    For Stage 1 we render only the upper-left boundary-triangle cell; the
    other three corners are added in Stage 2 along with their detectors.
    The shape is a simple right-pointing triangle, vertices roughly:
       (3, 84)  top-left
       (3, 105) bottom-left
       (24, 94) right tip
    Drawn by filling row-by-row.
    """
    # Linear interpolation: at row r in [84, 105], the right-side x is
    # x_right = 24 - 21 * abs(r - 94) / 11   -- crude diamond/triangle
    for r in range(84, 106):
        # Distance from the apex row 94 (range 0..11)
        d = abs(r - 94)
        x_right = max(4, 24 - int(round(21 * d / 11)))
        Y[r, 3:x_right] = tp_chart.BLACK_Y10


def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    _draw_tartan(Y, U, V)
    _draw_gray_strip(Y, U, V)
    _draw_boundary_triangle_upper_left(Y)
    return Y, U, V


# =====================================================================
# CLI
# =====================================================================

def _yuv422p10_to_bgr8(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Convert yuv422p10le planes to 8-bit BGR for image writing."""
    h, w = Y.shape
    # Upsample chroma to full width.
    U_full = np.repeat(U, 2, axis=1)[:, :w]
    V_full = np.repeat(V, 2, axis=1)[:, :w]
    # Apply BT.601 limited-range inverse.
    y = (Y.astype(np.float32) - tp_chart.BLACK_Y10) / tp_chart.Y_RANGE
    cb = (U_full.astype(np.float32) - tp_chart.CHROMA_CENTER) / 896.0
    cr = (V_full.astype(np.float32) - tp_chart.CHROMA_CENTER) / 896.0
    r = np.clip(y + 1.402 * cr, 0.0, 1.0)
    g = np.clip(y - 0.344136 * cb - 0.714136 * cr, 0.0, 1.0)
    b = np.clip(y + 1.772 * cb, 0.0, 1.0)
    bgr = np.stack([b, g, r], axis=-1) * 255.0
    return bgr.astype(np.uint8)


def _write_yuv422p10le(path: str, Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(Y.astype("<u2").tobytes())
        f.write(U.astype("<u2").tobytes())
        f.write(V.astype("<u2").tobytes())


def _parse_raster(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raster", default="720x486", help="WIDTHxHEIGHT")
    p.add_argument("--output", required=True, help="output file (.png or .yuv)")
    args = p.parse_args()
    w, h = _parse_raster(args.raster)
    Y, U, V = synthesize(w, h)
    if args.output.endswith(".png"):
        import cv2
        bgr = _yuv422p10_to_bgr8(Y, U, V)
        cv2.imwrite(args.output, bgr)
    elif args.output.endswith(".yuv"):
        _write_yuv422p10le(args.output, Y, U, V)
    else:
        raise SystemExit("output must end in .png or .yuv")
    print(f"wrote {args.output} ({w}x{h})")


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_synthesize.py`

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(sw2): tp_synthesize boundary triangle + CLI"
```

---

## Task 7: `tp_register.py` — Landmark detector

**Files:**
- Create: `tp_register.py`
- Create: `test_cases/test_tp_register.py`

- [ ] **Step 1: Write the failing test**

Create `test_cases/test_tp_register.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_register: landmark detection + affine fit."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_register


def approx(a, b, tol):
    return abs(a - b) <= tol


def test_detect_landmark_on_synthesized_ideal():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    for lm in tp_chart.GRID_LANDMARKS:
        result = tp_register.detect_landmark(
            Y, lm["ideal_x"], lm["ideal_y"], lm["search_window_px"],
        )
        assert result is not None, f"no detection at {lm['id']}"
        det_x, det_y, conf = result
        assert approx(det_x, lm["ideal_x"], 0.6), (
            f"{lm['id']} x: got {det_x:.2f}, want {lm['ideal_x']}"
        )
        assert approx(det_y, lm["ideal_y"], 0.6), (
            f"{lm['id']} y: got {det_y:.2f}, want {lm['ideal_y']}"
        )
        assert conf > 0.05


def test_detect_landmark_off_grid_returns_none():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    # Look in a flat-grey region: cell centre at (270, 189) -- 30 px from
    # the nearest grid line at x=240/300 and 27 px from y=162/216, well
    # outside any 24-px search window.
    result = tp_register.detect_landmark(Y, 270, 189, 24)
    assert result is None or result[2] < 0.05


TESTS = [
    test_detect_landmark_on_synthesized_ideal,
    test_detect_landmark_off_grid_returns_none,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(TESTS)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_register.py`

Expected: `ModuleNotFoundError: No module named 'tp_register'`.

- [ ] **Step 3: Implement landmark detector**

Create `tp_register.py`:

```python
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

    # Reject single blob (no cross): require both >=2 columns and >=2 rows
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_register.py`

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(sw2): tp_register landmark detector"
```

---

## Task 8: `tp_register.py` — RANSAC affine fit

**Files:**
- Modify: `tp_register.py`
- Modify: `test_cases/test_tp_register.py`

- [ ] **Step 1: Add failing tests for affine fit**

Append to `test_cases/test_tp_register.py` (before `TESTS`):

```python
def test_fit_affine_identity():
    pts = np.array([[100.0, 50.0], [300.0, 50.0], [200.0, 200.0],
                    [50.0, 300.0], [400.0, 250.0]], dtype=np.float32)
    result = tp_register.fit_affine(pts.copy(), pts.copy())
    M = result["affine_matrix"]
    assert M is not None
    # Identity: M = [[1, 0, 0], [0, 1, 0]]
    np.testing.assert_allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-3)
    assert result["residuals_px"]["mean"] < 1e-3
    assert result["inliers"] == len(pts)


def test_fit_affine_translation():
    ideal = np.array([[100.0, 50.0], [300.0, 50.0], [200.0, 200.0],
                      [50.0, 300.0], [400.0, 250.0]], dtype=np.float32)
    detected = ideal + np.array([5.0, 7.0], dtype=np.float32)
    result = tp_register.fit_affine(detected, ideal)
    M = result["affine_matrix"]
    assert M is not None
    np.testing.assert_allclose(M[:, 2], [5.0, 7.0], atol=0.1)
    np.testing.assert_allclose(M[:, :2], np.eye(2), atol=1e-3)
    assert result["residuals_px"]["max"] < 0.5


def test_fit_affine_rejects_outlier():
    ideal = np.array([[100.0, 50.0], [300.0, 50.0], [200.0, 200.0],
                      [50.0, 300.0], [400.0, 250.0], [550.0, 400.0]],
                     dtype=np.float32)
    # All shift by (3, 4) except the last is wildly off.
    detected = ideal + np.array([3.0, 4.0], dtype=np.float32)
    detected[-1] += np.array([50.0, 60.0], dtype=np.float32)
    result = tp_register.fit_affine(detected, ideal)
    assert result["inliers"] == 5
    assert result["total"] == 6
    np.testing.assert_allclose(result["affine_matrix"][:, 2], [3.0, 4.0], atol=0.1)
```

Add the three new tests to `TESTS`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_register.py`

Expected: 3 new tests fail (`fit_affine` not defined).

- [ ] **Step 3: Implement RANSAC affine fit**

Append to `tp_register.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_register.py`

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(sw2): tp_register RANSAC affine fit"
```

---

## Task 9: `tp_register.py` — Top-level `register()`

**Files:**
- Modify: `tp_register.py`
- Modify: `test_cases/test_tp_register.py`

- [ ] **Step 1: Add failing test for end-to-end register()**

Append to `test_cases/test_tp_register.py`:

```python
def test_register_identity_on_synthesized_ideal():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    result = tp_register.register(Y)
    assert result["affine_matrix"] is not None
    np.testing.assert_allclose(result["affine_matrix"], [[1, 0, 0], [0, 1, 0]], atol=0.5)
    assert result["residuals_px"]["mean"] < 0.6
    assert result["quality_flag"] == "ok"
    assert result["inliers"] >= len(tp_chart.GRID_LANDMARKS) - 1


def test_register_recovers_translation():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    # Shift the synthesized frame down-right by (4, 3) by rolling the array.
    shifted = np.roll(np.roll(Y, 4, axis=1), 3, axis=0)
    result = tp_register.register(shifted)
    assert result["affine_matrix"] is not None
    np.testing.assert_allclose(
        result["affine_matrix"][:, 2], [4.0, 3.0], atol=0.7
    )
    np.testing.assert_allclose(result["affine_matrix"][:, :2], np.eye(2), atol=0.05)
    assert result["quality_flag"] == "ok"


def test_register_quality_flag_failure_when_no_landmarks_detect():
    # All-grey frame: no grid -> no detections.
    Y = np.full((486, 720), tp_chart.GREY_BACKGROUND_Y10, dtype=np.uint16)
    result = tp_register.register(Y)
    assert result["quality_flag"] == "failed"
    assert result["affine_matrix"] is None
```

Add to `TESTS`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_register.py`

Expected: 3 new tests fail (`register` not defined).

- [ ] **Step 3: Implement register()**

Append to `tp_register.py`:

```python
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
        det = detect_landmark(Y, lm["ideal_x"], lm["ideal_y"], lm["search_window_px"])
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
    return fit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_register.py`

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(sw2): tp_register top-level register() with quality flag"
```

---

## Task 10: `tp_measure.py` — Frame extraction + padding

**Files:**
- Create: `tp_measure.py`
- Create: `test_cases/test_tp_measure.py`

- [ ] **Step 1: Write the failing test**

Create `test_cases/test_tp_measure.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_measure: frame extraction, padding, sampling, JSON."""

import sys, os, json, tempfile, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_measure


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_synthesized_prores(path, width, height, frames=10):
    """Encode a synthesized SW2 frame as ProRes 422 HQ at the given raster.

    The synthesizer produces a single frame; we write `frames` copies so the
    extractor can pick frame 0 (or any small index).
    """
    Y, U, V = tp_synthesize.synthesize(width, height)
    raw = (Y.astype("<u2").tobytes()
           + U.astype("<u2").tobytes()
           + V.astype("<u2").tobytes())
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "-s", f"{width}x{height}",
        "-r", "30000/1001",
        "-i", "pipe:0",
        "-frames:v", str(frames),
        "-c:v", "prores_ks", "-profile:v", "3",
        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for _ in range(frames):
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0


def test_extract_frame_yuv422p10le_720x486(tmp_dir):
    path = os.path.join(tmp_dir, "ideal.mov")
    _write_synthesized_prores(path, 720, 486)
    Y, U, V, meta = tp_measure.extract_frame(path, frame_index=0)
    assert Y.shape == (486, 720)
    assert U.shape == (486, 360)
    assert V.shape == (486, 360)
    assert meta["raster_in"] == [720, 486]
    # Center pixel of YEL tartan box should be at the expected Y10.
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    x, y, w, h = yel["ideal_box"]
    cx, cy = x + w // 2, y + h // 2
    assert abs(int(Y[cy, cx]) - yel["expected"]["y10"]) < 5


def test_pad_to_486_centers_grey():
    Y_in = np.full((480, 720), 200, dtype=np.uint16)
    U_in = np.full((480, 360), 512, dtype=np.uint16)
    V_in = np.full((480, 360), 512, dtype=np.uint16)
    Y, U, V, offsets = tp_measure.pad_to_486(Y_in, U_in, V_in)
    assert Y.shape == (486, 720)
    assert offsets == {"top": 3, "bottom": 3, "left": 0, "right": 0}
    # Top 3 rows are grey-background; rows 3..483 are 200; bottom 3 are grey.
    assert (Y[:3, :] == tp_chart.GREY_BACKGROUND_Y10).all()
    assert (Y[3:483, :] == 200).all()
    assert (Y[483:, :] == tp_chart.GREY_BACKGROUND_Y10).all()


def test_extract_pad_720x480_dvd_like(tmp_dir):
    path = os.path.join(tmp_dir, "dvd.mov")
    # Take the 720x486 ideal and render frames at 720x480 (no padding inside).
    # We do this by truncating the synthesized planes before encoding.
    Y, U, V = tp_synthesize.synthesize(720, 486)
    Y480 = Y[3:483, :]
    U480 = U[3:483, :]
    V480 = V[3:483, :]
    raw = (Y480.astype("<u2").tobytes()
           + U480.astype("<u2").tobytes()
           + V480.astype("<u2").tobytes())
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "-s", "720x480", "-r", "30000/1001",
        "-i", "pipe:0", "-frames:v", "5",
        "-c:v", "prores_ks", "-profile:v", "3",
        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for _ in range(5):
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()
    Y, U, V, meta = tp_measure.extract_frame(path, frame_index=0)
    Y_padded, U_padded, V_padded, offsets = tp_measure.pad_to_486(Y, U, V)
    assert Y_padded.shape == (486, 720)
    assert offsets["top"] == 3 and offsets["bottom"] == 3


TESTS_TMPDIR = [
    test_extract_frame_yuv422p10le_720x486,
    test_extract_pad_720x480_dvd_like,
]
TESTS_NO_TMPDIR = [
    test_pad_to_486_centers_grey,
]


def main():
    failed = 0
    for t in TESTS_NO_TMPDIR:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    tmp = tempfile.mkdtemp(prefix="tp_measure_test_")
    try:
        for t in TESTS_TMPDIR:
            try:
                t(tmp)
                print(f"PASS  {t.__name__}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {t.__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(TESTS_NO_TMPDIR) + len(TESTS_TMPDIR)
    if failed:
        print(f"\n{failed}/{total} tests failed")
        sys.exit(1)
    print(f"\nAll {total} tests passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_measure.py`

Expected: `ModuleNotFoundError: No module named 'tp_measure'`.

- [ ] **Step 3: Implement extraction + padding**

Create `tp_measure.py`:

```python
#!/usr/bin/env python3
"""tp_measure: per-capture SW2 measurement.

Pipeline: probe -> extract single frame (no deinterlace) -> pad to 720x486 ->
register against ideal -> sample tartan + gray patches -> write JSON.

CLI:
    python tp_measure.py <capture> --frame N --output cap.json
"""

from __future__ import annotations
import json
import subprocess
from typing import Any, Dict, Tuple

import numpy as np

import common  # existing module
import tp_chart
import tp_register


_TOOL_VERSION = "tp_measure 0.1"


def extract_frame(
    capture_path: str, frame_index: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Decode a single yuv422p10le frame from `capture_path` at `frame_index`.

    Uses ffmpeg with NO deinterlace filter: interlaced sources are weaved
    (lines as-stored).
    """
    width, height = common.probe_video(capture_path)

    # Probe field order to detect unexpected progressive sources.
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=field_order,codec_name",
        "-of", "default=nw=1", capture_path,
    ]
    probe_out = subprocess.check_output(probe_cmd, text=True)
    field_order = "unknown"
    codec_name = "unknown"
    for line in probe_out.strip().splitlines():
        k, _, v = line.partition("=")
        if k == "field_order":
            field_order = v.strip() or "unknown"
        elif k == "codec_name":
            codec_name = v.strip() or "unknown"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", capture_path,
        "-vf", f"select=eq(n\\,{int(frame_index)})",
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    y_size = width * height
    u_size = (width // 2) * height
    expected = (y_size + u_size + u_size) * 2
    if len(raw) != expected:
        raise RuntimeError(
            f"ffmpeg returned {len(raw)} bytes; expected {expected} "
            f"for {width}x{height} yuv422p10le"
        )
    arr = np.frombuffer(raw, dtype=np.uint16)
    Y = arr[:y_size].reshape(height, width).copy()
    U = arr[y_size:y_size + u_size].reshape(height, width // 2).copy()
    V = arr[y_size + u_size:].reshape(height, width // 2).copy()

    meta: Dict[str, Any] = {
        "capture": capture_path,
        "frame_index": int(frame_index),
        "raster_in": [width, height],
        "field_order": field_order,
        "codec_name": codec_name,
        "progressive_warning": (field_order == "progressive"),
        "decode": {
            "deinterlace": "none-weave-only",
            "pix_fmt": "yuv422p10le",
        },
    }
    return Y, U, V, meta


def pad_to_486(
    Y: np.ndarray, U: np.ndarray, V: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Pad Y/U/V planes up to height=486 with grey-background. No horizontal
    padding (NTSC active is always 720 wide; mismatches raise).
    """
    height = Y.shape[0]
    width = Y.shape[1]
    if width != 720:
        raise ValueError(f"unexpected width {width}; expected 720")
    if height >= 486:
        return Y, U, V, {"top": 0, "bottom": 0, "left": 0, "right": 0}
    delta = 486 - height
    top = delta // 2
    bottom = delta - top
    grey_y = tp_chart.GREY_BACKGROUND_Y10
    grey_c = tp_chart.CHROMA_CENTER

    def _pad(plane, c_w, fill):
        new = np.full((486, c_w), fill, dtype=plane.dtype)
        new[top:top + height, :] = plane
        return new

    Y_padded = _pad(Y, width, grey_y)
    U_padded = _pad(U, width // 2, grey_c)
    V_padded = _pad(V, width // 2, grey_c)
    return Y_padded, U_padded, V_padded, {
        "top": top, "bottom": bottom, "left": 0, "right": 0
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_measure.py`

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_measure.py test_cases/test_tp_measure.py
git commit -m "feat(sw2): tp_measure frame extraction + padding helpers"
```

---

## Task 11: `tp_measure.py` — Region sampling

**Files:**
- Modify: `tp_measure.py`
- Modify: `test_cases/test_tp_measure.py`

- [ ] **Step 1: Add failing test for sampling**

Append to `test_cases/test_tp_measure.py` (before TESTS lists):

```python
def test_sample_region_on_synthesized_identity():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    measurement = tp_measure.sample_region(Y, U, V, yel, identity)
    assert abs(measurement["measured_yuv10"][0] - yel["expected"]["y10"]) < 1.0
    assert abs(measurement["measured_yuv10"][1] - yel["expected"]["u10"]) < 1.0
    assert abs(measurement["measured_yuv10"][2] - yel["expected"]["v10"]) < 1.0
    assert abs(measurement["delta_yuv10"][0]) < 1.0
    assert abs(measurement["delta_yuv10"][1]) < 1.0
    assert abs(measurement["delta_yuv10"][2]) < 1.0


def test_sample_region_with_translation():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Translate ideal -> capture by (5, 7): a YEL pixel at ideal (cx,cy) is
    # actually at (cx+5, cy+7) in the capture. So affine = [[1,0,5],[0,1,7]].
    M = np.array([[1, 0, 5], [0, 1, 7]], dtype=np.float32)
    # Roll the synthesized frame by (5, 7) to simulate a shifted capture.
    Y_shifted = np.roll(np.roll(Y, 5, axis=1), 7, axis=0)
    U_shifted = np.roll(np.roll(U, 5 // 2, axis=1), 7, axis=0)
    V_shifted = np.roll(np.roll(V, 5 // 2, axis=1), 7, axis=0)
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    measurement = tp_measure.sample_region(Y_shifted, U_shifted, V_shifted, yel, M)
    assert abs(measurement["delta_yuv10"][0]) < 5.0
```

Add both to `TESTS_NO_TMPDIR`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_measure.py`

Expected: `sample_region` not defined → AttributeError.

- [ ] **Step 3: Implement sample_region**

Append to `tp_measure.py`:

```python
def _apply_affine(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    a, b, tx = M[0]
    c, d, ty = M[1]
    return a * x + b * y + tx, c * x + d * y + ty


def sample_region(
    Y: np.ndarray, U: np.ndarray, V: np.ndarray,
    region: Dict[str, Any], affine: np.ndarray,
) -> Dict[str, Any]:
    """Sample the centre window of `region` from a registered capture.

    `affine` maps ideal coords -> capture coords (2x3, applied as
    [a, b, tx; c, d, ty]).
    """
    x, y, w, h = region["ideal_box"]
    cx_ideal = x + w / 2.0
    cy_ideal = y + h / 2.0
    cx_cap, cy_cap = _apply_affine(affine, cx_ideal, cy_ideal)

    size_frac = float(region["sample"]["size_frac"])
    half_w = max(1, int(round(w * size_frac / 2.0)))
    half_h = max(1, int(round(h * size_frac / 2.0)))

    x0 = max(0, int(round(cx_cap)) - half_w)
    x1 = min(Y.shape[1], int(round(cx_cap)) + half_w)
    y0 = max(0, int(round(cy_cap)) - half_h)
    y1 = min(Y.shape[0], int(round(cy_cap)) + half_h)

    y_patch = Y[y0:y1, x0:x1].astype(np.float64)
    # Chroma subsampled: x is at 2x luma resolution.
    cx0, cx1 = x0 // 2, max(x0 // 2 + 1, x1 // 2)
    u_patch = U[y0:y1, cx0:cx1].astype(np.float64)
    v_patch = V[y0:y1, cx0:cx1].astype(np.float64)

    measured = (
        float(y_patch.mean()) if y_patch.size else float("nan"),
        float(u_patch.mean()) if u_patch.size else float("nan"),
        float(v_patch.mean()) if v_patch.size else float("nan"),
    )
    e = region["expected"]
    delta = (
        measured[0] - e["y10"],
        measured[1] - e["u10"],
        measured[2] - e["v10"],
    )

    # Saturation as ratio of chroma-magnitude vs ideal.
    measured_chroma = ((measured[1] - tp_chart.CHROMA_CENTER) ** 2
                       + (measured[2] - tp_chart.CHROMA_CENTER) ** 2) ** 0.5
    ideal_chroma = ((e["u10"] - tp_chart.CHROMA_CENTER) ** 2
                    + (e["v10"] - tp_chart.CHROMA_CENTER) ** 2) ** 0.5
    sat_pct = (measured_chroma / ideal_chroma * 100.0) if ideal_chroma > 1e-6 else None

    return {
        "id": region["id"],
        "name": region["name"],
        "ideal_yuv10": [e["y10"], e["u10"], e["v10"]],
        "measured_yuv10": list(measured),
        "delta_yuv10": list(delta),
        "sat_pct_vs_ideal": sat_pct,
        "patch_size_px": [x1 - x0, y1 - y0],
        "patch_center_capture_xy": [cx_cap, cy_cap],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_measure.py`

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_measure.py test_cases/test_tp_measure.py
git commit -m "feat(sw2): tp_measure region sampling helper"
```

---

## Task 12: `tp_measure.py` — End-to-end pipeline + JSON writer + CLI

**Files:**
- Modify: `tp_measure.py`
- Modify: `test_cases/test_tp_measure.py`

- [ ] **Step 1: Add failing test for end-to-end JSON output**

Append to `test_cases/test_tp_measure.py`:

```python
def test_measure_end_to_end_zero_deltas(tmp_dir):
    capture_path = os.path.join(tmp_dir, "ideal.mov")
    json_path = os.path.join(tmp_dir, "out.json")
    _write_synthesized_prores(capture_path, 720, 486, frames=5)
    cmd = [
        "python", "tp_measure.py", capture_path,
        "--frame", "0",
        "--output", json_path,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    with open(json_path) as f:
        data = json.load(f)
    assert data["_meta"]["raster_in"] == [720, 486]
    assert data["_meta"]["registration"]["quality_flag"] == "ok"
    assert len(data["tartan"]) == 8
    assert len(data["grays"]) == 4
    # On the synthesized ideal, all deltas should be tiny.
    for patch in data["tartan"]:
        assert abs(patch["delta_yuv10"][0]) < 2.0, patch
    for g in data["grays"]:
        assert abs(g["delta_y10"]) < 1.0, g


def test_measure_end_to_end_dvd_padding(tmp_dir):
    capture_path = os.path.join(tmp_dir, "dvd.mov")
    json_path = os.path.join(tmp_dir, "dvd.json")
    # Build 720x480 ProRes: same trick as test_extract_pad_720x480_dvd_like
    Y, U, V = tp_synthesize.synthesize(720, 486)
    Y480, U480, V480 = Y[3:483, :], U[3:483, :], V[3:483, :]
    raw = (Y480.astype("<u2").tobytes()
           + U480.astype("<u2").tobytes()
           + V480.astype("<u2").tobytes())
    enc_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "-s", "720x480", "-r", "30000/1001",
        "-i", "pipe:0", "-frames:v", "5",
        "-c:v", "prores_ks", "-profile:v", "3",
        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
        capture_path,
    ]
    proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    for _ in range(5):
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()
    cmd = [
        "python", "tp_measure.py", capture_path,
        "--frame", "0", "--output", json_path,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    with open(json_path) as f:
        data = json.load(f)
    assert data["_meta"]["raster_in"] == [720, 480]
    assert data["_meta"]["padding_offsets"] == {"top": 3, "bottom": 3, "left": 0, "right": 0}
    assert data["_meta"]["registration"]["quality_flag"] in ("ok", "warn")
```

Add both to `TESTS_TMPDIR`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_measure.py`

Expected: tests fail because `tp_measure.py` has no `__main__` / argparse.

- [ ] **Step 3: Implement the CLI / main flow**

Append to `tp_measure.py`:

```python
import hashlib
import sys


def _ideal_frame_md5() -> str:
    Y, U, V = _ideal_for_md5()
    h = hashlib.md5()
    h.update(Y.tobytes()); h.update(U.tobytes()); h.update(V.tobytes())
    return h.hexdigest()


def _ideal_for_md5():
    import tp_synthesize
    return tp_synthesize.synthesize(720, 486)


def measure(capture_path: str, frame_index: int) -> Dict[str, Any]:
    Y, U, V, meta = extract_frame(capture_path, frame_index)
    Y_p, U_p, V_p, padding = pad_to_486(Y, U, V)
    meta["raster_processed"] = [720, 486]
    meta["padding_offsets"] = padding

    if meta["progressive_warning"]:
        print(
            f"WARNING: source field_order={meta['field_order']} indicates "
            f"progressive scan; SW2 analysis assumes interlaced source.",
            file=sys.stderr,
        )

    reg = tp_register.register(Y_p)
    meta["registration"] = {
        "affine": reg["affine_matrix"].tolist() if reg["affine_matrix"] is not None else None,
        "residuals_px": reg["residuals_px"],
        "inliers": reg["inliers"],
        "total": reg["total"],
        "landmarks_used": reg["landmarks_used"],
        "quality_flag": reg["quality_flag"],
        "quality_reason": reg.get("quality_reason"),
    }

    if reg["affine_matrix"] is None:
        raise RuntimeError(
            f"registration failed for {capture_path}: "
            f"{reg.get('quality_reason', 'no affine')}"
        )

    M = reg["affine_matrix"]
    tartan = [sample_region(Y_p, U_p, V_p, r, M) for r in tp_chart.TARTAN_REGIONS]
    grays_raw = [sample_region(Y_p, U_p, V_p, r, M) for r in tp_chart.GRAY_REGIONS]
    # Reshape gray records to the schema in the spec (flat ideal_y10 + delta_y10).
    grays = []
    for r, raw in zip(tp_chart.GRAY_REGIONS, grays_raw):
        grays.append({
            "id": r["id"],
            "name": r["name"],
            "ideal_y10": r["expected"]["y10"],
            "measured_y10": raw["measured_yuv10"][0],
            "delta_y10": raw["delta_yuv10"][0],
            "u10": raw["measured_yuv10"][1],
            "v10": raw["measured_yuv10"][2],
            "patch_size_px": raw["patch_size_px"],
        })

    meta["ideal_frame_md5"] = _ideal_frame_md5()
    meta["tp_chart_version"] = tp_chart.TP_CHART_VERSION
    meta["tool_version"] = _TOOL_VERSION

    return {"_meta": meta, "tartan": tartan, "grays": grays}


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="path to capture (mov/avi/mkv/...)")
    p.add_argument("--frame", type=int, default=60, help="frame index (default 60)")
    p.add_argument("--output", required=True, help="output JSON path")
    args = p.parse_args()
    data = measure(args.capture, args.frame)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(
        f"wrote {args.output}: "
        f"registration={data['_meta']['registration']['quality_flag']}, "
        f"residuals_mean={data['_meta']['registration']['residuals_px']['mean']:.2f}px"
    )


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_measure.py`

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_measure.py test_cases/test_tp_measure.py
git commit -m "feat(sw2): tp_measure end-to-end CLI + JSON writer"
```

---

## Task 13: `tp_compare.py` — Skeleton + registration summary section

**Files:**
- Create: `tp_compare.py`
- Create: `test_cases/test_tp_compare.py`

- [ ] **Step 1: Write the failing test**

Create `test_cases/test_tp_compare.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_compare HTML rendering."""

import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tp_compare


def _make_capture_json(tag, residual_mean=0.4, quality="ok"):
    return {
        "_meta": {
            "capture": f"/path/{tag}.mov",
            "frame_index": 60,
            "raster_in": [720, 480],
            "raster_processed": [720, 486],
            "padding_offsets": {"top": 3, "bottom": 3, "left": 0, "right": 0},
            "field_order": "tb",
            "progressive_warning": False,
            "decode": {"deinterlace": "none-weave-only", "pix_fmt": "yuv422p10le"},
            "registration": {
                "affine": [[1.001, 0.0, 2.5], [0.0, 1.0, -1.0]],
                "residuals_px": {"mean": residual_mean, "max": residual_mean * 2},
                "inliers": 7, "total": 8,
                "landmarks_used": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
                "quality_flag": quality,
                "quality_reason": None,
            },
            "tool_version": "tp_measure 0.1",
            "tp_chart_version": 1,
            "ideal_frame_md5": "abc123",
        },
        "tartan": [
            {"id": "YEL", "name": "yellow_75", "ideal_yuv10": [646, 176, 567],
             "measured_yuv10": [640, 173, 569], "delta_yuv10": [-6, -3, 2],
             "sat_pct_vs_ideal": 76.5, "patch_size_px": [3, 3],
             "patch_center_capture_xy": [15, 13]},
        ],
        "grays": [
            {"id": "G1", "name": "gray_step_20", "ideal_y10": 239.2,
             "measured_y10": 234.0, "delta_y10": -5.2,
             "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6]},
        ],
    }


def test_render_registration_summary_contains_per_capture_data():
    a = _make_capture_json("alpha", residual_mean=0.4, quality="ok")
    b = _make_capture_json("beta", residual_mean=2.5, quality="warn")
    html = tp_compare.render_registration_summary([a, b])
    assert "alpha.mov" in html
    assert "beta.mov" in html
    assert "0.40" in html  # residual mean
    assert "warn" in html


TESTS = [
    test_render_registration_summary_contains_per_capture_data,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(TESTS)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_compare.py`

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement skeleton + registration summary**

Create `tp_compare.py`:

```python
#!/usr/bin/env python3
"""tp_compare: render multi-capture SW2 measurement comparison HTML.

CLI:
    python tp_compare.py cap1.json cap2.json ... --output report.html
"""

from __future__ import annotations
import html as _h
import json
from typing import Any, Dict, List


def _basename(path: str) -> str:
    import os
    return os.path.basename(path)


def render_registration_summary(captures: List[Dict[str, Any]]) -> str:
    rows = []
    for c in captures:
        m = c["_meta"]
        reg = m["registration"]
        affine = reg["affine"]
        if affine is not None:
            tx = affine[0][2]
            ty = affine[1][2]
            sx = affine[0][0]
            sy = affine[1][1]
            shear = affine[0][1]
            transform = (
                f"tx={tx:+.2f} ty={ty:+.2f} "
                f"sx={sx:.4f} sy={sy:.4f} shear={shear:+.4f}"
            )
        else:
            transform = "<i>registration failed</i>"
        flag = reg.get("quality_flag", "?")
        flag_class = {"ok": "ok", "warn": "warn", "failed": "bad"}.get(flag, "")
        residuals = reg["residuals_px"]
        warn_prog = " (progressive!)" if m.get("progressive_warning") else ""
        rows.append(
            f"<tr>"
            f"<td>{_h.escape(_basename(m['capture']))}{warn_prog}</td>"
            f"<td>{m['raster_in'][0]}x{m['raster_in'][1]}</td>"
            f"<td>{m.get('field_order', 'unknown')}</td>"
            f"<td class='{flag_class}'>{flag}</td>"
            f"<td>{residuals['mean']:.2f}</td>"
            f"<td>{residuals['max']:.2f}</td>"
            f"<td>{reg['inliers']}/{reg['total']}</td>"
            f"<td><code>{transform}</code></td>"
            f"</tr>"
        )
    table_body = "\n".join(rows)
    return f"""
<section class="registration">
  <h2>Registration Summary</h2>
  <table class="data">
    <thead>
      <tr>
        <th>Capture</th><th>Raster</th><th>Field</th>
        <th>Quality</th><th>Mean&nbsp;px</th><th>Max&nbsp;px</th>
        <th>Inliers</th><th>Affine</th>
      </tr>
    </thead>
    <tbody>
{table_body}
    </tbody>
  </table>
</section>
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_compare.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tp_compare.py test_cases/test_tp_compare.py
git commit -m "feat(sw2): tp_compare skeleton + registration summary section"
```

---

## Task 14: `tp_compare.py` — Tartan deltas table

**Files:**
- Modify: `tp_compare.py`
- Modify: `test_cases/test_tp_compare.py`

- [ ] **Step 1: Add failing test**

Append to `test_cases/test_tp_compare.py`:

```python
def test_render_tartan_deltas_contains_swatches_and_deltas():
    a = _make_capture_json("alpha")
    b = _make_capture_json("beta")
    # Mutate one delta in beta to verify it gets rendered.
    b["tartan"][0]["delta_yuv10"] = [-50.0, 5.0, -2.0]
    b["tartan"][0]["measured_yuv10"] = [596.0, 181.0, 565.0]
    html = tp_compare.render_tartan_deltas([a, b])
    assert "YEL" in html
    assert "-50" in html or "-50.0" in html
    # Should include an inline-style swatch background-color (rgb)
    assert "background-color: rgb(" in html or "background:rgb(" in html


TESTS.append(test_render_tartan_deltas_contains_swatches_and_deltas)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_compare.py`

Expected: AttributeError on `render_tartan_deltas`.

- [ ] **Step 3: Implement tartan deltas section**

Append to `tp_compare.py`:

```python
import tp_chart


def _swatch(rgb_tuple) -> str:
    r, g, b = rgb_tuple
    return f"<span class='swatch' style='background-color: rgb({r},{g},{b});'></span>"


def _delta_class(d: float) -> str:
    a = abs(d)
    if a < 5:
        return "delta-good"
    if a < 15:
        return "delta-warn"
    return "delta-bad"


def render_tartan_deltas(captures: List[Dict[str, Any]]) -> str:
    # Use TARTAN_REGIONS as the canonical column ordering.
    region_ids = [r["id"] for r in tp_chart.TARTAN_REGIONS]

    head = "<tr><th>Capture</th>" + "".join(
        f"<th>{rid}</th>" for rid in region_ids
    ) + "</tr>"

    body_rows = []
    for c in captures:
        cap_name = _basename(c["_meta"]["capture"])
        cells = [f"<td>{_h.escape(cap_name)}</td>"]
        by_id = {p["id"]: p for p in c["tartan"]}
        for rid in region_ids:
            p = by_id.get(rid)
            if p is None:
                cells.append("<td>-</td>")
                continue
            ideal_rgb = tp_chart.yuv10_to_rgb8(*p["ideal_yuv10"])
            meas_rgb = tp_chart.yuv10_to_rgb8(*p["measured_yuv10"])
            dy, du, dv = p["delta_yuv10"]
            cls = _delta_class(dy)
            cells.append(
                f"<td class='{cls}'>"
                f"{_swatch(ideal_rgb)}{_swatch(meas_rgb)}"
                f"<div class='delta'>"
                f"&Delta;Y={dy:+.1f}<br>"
                f"&Delta;U={du:+.1f}<br>"
                f"&Delta;V={dv:+.1f}"
                f"</div></td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    body = "\n".join(body_rows)
    return f"""
<section class="tartan">
  <h2>Tartan Deltas (measured vs ideal)</h2>
  <p>Each cell shows ideal swatch | measured swatch and the YUV10 deltas.</p>
  <table class="data tartan-table">
    <thead>{head}</thead>
    <tbody>
{body}
    </tbody>
  </table>
</section>
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_compare.py`

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_compare.py test_cases/test_tp_compare.py
git commit -m "feat(sw2): tp_compare tartan deltas table with swatches"
```

---

## Task 15: `tp_compare.py` — Gray deltas + linearity plot

**Files:**
- Modify: `tp_compare.py`
- Modify: `test_cases/test_tp_compare.py`

- [ ] **Step 1: Add failing test**

Append to `test_cases/test_tp_compare.py`:

```python
def test_render_gray_deltas_contains_table_and_chart_data():
    a = _make_capture_json("alpha")
    a["grays"] = [
        {"id": "G1", "name": "gray_step_20", "ideal_y10": 239.2,
         "measured_y10": 234.0, "delta_y10": -5.2, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
        {"id": "G2", "name": "gray_step_40", "ideal_y10": 414.4,
         "measured_y10": 406.0, "delta_y10": -8.4, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
        {"id": "G3", "name": "gray_step_60", "ideal_y10": 589.6,
         "measured_y10": 579.0, "delta_y10": -10.6, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
        {"id": "G4", "name": "gray_step_80", "ideal_y10": 764.8,
         "measured_y10": 752.0, "delta_y10": -12.8, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
    ]
    html = tp_compare.render_gray_deltas([a])
    assert "G1" in html and "G4" in html
    assert "-5.2" in html or "-5.20" in html
    assert "-12.8" in html or "-12.80" in html
    # Chart.js data block
    assert "Chart" in html
    assert "239.2" in html  # ideal Y10 for G1


TESTS.append(test_render_gray_deltas_contains_table_and_chart_data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_compare.py`

Expected: AttributeError on `render_gray_deltas`.

- [ ] **Step 3: Implement gray deltas + linearity plot**

Append to `tp_compare.py`:

```python
def render_gray_deltas(captures: List[Dict[str, Any]]) -> str:
    region_ids = [r["id"] for r in tp_chart.GRAY_REGIONS]
    head = "<tr><th>Capture</th>" + "".join(
        f"<th>{rid}</th>" for rid in region_ids
    ) + "</tr>"

    body_rows = []
    for c in captures:
        cap_name = _basename(c["_meta"]["capture"])
        cells = [f"<td>{_h.escape(cap_name)}</td>"]
        by_id = {g["id"]: g for g in c["grays"]}
        for rid in region_ids:
            g = by_id.get(rid)
            if g is None:
                cells.append("<td>-</td>")
                continue
            cls = _delta_class(g["delta_y10"])
            cells.append(
                f"<td class='{cls}'>"
                f"&Delta;Y={g['delta_y10']:+.2f}"
                f"<div class='small'>Y10={g['measured_y10']:.1f}</div>"
                f"</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Chart.js linearity plot: x = ideal Y10, y = measured Y10, one dataset per capture.
    ideals = [tp_chart.GRAY_IDEAL_Y10[i] for i in range(4)]
    datasets = []
    for c in captures:
        by_id = {g["id"]: g for g in c["grays"]}
        cap_name = _basename(c["_meta"]["capture"])
        ys = [by_id.get(rid, {}).get("measured_y10", None) for rid in region_ids]
        datasets.append({"label": cap_name, "data": ys})
    chart_data = {
        "labels": [f"{v:.1f}" for v in ideals],
        "datasets": datasets,
        "ideal": ideals,
    }
    chart_json = json.dumps(chart_data)

    return f"""
<section class="gray">
  <h2>Gray Step Deltas + Linearity</h2>
  <table class="data gray-table">
    <thead>{head}</thead>
    <tbody>
{''.join(body_rows)}
    </tbody>
  </table>
  <div class="chart-wrap">
    <canvas id="grayLinearity" width="640" height="320"></canvas>
  </div>
  <script>
    (function() {{
      const data = {chart_json};
      const datasets = data.datasets.map(function(ds) {{
        return {{
          label: ds.label,
          data: ds.data,
          fill: false,
          tension: 0.0,
        }};
      }});
      // Add ideal as the reference line.
      datasets.unshift({{label: "ideal", data: data.ideal, borderDash: [5, 5], fill: false}});
      const ctx = document.getElementById("grayLinearity").getContext("2d");
      new Chart(ctx, {{
        type: "line",
        data: {{labels: data.labels, datasets: datasets}},
        options: {{responsive: false, scales: {{y: {{title: {{display: true, text: "measured Y10"}}}}, x: {{title: {{display: true, text: "ideal Y10"}}}}}}}}
      }});
    }})();
  </script>
</section>
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_compare.py`

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_compare.py test_cases/test_tp_compare.py
git commit -m "feat(sw2): tp_compare gray deltas + linearity plot"
```

---

## Task 16: `tp_compare.py` — Page assembly + CLI

**Files:**
- Modify: `tp_compare.py`
- Modify: `test_cases/test_tp_compare.py`

- [ ] **Step 1: Add failing test**

Append to `test_cases/test_tp_compare.py`:

```python
def test_compare_cli_writes_html(tmp_dir):
    import subprocess
    a = _make_capture_json("alpha")
    b = _make_capture_json("beta")
    a_path = os.path.join(tmp_dir, "a.json")
    b_path = os.path.join(tmp_dir, "b.json")
    out_path = os.path.join(tmp_dir, "report.html")
    with open(a_path, "w") as f:
        json.dump(a, f)
    with open(b_path, "w") as f:
        json.dump(b, f)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        ["python", "tp_compare.py", a_path, b_path, "--output", out_path],
        check=True, cwd=project_root,
    )
    with open(out_path) as f:
        html = f.read()
    assert "<html" in html
    assert "Registration Summary" in html
    assert "Tartan Deltas" in html
    assert "Gray Step Deltas" in html
    assert "Chart.js" in html or "chart.js" in html  # CDN script reference


# Switch test runner to support tmp_dir for new test
TESTS_NO_TMPDIR = [
    test_render_registration_summary_contains_per_capture_data,
    test_render_tartan_deltas_contains_swatches_and_deltas,
    test_render_gray_deltas_contains_table_and_chart_data,
]
TESTS_TMPDIR = [test_compare_cli_writes_html]


def main():
    failed = 0
    for t in TESTS_NO_TMPDIR:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    tmp = tempfile.mkdtemp(prefix="tp_compare_test_")
    try:
        for t in TESTS_TMPDIR:
            try:
                t(tmp); print(f"PASS  {t.__name__}")
            except Exception as e:
                failed += 1; print(f"FAIL  {t.__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(TESTS_NO_TMPDIR) + len(TESTS_TMPDIR)
    if failed:
        print(f"\n{failed}/{total} tests failed"); sys.exit(1)
    print(f"\nAll {total} tests passed")
```

(Replace the old `TESTS = [...]` and `main()` block at the bottom of the file with the above.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_cases/test_tp_compare.py`

Expected: `tp_compare.py` has no `__main__`.

- [ ] **Step 3: Implement page assembly + CLI**

Append to `tp_compare.py`:

```python
_CSS = """
body { background: #181a1f; color: #d8dde6; font-family: system-ui, sans-serif; margin: 24px; }
h1 { color: #fff; }
h2 { color: #fff; border-bottom: 1px solid #2a2e36; padding-bottom: 4px; }
table.data { border-collapse: collapse; margin: 12px 0; }
table.data th, table.data td { border: 1px solid #2a2e36; padding: 6px 10px; vertical-align: top; }
table.data th { background: #21252b; color: #fff; }
.swatch { display: inline-block; width: 18px; height: 18px; border: 1px solid #444; margin-right: 4px; vertical-align: middle; }
.delta { font-size: 11px; margin-top: 4px; color: #b8c0cc; }
.delta-good { background: rgba(80,200,120,0.10); }
.delta-warn { background: rgba(240,180,80,0.15); }
.delta-bad  { background: rgba(220,80,80,0.18); }
.ok   { color: #61c08f; font-weight: 600; }
.warn { color: #f0b450; font-weight: 600; }
.bad  { color: #e26464; font-weight: 600; }
.small { font-size: 11px; color: #b8c0cc; }
code { color: #c5d1e0; }
"""


def render_page(captures: List[Dict[str, Any]]) -> str:
    title = f"SW2 Comparison — {len(captures)} captures"
    sections = (
        render_registration_summary(captures)
        + render_tartan_deltas(captures)
        + render_gray_deltas(captures)
    )
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{_h.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{_CSS}</style>
</head><body>
<h1>{_h.escape(title)}</h1>
{sections}
</body></html>
"""


def _main():
    import argparse, sys
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+", help="per-capture JSON files from tp_measure")
    p.add_argument("--output", required=True, help="output HTML path")
    args = p.parse_args()
    captures = []
    for path in args.inputs:
        with open(path) as f:
            captures.append(json.load(f))
    html = render_page(captures)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output} ({len(captures)} captures)")


if __name__ == "__main__":
    _main()
```

Also at the top of `tp_compare.py` add a `sys` import if missing (the `_main` uses `argparse`; tests use `sys.exit` already imported via the test header).

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_cases/test_tp_compare.py`

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_compare.py test_cases/test_tp_compare.py
git commit -m "feat(sw2): tp_compare page assembly + CLI"
```

---

## Task 17: End-to-end smoke test on the real DVD-rip sample

**Files:**
- Modify (read only — capture results):  `/wintmp/analog_video/tp_compare/sample/sw2_dvdrip_sample.mov`
- Create: `tp_smoke_outputs/dvdrip.json`, `tp_smoke_outputs/report.html` (transient artifacts; do NOT commit)

The point of this task is to validate that the full pipeline runs end-to-end on a real capture and produces sensible output. It is NOT a unit test — it's a manual sanity check whose deviations from synthesized-ideal expectations may indicate that ideal coordinates need calibration.

- [ ] **Step 1: Run all unit tests one more time**

Run, in this order:

```bash
python test_cases/test_tp_chart.py
python test_cases/test_tp_synthesize.py
python test_cases/test_tp_register.py
python test_cases/test_tp_measure.py
python test_cases/test_tp_compare.py
python test_cases/test_metrics.py   # existing — verify untouched
```

Expected: every script reports all-pass and exits 0.

- [ ] **Step 2: Synthesize the ideal frame as a PNG for visual reference**

```bash
mkdir -p tp_smoke_outputs
python tp_synthesize.py --raster 720x486 --output tp_smoke_outputs/ideal.png
```

Expected: file exists, opens to a 720x486 image showing grey background, black grid, the top-left tartan + gray strip + boundary triangle.

- [ ] **Step 3: Run tp_measure on the DVD-rip sample**

```bash
python tp_measure.py /wintmp/analog_video/tp_compare/sample/sw2_dvdrip_sample.mov \
    --frame 60 --output tp_smoke_outputs/dvdrip.json
```

Expected output to stdout includes a registration line such as:
```
wrote tp_smoke_outputs/dvdrip.json: registration=ok, residuals_mean=...px
```

If `quality_flag=failed`, registration could not be done — STOP and report. Likely root cause: GRID_LANDMARKS coords don't fall on grid intersections in the real chart layout. Capture a screenshot of `tp_smoke_outputs/ideal.png` next to a frame extracted from the dvdrip sample (use the codex `extract_frame_assets` pattern or `ffmpeg ... -vf select=eq(n\,60) -frames:v 1 dvdrip_60.png`) and bring back to the spec author for landmark recalibration.

- [ ] **Step 4: Inspect the per-capture JSON values for sanity**

Open `tp_smoke_outputs/dvdrip.json`. Confirm:

- `_meta.raster_in == [720, 480]` and `_meta.padding_offsets.top == _meta.padding_offsets.bottom == 3`.
- `_meta.registration.quality_flag` is `"ok"` or `"warn"` (not `"failed"`).
- `_meta.registration.inliers >= 5` out of 8.
- Each tartan patch has `measured_yuv10` close to `ideal_yuv10` (within ~50 codes per channel; large deviations on YEL/CYN/BLU/RED indicate ideal-coord drift, not necessarily processor flaws).
- Gray-step `measured_y10` values are MONOTONICALLY INCREASING from G1 to G4. (If not, registration or coord drift is suspect.)

- [ ] **Step 5: Run tp_compare with one capture and inspect the report**

```bash
python tp_compare.py tp_smoke_outputs/dvdrip.json --output tp_smoke_outputs/report.html
```

Open `tp_smoke_outputs/report.html` in a browser. Confirm three sections render: Registration Summary, Tartan Deltas (with swatch pairs), and Gray Step Deltas with a Chart.js linearity plot.

- [ ] **Step 6: (Optional) Compare two captures**

If a second capture is available (e.g., `/wintmp/analog_video/tp_compare/tpgSw2_composite_snellhdSnellGroup_pedIn_sdi.avi`):

```bash
python tp_measure.py /wintmp/analog_video/tp_compare/tpgSw2_composite_snellhdSnellGroup_pedIn_sdi.avi \
    --frame 60 --output tp_smoke_outputs/snellhd.json
python tp_compare.py tp_smoke_outputs/dvdrip.json tp_smoke_outputs/snellhd.json \
    --output tp_smoke_outputs/report.html
```

Inspect: deltas should differ between the two captures, and the Chart.js plot should show two lines next to the dashed ideal.

- [ ] **Step 7: Commit (.gitignore guards the smoke artifacts)**

The repo already ignores `tp_smoke_outputs/` via the broad cache/report ignore added in `2729d09 chore: broaden .gitignore for caches and generated reports`. Verify with `git status` — the `tp_smoke_outputs/` directory should not appear as untracked. If it does, add it to `.gitignore` in this final commit:

```bash
# Only if needed (verify with git status first)
echo "tp_smoke_outputs/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore tp smoke-test output directory"
```

If nothing needs to be added, do not create an empty commit.

---

## Self-Review

**Spec coverage check** (against `docs/superpowers/specs/2026-05-10-sw2-tp-compare-design.md`):

- Architecture (3 sibling scripts + tp_chart.py): Tasks 1–16 ✓
- Resolution handling (pad 720×480 to 720×486): Task 10 ✓
- Reference values (BT.601 limited range, gray ideals, grey background): Task 1 ✓
- Region table + landmark catalog: Task 2 ✓
- Synthesizer scope (grey + grid + tartan + grays + boundary triangle): Tasks 3–6 ✓
- Registration (grid-line intersections, RANSAC affine, quality flag): Tasks 7–9 ✓
- tp_measure pipeline + JSON shape: Tasks 10–12 ✓
- Comparison HTML (3 sections + linearity plot): Tasks 13–16 ✓
- Testing (synthetic + end-to-end): Tasks 1–16 plus smoke in Task 17 ✓
- Stages 2/3 are explicitly out of scope per the spec. ✓
- Open issues (residual thresholds, NTSC triangle polarity, ranking semantics, frame-selection policy, grey-level offset, patch-size calibration): all left to follow-up; placeholder values are clearly noted in code comments.

**Type / signature consistency:**
- `synthesize(width, height) -> (Y, U, V)` — used the same way in tp_synthesize, tp_measure (`_ideal_for_md5`), and tests.
- `detect_landmark(Y, ideal_x, ideal_y, search_window_px)` — Y is the luma plane only; consistent everywhere.
- `fit_affine(detected_pts, ideal_pts)` returns dict with keys `affine_matrix`, `residuals_px`, `inliers`, `total` — used by `register()` which then adds `landmarks_used` and `quality_flag`. Consistent.
- `sample_region(Y, U, V, region, affine)` — returns dict with `id`, `name`, `ideal_yuv10`, `measured_yuv10`, `delta_yuv10`, `sat_pct_vs_ideal`, `patch_size_px`, `patch_center_capture_xy`. tp_measure adapts gray output by reshaping into a flat `delta_y10`. Consistent.
- HTML rendering functions `render_registration_summary`, `render_tartan_deltas`, `render_gray_deltas`, `render_page` all accept a list of capture dicts. Consistent.

**Placeholder scan:** No `TBD`, `TODO`, or "implement later" tokens. Every step shows the exact code or command. The Stage 2/3 work is intentionally out of plan-scope (per spec).
