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
# Values pre-computed (64 + N*876) to avoid floating-point accumulation errors.
GRAY_IDEAL_Y10: Tuple[float, float, float, float] = (239.2, 414.4, 589.6, 764.8)


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
