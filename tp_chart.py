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
# Top-left tartan + gray region (Stage 1 scope). Centers are calibrated
# from the Snell HD SDI capture (see tp_calibrate.py + the snellhd_calib.json
# checked in under tp_smoke_outputs/).
#
#   Top tartan row    y ~10 :  YEL    CYN    BLU    RED        (75% colors)
#   Bottom tartan row y ~29 :  MAG    GRN    RED2   CYN2       (75% colors)
#   Gray strip        y ~44 :  G1     G2     G3     G4         (20/40/60/80% IRE)
#
# The bottom tartan row contains the OTHER 75% SMPTE colors (magenta, green)
# plus repeats of red and cyan to create vertical chroma transitions between
# rows -- the comb-decoder vertical-transient test the SW2 spec describes.
# Earlier code mislabelled this row as "low saturation"; the codex starter's
# y=40 sample window straddled the bottom tartan AND the gray strip, mixing
# saturated chroma with neutral gray to produce values that *looked* like a
# low-sat row. Calibration against real captures confirmed the bottom row is
# full 75% saturation.
#
# Box dimensions are smaller than the previous 30x27 guess to match the
# actual content height in the chart (rows are ~18-20 px tall, gray strip is
# ~10 px tall). Sample window = box * sample.size_frac (defaults to 0.2).

TARTAN_BOX_W = 30
TARTAN_BOX_H = 18
GRAY_BOX_W = 30
GRAY_BOX_H = 10

# (id, name, R, G, B) at 75% saturation. Same Rec.601 math for top and bottom;
# the difference is only the SMPTE color picked per column.
_TARTAN_TOP_COLORS = [
    ("YEL",  "yellow_75",   0.75, 0.75, 0.00),
    ("CYN",  "cyan_75",     0.00, 0.75, 0.75),
    ("BLU",  "blue_75",     0.00, 0.00, 0.75),
    ("RED",  "red_75",      0.75, 0.00, 0.00),
]

_TARTAN_BOT_COLORS = [
    ("MAG",  "magenta_75",  0.75, 0.00, 0.75),
    ("GRN",  "green_75",    0.00, 0.75, 0.00),
    ("RED2", "red_75_b",    0.75, 0.00, 0.00),   # same color as top RED, different position
    ("CYN2", "cyan_75_b",   0.00, 0.75, 0.75),   # same color as top CYN, different position
]

# Calibrated centers in 720x486 ideal coords (see tp_smoke_outputs/snellhd_calib.json).
_TARTAN_TOP_CENTERS = [(16, 10), (46, 10), (76,  9), (108, 9)]
_TARTAN_BOT_CENTERS = [(16, 28), (47, 29), (77, 29), (107, 29)]
_GRAY_CENTERS       = [(17, 44), (47, 44), (77, 43), (106, 44)]

_SAMPLE = {"kind": "center_window", "size_frac": 0.2}


def _box_from_center(cx: int, cy: int, w: int, h: int) -> tuple:
    """Convert a (cx, cy, w, h) into an (x, y, w, h) box anchored at the
    top-left. Used to fit the existing region "ideal_box" schema.

    The starting x is snapped to even so the box satisfies the 4:2:2 chroma
    alignment contract enforced by tp_synthesize._fill_box_yuv422. With w=30
    (even), `cx - 15` is odd when cx is even — snap right by 1 px in that
    case. The resulting <=1 px box shift is well inside the flat interior
    of the tartan/gray content, so sampling is unaffected.
    """
    x = cx - w // 2
    if x % 2 != 0:
        x += 1
    return (x, cy - h // 2, w, h)


def _build_tartan_regions():
    regions = []
    for (rid, name, r, g, b), (cx, cy) in zip(_TARTAN_TOP_COLORS, _TARTAN_TOP_CENTERS):
        y10, u10, v10 = rgb_norm_to_yuv10(r, g, b)
        regions.append({
            "id": rid,
            "name": name,
            "kind": "tartan_rect",
            "ideal_box": _box_from_center(cx, cy, TARTAN_BOX_W, TARTAN_BOX_H),
            "expected": {"y10": y10, "u10": u10, "v10": v10},
            "sample": _SAMPLE.copy(),
        })
    for (rid, name, r, g, b), (cx, cy) in zip(_TARTAN_BOT_COLORS, _TARTAN_BOT_CENTERS):
        y10, u10, v10 = rgb_norm_to_yuv10(r, g, b)
        regions.append({
            "id": rid,
            "name": name,
            "kind": "tartan_rect",
            "ideal_box": _box_from_center(cx, cy, TARTAN_BOX_W, TARTAN_BOX_H),
            "expected": {"y10": y10, "u10": u10, "v10": v10},
            "sample": _SAMPLE.copy(),
        })
    return regions


def _build_gray_regions():
    return [
        {
            "id": f"G{i+1}",
            "name": f"gray_step_{int((i + 1) * 20)}",
            "kind": "gray_step",
            "ideal_box": _box_from_center(
                _GRAY_CENTERS[i][0], _GRAY_CENTERS[i][1],
                GRAY_BOX_W, GRAY_BOX_H,
            ),
            "expected": {"y10": GRAY_IDEAL_Y10[i], "u10": 512, "v10": 512},
            "sample": _SAMPLE.copy(),
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
