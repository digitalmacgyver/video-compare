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
