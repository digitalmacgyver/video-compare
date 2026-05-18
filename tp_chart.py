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

# Calibrated centers in 720x486 ideal coords.
# - Initial values from tp_smoke_outputs/snellhd_calib.json (May 10 2026).
# - Operator nudge May 11 2026: shift YEL/CYN +2y, BLU/RED +3y, MAG/GRN/RED2/CYN2 +1y,
#   G1/G2 +2y, G3/G4 +3y -- previous values landed slightly above the visual
#   center of each box, with G3/G4 close enough to the bottom-tartan row above
#   that gray-sampling picked up a chroma cast.
_TARTAN_TOP_CENTERS = [(16, 12), (46, 12), (76, 12), (108, 12)]
_TARTAN_BOT_CENTERS = [(16, 29), (47, 30), (77, 30), (107, 30)]
_GRAY_CENTERS       = [(17, 46), (47, 46), (77, 46), (106, 47)]

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
# IDEAL PICTURE BOX
# =====================================================================
#
# Active picture extent in ideal 720x486 coords (full SDI raster). Used by
# detect_geometry to compare the apex-derived active picture box (in
# capture coords) against where the chart's ideal active picture would
# appear in capture coords after the final affine.

IDEAL_PICTURE_BOX = {"left": 0, "top": 0, "right": 719, "bottom": 485}
IDEAL_PICTURE_BOX_CORNERS = [
    (IDEAL_PICTURE_BOX["left"],  IDEAL_PICTURE_BOX["top"]),     # TL
    (IDEAL_PICTURE_BOX["right"], IDEAL_PICTURE_BOX["top"]),     # TR
    (IDEAL_PICTURE_BOX["left"],  IDEAL_PICTURE_BOX["bottom"]),  # BL
    (IDEAL_PICTURE_BOX["right"], IDEAL_PICTURE_BOX["bottom"]),  # BR
]


# =====================================================================
# REGISTRATION LANDMARK CATALOG
# =====================================================================
#
# Black grid intersections on the grey background. 12 anchors in the safe
# interior of the chart (avoiding the tartan/gray strip on the upper-left,
# the busy top of the chart, and the chart border).
#
# Grid spacing: 60 px (x) × 54 px (y), so valid intersections are multiples
# of 60 and 54.
#
# Diagnostic context: the Stage 1 catalog (8 anchors including L8 at
# (360, 378) and L1-L3 at y=54) gave only 4 of 8 surviving RANSAC -- L8 has
# no clean intersection, the top row was near the boundary triangles and
# tartan and showed 1-4 px noise. The y∈{108..270} interior gives sub-px
# residuals against a single-affine fit on the real captures.

_LANDMARK_GRID = [
    # (id, ideal_x, ideal_y) — May 2026 catalog refresh after chart-layout
    # review (docs/sw2_chart_layout.md). Anchors dropped:
    #   L2  (360, 108): the cells above L2 are the merged 2-wide "SW2/NTSC"
    #                   text box (cells (2,6)+(2,7)) with no vertical grid
    #                   line at x=360 — no proper "+" intersection here.
    #   L5  (360, 162), L6 (480, 162),
    #   L8  (360, 216), L11 (420, 270): all sit inside the moving zone-plate
    #                   reserved area (cells (3,4)–(6,9)) where the chart
    #                   pattern is non-deterministic per frame.
    # Added:
    #   L13, L14 at (180, 378) and (540, 378) — shared corners of empty /
    #   circle-only cells just below the zone plate and above the S&W banner.
    #   L15, L16 at (180, 54) and (540, 54) — top-row anchors near the gray
    #   strip / upper-left composite, intended to constrain the affine close
    #   to the y≈46 gray-strip sample row and prevent extrapolation drift.
    # Note: intersection points are 2x2-px dark regions (grid lines are 2 px
    # wide w/ ~1 px antialias falloff), so detected positions can land on
    # any of the 4 corner pixels — operator-observed ±1 px wobble is expected.
    ("L1",  240, 108),
    ("L3",  480, 108),
    ("L4",  180, 162),
    ("L7",  180, 216),
    ("L9",  540, 216),
    ("L10", 180, 270),
    ("L12", 540, 270),
    ("L13", 180, 378),
    ("L14", 540, 378),
    ("L15", 180,  54),
    ("L16", 540,  54),
]

GRID_LANDMARKS = [
    {
        "id": rid,
        "kind": "grid_intersection",
        "ideal_x": x,
        "ideal_y": y,
        "search_window_px": 24,
    }
    for rid, x, y in _LANDMARK_GRID
]


# =====================================================================
# BOUNDARY TRIANGLES (Stage 2)
# =====================================================================
#
# 4 corner-region triangles that mark the picture boundaries. TL/TR point
# UP (apex toward top edge); BL/BR point DOWN (apex toward bottom edge).
# The base of each triangle (back edge, furthest from the picture edge it
# marks) sits 30 px inside the chart from the corresponding edge.
#
# Registration uses the back corners (clip-resistant). The apex can be
# either detected from data or inferred from back_midpoint + chart-spec
# offset (apex - back_midpoint).

# Triangle dimensions from operator calibration against the snellhd SDI
# capture (tp_smoke_outputs/stage2_calib.json, May 11 2026): triangles are
# smaller than the canonical SW2 spec implied. Real-chart parameters:
_BASE_HALF = 8    # half the back-edge width (16 px total)
_HEIGHT    = 15   # apex distance from back_midpoint
_BACK_INSET = 16  # back edge sits 16 px inside the chart from the edge


def _make_triangle(rid, orientation, x_center, edge):
    """Build a triangle entry. `edge` is the chart edge it marks
    ('top' or 'bottom'); the back edge sits `_BACK_INSET` px inside
    that edge; the apex sits at the edge."""
    if edge == "top":
        back_y = _BACK_INSET
        apex_y = back_y - _HEIGHT
    elif edge == "bottom":
        back_y = 485 - _BACK_INSET
        apex_y = back_y + _HEIGHT
    else:
        raise ValueError(edge)
    bc1 = (x_center - _BASE_HALF, back_y)
    bc2 = (x_center + _BASE_HALF, back_y)
    bm  = (x_center, back_y)
    apex = (x_center, apex_y)
    return {
        "id": rid,
        "kind": "boundary_triangle",
        "orientation": orientation,
        "ideal_back_corner_1": bc1,
        "ideal_back_corner_2": bc2,
        "ideal_back_midpoint": bm,
        "ideal_apex": apex,
        "search_window_px": 40,
    }


# Triangle x-centers from operator calibration against snellhd. The real
# chart is SYMMETRIC top/bottom (not asymmetric on the bottom edge as I'd
# initially inferred from the cell-border reading) — all four triangle
# midpoints sit at x=181 (left) or x=538 (right).
BOUNDARY_TRIANGLES = [
    _make_triangle("TL", "apex_up",   181, "top"),
    _make_triangle("TR", "apex_up",   538, "top"),
    _make_triangle("BL", "apex_down", 181, "bottom"),
    _make_triangle("BR", "apex_down", 538, "bottom"),
]


# =====================================================================
# REGISTRATION CROSS (Stage 2)
# =====================================================================
#
# Picture-center sub-pixel registration feature: a black box containing a
# centered white "+". Used for sub-pixel registration accuracy and for
# detecting directionally-biased aperture / sharpening filters via the
# horizontal-vs-vertical arm-length asymmetry.

# Position from operator calibration against snellhd: cross is at (568, 36)
# in the upper-right composite (cells (1,10)/(1,11) area, below the top-row
# tartan strip rather than next to it).
REGISTRATION_CROSS = {
    "id": "RC",
    "kind": "registration_cross",
    "ideal_x": 568,
    "ideal_y": 36,
    "ideal_arm_len_px": 17,    # tip-to-tip length of each arm
    "ideal_arm_thickness_px": 3,
    "box_size_px": 24,
    "search_window_px": 40,
}


# =====================================================================
# BLACK CIRCLE (Stage 2)
# =====================================================================
#
# Black ring centered on the picture. Diameter = picture height (486 px),
# ring thickness 168 ns @ 13.5 MHz ~ 3 samples. Used to measure aspect
# ratio (rx vs ry) and any picture-vs-spec scaling drift.

# Operator calibration confirmed the real chart is rendered with NTSC
# 10:11 PAR baked in — the captured ring is elliptical (rx ≈ 266, ry ≈ 242,
# ratio ≈ 11/10). We keep the catalog and the synthesizer at the chart-
# spec literal (round circle, radius = picture_height / 2 = 243), so the
# synthesized fixture is round and clean. On real captures the elliptical
# ring still detects within the annulus on its y-axis sides; cv2.fitEllipse
# returns biased rx/ry that under-report the PAR aspect (aspect_ratio_check
# ≈ 1.0 not ≈ 0.91 — a known limitation; flag for future PAR-aware fit).
BLACK_CIRCLE = {
    "id": "BC",
    "kind": "black_circle",
    "ideal_cx": 359,
    "ideal_cy": 243,
    "expected_radius_px": 243,
    "ring_thickness_px": 3,
    "search_band_px": 15,
}


# =====================================================================
# BURST / WEDGE REGIONS (Stage 3 — frequency response)
# =====================================================================
#
# Frequency probes drawn elsewhere on the chart. Each region is sampled
# (after Stage 2 affine) and FFT-analyzed by tp_freq.measure().
#
# Coordinates here are initial values picked from docs/sw2_chart_layout.md;
# operator calibration via tp_calibrate.py --preset stage3-bursts can refine
# them on real captures.

NTSC_SAMPLE_RATE_MHZ = 13.5  # horizontal sample rate for NTSC SDI 720-wide

# NTSC pixel aspect ratio: the 720x486 raster is displayed at 4:3, so each
# pixel is taller than wide by 10/11. A chart pattern that is logically a
# perfect circle therefore appears in raster coords as an ellipse with
# horizontal semi-axis = vertical semi-axis * 11/10. Used by the black-
# circle detector to size its elliptical annulus, and by tp_fiducial_crops
# to predict the visible ring path when the detector returns None.
NTSC_PAR_X_OVER_Y = 11.0 / 10.0

# 300/400 tvl diagonal bursts: TVL maps to horizontal frequency at
# fs/(2 * picture_width_px / tvl) ≈ ; we use plausible nominal values that
# match the labels visible on the chart. The classifier doesn't depend on
# absolute frequency, only relative modulation across regions.
# Boxes are 28x28 centered in each 60x54 cell so they leave a >=12-px
# margin from cell borders. This keeps bursts clear of grid-landmark
# search windows (24-px window = 12-px half-width) at neighboring cell
# corners.
_BURST_RAW = [
    # (id,                kind,             freq_MHz, ideal_box,          extras)
    ("BURST_3p58",        "burst_vertical", 3.58,     (76,  67,  28, 28), {}),
    ("BURST_4p43",        "burst_vertical", 4.43,     (616, 67,  28, 28), {}),
    ("BURST_4p286_SECAM", "burst_vertical", 4.286,    (76,  391, 28, 28), {}),
    ("BURST_300TVL_DIAG", "burst_diagonal", 3.95,     (256, 67,  28, 28), {"stripe_angle_deg": 45}),
    ("BURST_400TVL_DIAG", "burst_diagonal", 5.27,     (436, 67,  28, 28), {"stripe_angle_deg": 45}),
    # Continuous frequency wedge in column 10 (cells 3..6,10) spans
    # 1.5 MHz at y=108 to 5.5 MHz at y=324 (slope 1 MHz per 54 px).
    # Sample at 0.5 MHz intervals — 7 measurement points characterizes the
    # rolloff curve cleanly and falls within the wedge column on the chart.
    ("WEDGE_2p0MHz",      "wedge_segment",  2.0,      (556, 121, 28, 28), {}),
    ("WEDGE_2p5MHz",      "wedge_segment",  2.5,      (556, 148, 28, 28), {}),
    ("WEDGE_3MHz",        "wedge_segment",  3.0,      (556, 175, 28, 28), {}),
    ("WEDGE_3p5MHz",      "wedge_segment",  3.5,      (556, 202, 28, 28), {}),
    ("WEDGE_4MHz",        "wedge_segment",  4.0,      (556, 229, 28, 28), {}),
    ("WEDGE_4p5MHz",      "wedge_segment",  4.5,      (556, 256, 28, 28), {}),
    ("WEDGE_5MHz",        "wedge_segment",  5.0,      (556, 283, 28, 28), {}),
    # Row-9 Y/C timing chroma bursts (alternating chroma stripes at the
    # labeled frequency). Used to measure chroma bandwidth AND the cross-
    # luma (dot-crawl) that the decoder injects at chroma transitions.
    # 0.5 MHz (cells 9,6-7) is blue/yellow alternation; 1.0/1.5 MHz
    # (cells 9,5 and 9,8) are red/cyan alternation.
    ("YC_BURST_1p0MHZ",   "chroma_burst",   1.0,      (252, 446, 36, 28),
        {"color_pair": "red_cyan"}),
    ("YC_BURST_0p5MHZ",   "chroma_burst",   0.5,      (312, 446, 96, 28),
        {"color_pair": "blue_yellow"}),
    ("YC_BURST_1p5MHZ",   "chroma_burst",   1.5,      (432, 446, 36, 28),
        {"color_pair": "red_cyan"}),
]

# Continuous-wedge geometry for synth rendering and visual crops.
# Spans cells (3..6, 10) of the chart. Frequency rises linearly from
# `freq_top` at y=`y_top` to `freq_bottom` at y=`y_bottom`.
# `x` is inset by 16 px from the cell border at x=540 so neither the
# vertical grid line at x=540 nor grid landmark L9's 24-px search
# window (x=528..552) is contaminated by wedge stripes. Real captures
# already exhibit a similar inset; the existing 28-wide measurement
# boxes (x=556..584) fit cleanly inside.
WEDGE_COLUMN = {
    "x":           556,
    "y_top":       108,
    "y_bottom":    324,
    "width":       44,
    "freq_top":    1.5,
    "freq_bottom": 5.5,
}


def wedge_column_freq_at_y(y: float) -> float:
    """Return the local frequency (MHz) at vertical position y inside the
    continuous wedge. Used by tp_synthesize and by tp_compare's visual
    crop labeling to keep both sides consistent."""
    w = WEDGE_COLUMN
    t = (y - w["y_top"]) / (w["y_bottom"] - w["y_top"])
    return w["freq_top"] + t * (w["freq_bottom"] - w["freq_top"])


# =====================================================================
# CHROMA NON-LINEARITY STAIRCASE (cells 9,1 - 9,3)
# =====================================================================
#
# Three magenta boxes at increasing saturation: 33%, 66%, 100%. Each
# box is full magenta (R=B=N, G=0) so both chroma magnitude AND luma
# scale linearly with N. Used to measure chroma gain linearity (does
# chroma scale 1:2:3 as expected?) and differential phase (does the
# hue stay constant across the three boxes?).
#
# Sample windows are 28x18 centered in the 60x54 cells.

_CHROMA_STAIRCASE_RAW = [
    # (id,       level (0..1), center_xy)
    ("MAG_33",   1.0 / 3.0,    (30,  459)),
    ("MAG_66",   2.0 / 3.0,    (90,  459)),
    ("MAG_100",  1.0,          (150, 459)),
]


def _build_chroma_staircase_regions():
    out = []
    for rid, level, (cx, cy) in _CHROMA_STAIRCASE_RAW:
        r = level
        g = 0.0
        b = level
        y10, u10, v10 = rgb_norm_to_yuv10(r, g, b)
        out.append({
            "id":          rid,
            "level":       float(level),
            "ideal_rgb":   (r, g, b),
            "ideal_yuv10": (y10, u10, v10),
            "center_xy":   (cx, cy),
            "sample":      {"kind": "center_window",
                            "size": (28, 18)},
            "box":         _box_from_center(cx, cy, 28, 18),
        })
    return out


CHROMA_STAIRCASE_REGIONS = _build_chroma_staircase_regions()


BURST_REGIONS = [
    {
        "id":            rid,
        "kind":          kind,
        "frequency_MHz": freq,
        "ideal_box":     box,
        "sample":        {"kind": "center_window", "size_frac": 0.6},
        **extras,
    }
    for rid, kind, freq, box, extras in _BURST_RAW
]


# =====================================================================
# ARTIFACT REGIONS (Stage 3 — decoder-artifact detection)
# =====================================================================
#
# Boxes sampled for hanging-dots / dot-crawl / cross-color / cross-luma /
# zone-plate-chroma-leak metrics. Coordinates are initial values derived
# from the chart layout doc; operator calibration via tp_calibrate.py
# --preset stage3-artifacts can refine on real captures.
#
# All boxes use even x and even w to satisfy 4:2:2 alignment (so the U/V
# half-x crop indices are integers).

_ARTIFACT_RAW = [
    # id,                    artifact_kind,             ideal_box
    # HD strips sit in the grey row-8 cells immediately above row 9's
    # chroma blocks, avoiding cells (8,2) (burst) and (8,11) (wedge) which
    # have busy content that would swamp the cross-luma signal.
    ("HD_RED_TOP",           "hanging_dots",            (550, 427, 46, 5)),
    ("HD_MAGENTA_TOP",       "hanging_dots",            (10,  427, 46, 5)),
    ("DC_TARTAN_BELOW",      "dot_crawl",               (10,  64,  44,  40)),
    # XC bursts sample the same boxes the BURST_REGIONS bursts cover so we
    # have chroma-leak readings on all four row-2 frequency bursts plus
    # the two highest-frequency wedge probes.
    ("XC_BURST_3p58",        "cross_color",             (76,  67,  28,  28)),
    ("XC_BURST_4p43",        "cross_color",             (616, 67,  28,  28)),
    ("XC_BURST_300TVL",      "cross_color",             (256, 67,  28,  28)),
    ("XC_BURST_400TVL",      "cross_color",             (436, 67,  28,  28)),
    ("XC_WEDGE_2p0MHz",      "cross_color",             (556, 121, 28,  28)),
    ("XC_WEDGE_2p5MHz",      "cross_color",             (556, 148, 28,  28)),
    ("XC_WEDGE_3MHz",        "cross_color",             (556, 175, 28,  28)),
    ("XC_WEDGE_3p5MHz",      "cross_color",             (556, 202, 28,  28)),
    ("XC_WEDGE_4MHz",        "cross_color",             (556, 229, 28,  28)),
    ("XC_WEDGE_4p5MHz",      "cross_color",             (556, 256, 28,  28)),
    ("XC_WEDGE_5MHz",        "cross_color",             (556, 283, 28,  28)),
    ("XL_RED_INTERIOR",      "cross_luma",              (600, 445, 60,  30)),
    ("XL_MAGENTA_INTERIOR",  "cross_luma",              (130, 445, 40,  30)),
    ("ZP_CHROMA_LEAK",       "zone_plate_chroma_leak",  (180, 108, 360, 216)),
    # Radial wedge in cell (8,11). Designed to expose decoder cross-color
    # (the wedge is black/white only, so any chroma is decoder-induced)
    # and horizontal/vertical enhancement asymmetry (a decoder that
    # sharpens H more than V, or vice versa, shows different luma
    # modulation along H vs V cross-sections through the wedge centre).
    ("XC_RADIAL_WEDGE",      "cross_color",             (604, 384, 52, 44)),
    ("WEDGE_HV_SYMMETRY",    "wedge_hv_symmetry",       (604, 384, 52, 44)),
]

ARTIFACT_REGIONS = [
    {
        "id":            rid,
        "artifact_kind": kind,
        "ideal_box":     box,
        "sample":        {"kind": "center_window", "size_frac": 1.0},
    }
    for rid, kind, box in _ARTIFACT_RAW
]
