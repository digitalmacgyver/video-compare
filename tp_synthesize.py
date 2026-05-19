#!/usr/bin/env python3
"""Synthesize the ideal SW2 NTSC test pattern frame at 720x486.

Stage 1 scope: grey background, black main grid, top-left tartan (4x2),
4-step gray strip below it.
Stage 2 adds: all 4 boundary triangles, the registration cross, and the
black circle.

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
    assert x % 2 == 0 and w % 2 == 0, (
        f"_fill_box_yuv422 requires even x and w for exact 4:2:2 alignment; "
        f"got x={x}, w={w}"
    )
    Y[y:y + h, x:x + w] = int(round(y10))
    cx0, cx1 = x // 2, (x + w) // 2
    U[y:y + h, cx0:cx1] = int(round(u10))
    V[y:y + h, cx0:cx1] = int(round(v10))


def _draw_tartan(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    """Render the 4x2 tartan region by filling each box with its ideal YUV10 codes."""
    for r in tp_chart.TARTAN_REGIONS:
        e = r["expected"]
        _fill_box_yuv422(Y, U, V, r["ideal_box"], e["y10"], e["u10"], e["v10"])


def _draw_gray_strip(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    """Render the 4-step gray strip by filling each box with its ideal Y10 code."""
    for r in tp_chart.GRAY_REGIONS:
        e = r["expected"]
        _fill_box_yuv422(Y, U, V, r["ideal_box"], e["y10"], e["u10"], e["v10"])


def _fill_triangle_y(Y: np.ndarray, p1, p2, p3, y10: int = None) -> None:
    """Fill a triangle on the Y plane only (chroma not affected — boundary
    triangles are black on the chroma-center-grey background).

    Uses cv2.fillPoly which handles the geometry. p1/p2/p3 are (x, y)
    integer coords."""
    import cv2
    if y10 is None:
        y10 = tp_chart.BLACK_Y10
    pts = np.array([[p1, p2, p3]], dtype=np.int32)
    cv2.fillPoly(Y, pts, int(y10))


def _draw_saturated_chroma_blocks(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    """Render the row-9 saturated chroma blocks so Stage 3 artifact
    metrics have real chroma transitions to sample around.

    - Cells (9,10)-(9,12): 100% red block.
    - Cells (9,1)-(9,3): magenta steps at 33/66/100% saturation.
    """
    # Red 100% block: x=540, y=432, w=180, h=54.
    y10, u10, v10 = tp_chart.rgb_norm_to_yuv10(1.0, 0.0, 0.0)
    _fill_box_yuv422(Y, U, V, (540, 432, 180, 54), y10, u10, v10)
    # Magenta steps.
    for i, frac in enumerate([1.0 / 3.0, 2.0 / 3.0, 1.0]):
        y10, u10, v10 = tp_chart.rgb_norm_to_yuv10(frac, 0.0, frac)
        _fill_box_yuv422(Y, U, V, (i * 60, 432, 60, 54), y10, u10, v10)


_YC_COLOR_PAIRS = {
    "red_cyan":    ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
    "blue_yellow": ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
}


def _draw_chroma_bursts(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> None:
    """Render the row-9 Y/C timing bursts: alternating colored stripes at
    the labeled chroma frequency. Both Y and chroma alternate together
    (the chart uses high-contrast color pairs like red/cyan and
    blue/yellow), so a Y/C-timing-aligned decoder reproduces the stripes
    cleanly while a misaligned one shows chroma offset from the luma."""
    for r in tp_chart.BURST_REGIONS:
        if r["kind"] != "chroma_burst":
            continue
        x, y, w, h = r["ideal_box"]
        period_luma = tp_chart.NTSC_SAMPLE_RATE_MHZ / r["frequency_MHz"]
        pair = _YC_COLOR_PAIRS[r["color_pair"]]
        yA, uA, vA = tp_chart.rgb_norm_to_yuv10(*pair[0])
        yB, uB, vB = tp_chart.rgb_norm_to_yuv10(*pair[1])
        xs_luma = np.arange(w, dtype=np.float32)
        phase_luma = 2.0 * np.pi * xs_luma / period_luma
        sel_luma = np.sin(phase_luma) >= 0
        stripe_y = np.where(sel_luma, yA, yB).astype(np.uint16)
        Y[y:y + h, x:x + w] = stripe_y[None, :]
        # U/V are half-x sampled; each chroma sample covers 2 luma px.
        xs_chroma = np.arange(w // 2, dtype=np.float32)
        phase_chroma = 2.0 * np.pi * (xs_chroma * 2) / period_luma
        sel_chroma = np.sin(phase_chroma) >= 0
        stripe_u = np.where(sel_chroma, uA, uB).astype(np.uint16)
        stripe_v = np.where(sel_chroma, vA, vB).astype(np.uint16)
        U[y:y + h, x // 2:(x + w) // 2] = stripe_u[None, :]
        V[y:y + h, x // 2:(x + w) // 2] = stripe_v[None, :]


def _sin_to_y10(phase: np.ndarray) -> np.ndarray:
    """Map a sinusoid in [-1, 1] to luma codes spanning [BLACK_Y10,
    WHITE_Y10]. Produces an antialiased, visually-symmetric stripe
    pattern with a single FFT peak at the fundamental — the right
    "perfect" reference for a burst test."""
    mid = (tp_chart.WHITE_Y10 + tp_chart.BLACK_Y10) / 2.0
    amp = (tp_chart.WHITE_Y10 - tp_chart.BLACK_Y10) / 2.0
    return (mid + amp * np.sin(phase)).clip(0, 1023).astype(np.uint16)


def _draw_vertical_bursts(Y: np.ndarray) -> None:
    """Render the three vertical-frequency bursts in cells (4..6, 1) —
    slightly-tilted near-horizontal stripes at the chart-spec vertical
    frequency. These probe vertical-axis resolution (scan converters,
    deinterlacers, vertical-aperture enhancers). Rendered as a smooth
    sinusoid so the reference image is visually symmetric."""
    for r in tp_chart.VERTICAL_BURST_REGIONS:
        x, y, w, h = r["ideal_box"]
        freq_cpr = r["freq_cycles_per_row"]
        angle_deg = r.get("stripe_angle_deg", 0.0)
        theta = np.deg2rad(angle_deg)
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        proj = xx * np.sin(theta) + yy * np.cos(theta)
        phase = 2.0 * np.pi * freq_cpr * proj
        Y[y:y + h, x:x + w] = _sin_to_y10(phase)


def _draw_radial_wedge(Y: np.ndarray) -> None:
    """Render the radial wedge (Siemens-star-style resolution probe) in
    cell (8,11). N smooth wedges between inner_radius and outer_radius
    around the chart-spec center, rendered at 4× supersampling and
    box-filtered down so the reference matches the visual appearance
    of the real chart: wedges visible at outer radii, fading toward
    grey near the center where they exceed pixel resolution."""
    rw = tp_chart.RADIAL_WEDGE
    cx, cy = rw["center_xy"]
    r_in  = float(rw["inner_radius_px"])
    r_out = float(rw["outer_radius_px"])
    n_pairs = int(rw["n_wedge_pairs"])
    cell_x, cell_y, cell_w, cell_h = rw["cell_box"]
    ss = 4  # supersampling factor (4× linear, 16 sub-samples per pixel)
    yi = np.arange(cell_h * ss, dtype=np.float32) / ss + cell_y
    xi = np.arange(cell_w * ss, dtype=np.float32) / ss + cell_x
    xx, yy = np.meshgrid(xi, yi)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    # Smooth wedge profile: sinusoid in angle. Modulation amplitude
    # tapers smoothly toward the inner / outer radii so the wedge
    # blends into the grey background instead of having a hard edge.
    mid = (tp_chart.WHITE_Y10 + tp_chart.BLACK_Y10) / 2.0
    amp = (tp_chart.WHITE_Y10 - tp_chart.BLACK_Y10) / 2.0
    in_band = (r >= r_in) & (r <= r_out)
    # Quarter-cosine taper across a 1-px edge band so the wedge
    # doesn't have a hard circular boundary.
    edge = 1.0
    inner_taper = np.clip((r - r_in) / edge, 0.0, 1.0)
    outer_taper = np.clip((r_out - r) / edge, 0.0, 1.0)
    taper = np.minimum(inner_taper, outer_taper)
    wedge_signal = mid + amp * taper * np.sin(theta * n_pairs)
    # Outside the band, fall back to the existing pixel content
    # (grey background from the chart).
    sub = Y[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]
    sub_ss = np.repeat(np.repeat(sub.astype(np.float32), ss, axis=0),
                       ss, axis=1)
    rendered_ss = np.where(in_band, wedge_signal, sub_ss)
    # Box-filter downsample (mean over each ss×ss block) for antialiasing.
    rendered = rendered_ss.reshape(cell_h, ss, cell_w, ss).mean(axis=(1, 3))
    Y[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w] = (
        rendered.clip(0, 1023).astype(np.uint16)
    )


def _draw_pulse_cells(Y: np.ndarray) -> None:
    """Render the 3 pulse-and-bar cells on the right edge of the chart.
    Each cell is filled with its background level and a sin²-shaped 2T
    pulse is overlaid at the chart-spec center."""
    for r in tp_chart.PULSE_REGIONS:
        x, y, w, h = r["cell_box"]
        bg = int(round(r["background_y10"]))
        peak = float(r["pulse_peak_y10"])
        cx, _ = r["pulse_center_xy"]
        fwhm = float(r["pulse_fwhm_px"])
        Y[y:y + h, x:x + w] = bg
        # sin² pulse: amplitude(dx) = cos²(π/2 * dx/fwhm) for |dx| ≤ fwhm.
        # Past ±fwhm the pulse is clamped to zero so the cell is purely
        # background outside the pulse footprint.
        radius = int(np.ceil(fwhm)) + 1
        xs = np.arange(cx - radius, cx + radius + 1, dtype=np.float32)
        deltas = xs - cx
        env = np.where(
            np.abs(deltas) <= fwhm,
            np.cos(np.pi / 2.0 * deltas / fwhm) ** 2,
            0.0,
        )
        delta_amp = peak - r["background_y10"]
        pulse_y10 = bg + (delta_amp * env)
        pulse_y10 = np.clip(pulse_y10, 0, 1023).astype(np.uint16)
        x0 = int(xs[0]); x1 = int(xs[-1]) + 1
        # Clip to the cell horizontal bounds so a pulse near the cell
        # edge doesn't spill into neighboring cells.
        cell_x0 = x
        cell_x1 = x + w
        seg_x0 = max(x0, cell_x0)
        seg_x1 = min(x1, cell_x1)
        if seg_x1 <= seg_x0:
            continue
        seg_off = seg_x0 - x0
        seg_len = seg_x1 - seg_x0
        Y[y:y + h, seg_x0:seg_x1] = pulse_y10[seg_off:seg_off + seg_len]


def _draw_wedge_column(Y: np.ndarray) -> None:
    """Render the continuous frequency wedge in column 10. Local
    frequency rises linearly from tp_chart.WEDGE_COLUMN['freq_top'] to
    'freq_bottom' over the y range, while stripes remain vertical (so
    horizontal frequency is what's modulated)."""
    w = tp_chart.WEDGE_COLUMN
    x0 = w["x"]; y0 = w["y_top"]; y1 = w["y_bottom"]; box_w = w["width"]
    sample_rate = tp_chart.NTSC_SAMPLE_RATE_MHZ
    for row_y in range(y0, y1):
        freq = tp_chart.wedge_column_freq_at_y(row_y)
        period = sample_rate / freq
        xs = np.arange(box_w, dtype=np.float32)
        phase = 2.0 * np.pi * xs / period
        Y[row_y, x0:x0 + box_w] = _sin_to_y10(phase)


def _draw_bursts(Y: np.ndarray) -> None:
    """Render BURST_REGIONS as black/white stripes at the chart frequency.

    Horizontal sample rate is tp_chart.NTSC_SAMPLE_RATE_MHZ (13.5 MHz);
    stripe period_px = sample_rate / freq_MHz. For burst_vertical and
    wedge_segment, stripes are vertical (frequency along x). For
    burst_diagonal, stripes are rotated by stripe_angle_deg.

    wedge_segment regions are skipped here — they are drawn as a single
    continuous wedge by _draw_wedge_column so the synth matches the real
    chart, which has a continuous narrowing pattern rather than discrete
    fixed-frequency tiles.
    """
    for r in tp_chart.BURST_REGIONS:
        x, y, w, h = r["ideal_box"]
        period = tp_chart.NTSC_SAMPLE_RATE_MHZ / r["frequency_MHz"]
        kind = r["kind"]
        if kind == "wedge_segment":
            continue
        if kind == "burst_vertical":
            xs = np.arange(w, dtype=np.float32)
            phase = 2.0 * np.pi * xs / period
            row = _sin_to_y10(phase)
            Y[y:y + h, x:x + w] = row[None, :]
        elif kind == "burst_diagonal":
            theta = np.deg2rad(r["stripe_angle_deg"])
            xs = np.arange(w, dtype=np.float32)
            ys = np.arange(h, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            proj = xx * np.cos(theta) + yy * np.sin(theta)
            phase = 2.0 * np.pi * proj / period
            Y[y:y + h, x:x + w] = _sin_to_y10(phase)


def _draw_boundary_triangles(Y: np.ndarray) -> None:
    """Render all 4 boundary triangles from tp_chart.BOUNDARY_TRIANGLES."""
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        bc1 = tuple(int(v) for v in tri["ideal_back_corner_1"])
        bc2 = tuple(int(v) for v in tri["ideal_back_corner_2"])
        apex = tuple(int(v) for v in tri["ideal_apex"])
        # cv2.fillPoly handles out-of-frame apex coords (clips them).
        _fill_triangle_y(Y, bc1, bc2, apex)


def _draw_black_circle(Y: np.ndarray) -> None:
    """Render the black circle ring as a PAR-elliptical shape (matches
    real NTSC captures, where a logically-round circle appears in the
    720x486 raster with horizontal semi-axis = vertical semi-axis ×
    11/10)."""
    import cv2
    bc = tp_chart.BLACK_CIRCLE
    ry = int(bc["expected_radius_px"])
    rx = int(round(ry * tp_chart.NTSC_PAR_X_OVER_Y))
    cv2.ellipse(
        Y,
        center=(bc["ideal_cx"], bc["ideal_cy"]),
        axes=(rx, ry),
        angle=0.0,
        startAngle=0.0,
        endAngle=360.0,
        color=int(tp_chart.BLACK_Y10),
        thickness=bc["ring_thickness_px"],
        lineType=cv2.LINE_AA,
    )


def _draw_registration_cross(Y: np.ndarray) -> None:
    """Render the registration cross from tp_chart.REGISTRATION_CROSS.

    A black square (box_size_px wide) with a white plus inside it
    (ideal_arm_len_px tip-to-tip, ideal_arm_thickness_px wide).
    """
    rc = tp_chart.REGISTRATION_CROSS
    cx, cy = rc["ideal_x"], rc["ideal_y"]
    box = rc["box_size_px"]
    arm_len = rc["ideal_arm_len_px"]
    arm_th  = rc["ideal_arm_thickness_px"]
    half_box = box // 2
    half_len = arm_len // 2
    half_th  = arm_th // 2
    # Black box
    Y[cy - half_box:cy + half_box, cx - half_box:cx + half_box] = tp_chart.BLACK_Y10
    # White cross arms
    Y[cy - half_th:cy + half_th + 1, cx - half_len:cx + half_len + 1] = tp_chart.WHITE_Y10
    Y[cy - half_len:cy + half_len + 1, cx - half_th:cx + half_th + 1] = tp_chart.WHITE_Y10


def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ideal SW2 frame as (Y, U, V) uint16 planes (yuv422p10le)."""
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    _draw_tartan(Y, U, V)
    _draw_gray_strip(Y, U, V)
    _draw_wedge_column(Y)
    _draw_pulse_cells(Y)
    _draw_radial_wedge(Y)
    _draw_vertical_bursts(Y)
    _draw_bursts(Y)
    _draw_saturated_chroma_blocks(Y, U, V)
    _draw_chroma_bursts(Y, U, V)
    _draw_boundary_triangles(Y)
    _draw_black_circle(Y)
    _draw_registration_cross(Y)
    return Y, U, V


# =====================================================================
# CLI
# =====================================================================

def _yuv422p10_to_bgr8(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Convert yuv422p10le planes to 8-bit BGR for image writing."""
    h, w = Y.shape
    U_full = np.repeat(U, 2, axis=1)[:, :w]
    V_full = np.repeat(V, 2, axis=1)[:, :w]
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
