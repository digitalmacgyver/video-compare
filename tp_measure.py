#!/usr/bin/env python3
"""tp_measure: per-capture SW2 measurement.

Pipeline: probe -> extract single frame (no deinterlace) -> pad to 720x486 ->
register against ideal -> sample tartan + gray patches -> write JSON.

CLI:
    python tp_measure.py <capture> --frame N --output cap.json
"""

from __future__ import annotations
import json
import os
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
    if height > 486:
        raise ValueError(f"height {height} exceeds NTSC 486; cannot pad down")
    if height == 486:
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


def _apply_affine(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    a, b, tx = M[0]
    c, d, ty = M[1]
    return float(a * x + b * y + tx), float(c * x + d * y + ty)


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


import hashlib
import sys


def _ideal_for_md5():
    import tp_synthesize
    return tp_synthesize.synthesize(720, 486)


def _ideal_frame_md5() -> str:
    Y, U, V = _ideal_for_md5()
    h = hashlib.md5()
    h.update(Y.tobytes())
    h.update(U.tobytes())
    h.update(V.tobytes())
    return h.hexdigest()


def fit_gray_ramp(grays: list) -> Dict[str, Any]:
    """Fit measured Y10 = slope * ideal_Y10 + intercept across the 4 gray steps,
    plus report RMS residuals against two classical NTSC pedestal-mismatch
    hypotheses. Returns a JSON-serializable dict; or None if input is malformed.

    The pedestal hypotheses assume the chart is the canonical NTSC-M SW2
    generated with 7.5 IRE setup. The 4 gray boxes are then at
    {26, 44.5, 62.5, 81} IRE in the analog signal.

      Mode A: decoder ignores setup but signal has it
              -> Y_meas = 64 + v_IRE/100 * 876
      Mode B: decoder expects setup but signal has none (PAL/NTSC-J)
              -> Y_meas = 64 + (v_IRE_no_setup - 7.5)/92.5 * 876
                 with v_IRE_no_setup in {20, 40, 60, 80}
    """
    import math
    if len(grays) != 4:
        return None
    by_id = {g["id"]: g for g in grays}
    pairs = []
    for gid in ("G1", "G2", "G3", "G4"):
        if gid not in by_id:
            return None
        g = by_id[gid]
        pairs.append((g["ideal_y10"], g["measured_y10"]))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    if den <= 0:
        return None
    slope = num / den
    intercept = my - slope * mx
    gain_pct_loss = (1.0 - slope) * 100.0
    pred_black = slope * 64 + intercept
    pred_white = slope * 940 + intercept

    def _rms(predicted):
        return math.sqrt(sum((ys[i] - predicted[i]) ** 2 for i in range(n)) / n)

    pred_linear = [slope * xs[i] + intercept for i in range(n)]
    pred_mode_a = [64 + v / 100.0 * 876 for v in (26, 44.5, 62.5, 81)]
    pred_mode_b = [64 + (v - 7.5) / 92.5 * 876 for v in (20, 40, 60, 80)]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "gain_pct_loss": float(gain_pct_loss),
        "predicted_y10_at_black": float(pred_black),
        "predicted_y10_at_white": float(pred_white),
        "rms_linear": float(_rms(pred_linear)),
        "rms_pedestal_a": float(_rms(pred_mode_a)),
        "rms_pedestal_b": float(_rms(pred_mode_b)),
    }


def _ser_pt(p):
    if p is None:
        return None
    return [float(p[0]), float(p[1])]


def _build_geometry_block(reg_full):
    """Convert the in-memory geometry from register_with_geometry into the
    JSON shape documented in the Stage 2 design spec."""
    geom = reg_full.get("geometry")
    if geom is None:
        return None
    initial = reg_full["initial"]
    final = reg_full["final"]

    tris_out = {}
    for tid, t in geom["fiducials"]["triangles"].items():
        if t is None:
            tris_out[tid] = None
            continue
        tris_out[tid] = {
            "back_corner_1": _ser_pt(t["back_corner_1"]),
            "back_corner_2": _ser_pt(t["back_corner_2"]),
            "back_midpoint": _ser_pt(t["back_midpoint"]),
            "apex_inferred": _ser_pt(t["apex_inferred"]),
            "apex_detected": _ser_pt(t["apex_detected"]),
            "confidence":    float(t["confidence"]),
            "failure_reason": None,
        }
    cross_out = None
    rc = geom["fiducials"]["cross"]
    if rc is not None:
        cross_out = {
            "center": [float(rc["x"]), float(rc["y"])],
            "h_arm_len_px": float(rc["h_arm_len_px"]),
            "v_arm_len_px": float(rc["v_arm_len_px"]),
            "confidence":   float(rc["confidence"]),
        }
    circle_out = None
    bc = geom["fiducials"]["circle"]
    if bc is not None:
        circle_out = {
            "center":      [float(bc["cx"]), float(bc["cy"])],
            "rx":          float(bc["rx"]),
            "ry":          float(bc["ry"]),
            "rx_horizontal_px": float(bc.get("rx_horizontal_px", 0.0)),
            "ry_vertical_px":   float(bc.get("ry_vertical_px", 0.0)),
            "rotation_deg": float(bc["rotation_deg"]),
            "fit_rms":     float(bc["fit_rms"]),
            "confidence":  float(bc["confidence"]),
        }

    derived = geom["derived"]
    derived_out = {}
    for key in ("active_picture_box", "picture_extent_px",
                "picture_offset_from_ideal", "corner_skew_px",
                "cross_offset_from_ideal",
                "aperture_symmetry", "aspect_ratio_check",
                "diameter_vs_picture_height", "circle_fit_rms",
                "summary"):
        derived_out[key] = derived.get(key)
    if derived.get("arrow_tip_coords"):
        derived_out["arrow_tip_coords"] = {
            k: _ser_pt(v) for k, v in derived["arrow_tip_coords"].items()
        }
    else:
        derived_out["arrow_tip_coords"] = None
    derived_out["clip_detected"] = derived.get("clip_detected")

    quality_flag, quality_reason = _compute_geometry_quality(geom, final)

    return {
        "fiducials": {"triangles": tris_out, "cross": cross_out, "circle": circle_out},
        "derived":   derived_out,
        "registration_refit": {
            "inlier_count_initial": int(initial.get("inliers", 0)),
            "inlier_count_final":   int(final.get("inliers", 0)),
            "final_residuals_px":   final["residuals_px"],
            "anchors_added":        reg_full.get("anchors_added", []),
        },
        "quality_flag":   quality_flag,
        "quality_reason": quality_reason,
    }


def _compute_geometry_quality(geom, final):
    """Map Stage 2 detection counts + refit residuals to a quality flag.

    The black circle's ring is elliptical on real NTSC captures (10:11 PAR),
    and our circular detector reports inflated fit_rms even when triangles
    and cross are perfectly registered. So the circle is informational only
    -- it does not gate the flag. Gating is by triangle/cross presence and
    the final-fit mean residual.
    """
    tris = geom["fiducials"]["triangles"]
    detected_count = sum(1 for t in tris.values() if t is not None)
    cross_ok = geom["fiducials"]["cross"] is not None
    mean_res = final["residuals_px"]["mean"]

    if detected_count < 2 and not cross_ok:
        return "failed", "cross missing and <2 triangles detected"
    if detected_count < 3 or not cross_ok:
        return "partial", f"only {detected_count}/4 triangles, cross={cross_ok}"
    if mean_res < 1.0:
        return "ok", None
    if mean_res < 2.0:
        return "warn", f"mean residual {mean_res:.2f} px above 1.0 px ok threshold"
    return "failed", f"mean residual {mean_res:.2f} px above 2.0 px warn threshold"


def _sample_box_yuv(Y, U, V, region, affine):
    """Sample a single staircase box (no expected/sat_pct logic — those
    are derived later from the linear-fit). Returns
    (mean_Y10, mean_U10, mean_V10, patch_size_px)."""
    import math
    x, y, w, h = region["box"]
    cx, cy = x + w / 2.0, y + h / 2.0
    cx_cap, cy_cap = _apply_affine(affine, cx, cy)
    half_w = max(2, w // 2)
    half_h = max(2, h // 2)
    x0 = max(0, int(round(cx_cap)) - half_w)
    x1 = min(Y.shape[1], int(round(cx_cap)) + half_w)
    y0 = max(0, int(round(cy_cap)) - half_h)
    y1 = min(Y.shape[0], int(round(cy_cap)) + half_h)
    yp = Y[y0:y1, x0:x1].astype(np.float64)
    cx0, cx1 = x0 // 2, max(x0 // 2 + 1, x1 // 2)
    up = U[y0:y1, cx0:cx1].astype(np.float64)
    vp = V[y0:y1, cx0:cx1].astype(np.float64)
    return (
        float(yp.mean()) if yp.size else float("nan"),
        float(up.mean()) if up.size else float("nan"),
        float(vp.mean()) if vp.size else float("nan"),
        [x1 - x0, y1 - y0],
    )


def measure_chroma_staircase(Y, U, V, affine) -> Dict[str, Any]:
    """Sample the three magenta staircase boxes (33/66/100% magenta)
    and compute the chroma non-linearity + differential phase metrics.

    Returns a dict shaped for tp_compare to consume:

        regions: per-box mean YUV, chroma magnitude/phase, ideal vs
                 measured, phase delta vs the lowest-level box.
        summary: linear-fit slope/intercept/R^2 on chroma magnitude
                 vs intended level, max step deviation in %, and
                 differential phase = max - min phase across boxes
                 (degrees).
    """
    import math
    regions_out = []
    levels, mags, phases = [], [], []
    ys, us, vs = [], [], []
    for r in tp_chart.CHROMA_STAIRCASE_REGIONS:
        y10, u10, v10, patch_px = _sample_box_yuv(Y, U, V, r, affine)
        du = u10 - tp_chart.CHROMA_CENTER
        dv = v10 - tp_chart.CHROMA_CENTER
        chroma_mag = float((du * du + dv * dv) ** 0.5)
        phase_deg = float(math.degrees(math.atan2(dv, du)))
        ideal_y10, ideal_u10, ideal_v10 = r["ideal_yuv10"]
        ideal_du = ideal_u10 - tp_chart.CHROMA_CENTER
        ideal_dv = ideal_v10 - tp_chart.CHROMA_CENTER
        ideal_mag = float((ideal_du * ideal_du + ideal_dv * ideal_dv) ** 0.5)
        ideal_phase = float(math.degrees(math.atan2(ideal_dv, ideal_du)))
        regions_out.append({
            "id":            r["id"],
            "level":         r["level"],
            "ideal_yuv10":   [ideal_y10, ideal_u10, ideal_v10],
            "measured_yuv10":[y10, u10, v10],
            "chroma_magnitude":       chroma_mag,
            "ideal_chroma_magnitude": ideal_mag,
            "chroma_phase_deg":       phase_deg,
            "ideal_phase_deg":        ideal_phase,
            "patch_size_px":          patch_px,
        })
        levels.append(r["level"])
        mags.append(chroma_mag)
        phases.append(phase_deg)
        ys.append(y10); us.append(u10); vs.append(v10)

    # Linear fit: chroma_magnitude = slope * level + intercept.
    levels_arr = np.asarray(levels, dtype=np.float64)
    mags_arr   = np.asarray(mags,   dtype=np.float64)
    n = len(levels)
    if n >= 2:
        slope, intercept = np.polyfit(levels_arr, mags_arr, 1)
        pred = slope * levels_arr + intercept
        ss_res = float(((mags_arr - pred) ** 2).sum())
        ss_tot = float(((mags_arr - mags_arr.mean()) ** 2).sum())
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        # Per-step deviation as % of full-scale chroma magnitude.
        full_scale = float(slope)  # 100% box predicted chroma = slope + intercept ≈ slope
        deviations = [(float(m - p) / full_scale * 100.0
                       if full_scale > 1e-6 else float("nan"))
                      for m, p in zip(mags_arr, pred)]
        max_dev_pct = float(max(abs(d) for d in deviations
                                if not (d != d)))  # skip NaN
    else:
        slope = intercept = r_squared = float("nan")
        deviations = []
        max_dev_pct = float("nan")

    diff_phase = float(max(phases) - min(phases)) if phases else float("nan")
    # Also surface the per-step phase delta vs the first box (33%).
    phase_step_deg = [float(p - phases[0]) for p in phases] if phases else []

    # Linear fit on luma — does the staircase preserve linear luma
    # ramping? (Y - BLACK_Y10) should scale with level too.
    ys_arr = np.asarray(ys, dtype=np.float64)
    lum_above_black = ys_arr - tp_chart.BLACK_Y10
    if n >= 2:
        y_slope, y_int = np.polyfit(levels_arr, lum_above_black, 1)
        y_pred = y_slope * levels_arr + y_int
        ss_res_y = float(((lum_above_black - y_pred) ** 2).sum())
        ss_tot_y = float(((lum_above_black - lum_above_black.mean()) ** 2).sum())
        y_r2 = float(1.0 - ss_res_y / ss_tot_y) if ss_tot_y > 0 else float("nan")
    else:
        y_r2 = float("nan")

    return {
        "regions": regions_out,
        "summary": {
            "chroma_fit_slope":     float(slope),
            "chroma_fit_intercept": float(intercept),
            "chroma_linearity_r2":  r_squared,
            "max_step_deviation_pct": max_dev_pct,
            "step_deviations_pct":  [float(d) for d in deviations],
            "differential_phase_deg": diff_phase,
            "phase_steps_deg":      phase_step_deg,
            "luma_linearity_r2":    y_r2,
        },
    }


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

    reg_full = tp_register.register_with_geometry(Y_p)
    reg = reg_full["final"]
    meta["registration"] = {
        "affine": reg["affine_matrix"].tolist() if reg["affine_matrix"] is not None else None,
        "residuals_px": reg["residuals_px"],
        "inliers": reg["inliers"],
        "total": reg["total"],
        "landmarks_used": reg_full["initial"].get("landmarks_used", []),
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

    luma_scale = fit_gray_ramp(grays)

    geometry_block = _build_geometry_block(reg_full)
    stage3 = _measure_stage3(Y_p, U_p, V_p, M, reg["quality_flag"])
    chroma_staircase = measure_chroma_staircase(Y_p, U_p, V_p, M)

    return {
        "_meta": meta,
        "tartan": tartan,
        "grays": grays,
        "luma_scale": luma_scale,
        "geometry": geometry_block,
        "frequency_response": stage3["frequency_response"],
        "artifacts":          stage3["artifacts"],
        "decoder_class":      stage3["decoder_class"],
        "chroma_staircase":   chroma_staircase,
    }


def _measure_stage3(Y, U, V, affine, registration_quality_flag):
    """Run the Stage 3 measurement layer (frequency response + artifacts +
    decoder-class classification). Returns a dict with three keys; each is
    None on registration failure or per-stage exception."""
    out = {"frequency_response": None, "artifacts": None, "decoder_class": None}
    if affine is None or registration_quality_flag == "failed":
        return out
    try:
        import tp_freq
        out["frequency_response"] = tp_freq.measure(Y, U, V, affine)
    except Exception as e:
        out["frequency_response"] = {"error": str(e)}
    try:
        import tp_artifacts
        out["artifacts"] = tp_artifacts.measure(Y, U, V, affine)
    except Exception as e:
        out["artifacts"] = {"error": str(e)}
    try:
        import tp_classify
        if (out["frequency_response"] and "regions" in out["frequency_response"]
                and out["artifacts"] and "regions" in out["artifacts"]):
            out["decoder_class"] = tp_classify.classify(
                out["frequency_response"], out["artifacts"]
            )
    except Exception as e:
        out["decoder_class"] = {"error": str(e)}
    return out


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="path to capture (mov/avi/mkv/...)")
    p.add_argument("--frame", type=int, default=60, help="frame index (default 60)")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--no-overlay", action="store_true",
                   help="skip writing the sibling _overlay.png")
    args = p.parse_args()
    data = measure(args.capture, args.frame)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(
        f"wrote {args.output}: "
        f"registration={data['_meta']['registration']['quality_flag']}, "
        f"residuals_mean={data['_meta']['registration']['residuals_px']['mean']:.2f}px"
    )
    if not args.no_overlay:
        stem = os.path.splitext(args.output)[0]
        try:
            import tp_sample_overlay
            import cv2
            bgr = tp_sample_overlay.annotate(args.capture, args.output, args.frame)
            overlay_path = stem + "_overlay.png"
            cv2.imwrite(overlay_path, bgr)
            print(f"wrote {overlay_path} ({bgr.shape[1]}x{bgr.shape[0]})")
        except Exception as e:
            print(f"WARNING: overlay PNG generation failed: {e}", file=sys.stderr)
        try:
            import tp_fiducial_crops
            import cv2
            fids_bgr = tp_fiducial_crops.build(args.capture, args.output, args.frame)
            fids_path = stem + "_fiducials.png"
            cv2.imwrite(fids_path, fids_bgr)
            print(f"wrote {fids_path} ({fids_bgr.shape[1]}x{fids_bgr.shape[0]})")
        except Exception as e:
            print(f"WARNING: fiducial-crops PNG generation failed: {e}", file=sys.stderr)
        try:
            import tp_wedge_crops
            import cv2
            wedge_bgr = tp_wedge_crops.build(args.capture, args.output, args.frame)
            wedge_path = stem + "_wedge.png"
            cv2.imwrite(wedge_path, wedge_bgr)
            print(f"wrote {wedge_path} ({wedge_bgr.shape[1]}x{wedge_bgr.shape[0]})")
        except Exception as e:
            print(f"WARNING: wedge-crops PNG generation failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    _main()
