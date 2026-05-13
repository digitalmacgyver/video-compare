"""Decoder-artifact metrics (Stage 3).

For each region in tp_chart.ARTIFACT_REGIONS:
  - hanging_dots:           peak-to-peak Y modulation in a thin strip at
                            a chroma transition (cross-luma at SC).
  - dot_crawl:               chroma RMS in a grey region adjacent to a
                            chroma transition (cross-color spread).
  - cross_color:             chroma RMS inside a luma-only burst.
  - cross_luma:              SC-band Y modulation inside a flat chroma block.
  - zone_plate_chroma_leak:  chroma RMS over the zone-plate container box
                            and a binary chroma_present flag.

The summary aggregates max values per artifact-kind for the classifier.
"""
from __future__ import annotations
import numpy as np

import tp_chart


THRESHOLDS = {
    "T_zp_threshold":       15.0,   # zone-plate chroma_rms threshold
    "T_hd_high":            25.0,   # hanging dots peak-to-peak Y10
    "T_hd_low":             12.0,
    "T_xc_high":            20.0,   # cross-color chroma_rms threshold
    "T_xc_low":             12.0,
    "T_xl_low":             15.0,   # cross-luma peak-to-peak Y10
    "T_chroma_bw_low_pct":  30.0,   # modulation pct at WEDGE_4/5MHz
}


def _apply_affine_pt(M, x, y):
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _project_box(M, box):
    x, y, w, h = box
    cx, cy = _apply_affine_pt(M, x + w / 2.0, y + h / 2.0)
    return (int(round(cx - w / 2.0)),
            int(round(cy - h / 2.0)),
            int(w), int(h))


def _crop_yuv(Y, U, V, box):
    """Crop Y at integer coords and U/V at half-x. Returns (y, u, v) windows
    or (None, None, None) if the box is fully out of frame."""
    x, y, w, h = box
    h_lim, w_lim = Y.shape
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(w_lim, x + w); y1 = min(h_lim, y + h)
    if x1 <= x0 or y1 <= y0:
        return None, None, None
    yw = Y[y0:y1, x0:x1].astype(np.float32)
    ux0 = x0 // 2; ux1 = (x1 + 1) // 2
    uw = U[y0:y1, ux0:ux1].astype(np.float32)
    vw = V[y0:y1, ux0:ux1].astype(np.float32)
    return yw, uw, vw


def _chroma_rms(u_win, v_win):
    if u_win is None or v_win is None or u_win.size == 0:
        return 0.0
    du = u_win - tp_chart.CHROMA_CENTER
    dv = v_win - tp_chart.CHROMA_CENTER
    return float(np.sqrt((du * du + dv * dv).mean()))


def _hanging_dots_pp(y_win):
    """Peak-to-peak Y modulation in the strip after removing row means."""
    if y_win is None or y_win.size == 0:
        return 0.0
    demean = y_win - y_win.mean(axis=1, keepdims=True)
    return float(demean.max() - demean.min())


def _cross_luma_sc_amp(y_win, sample_rate_MHz):
    """Bandpass Y around 3.0..4.0 MHz; return peak-to-peak of the bandpassed
    middle row. Captures cross-luma at the NTSC color subcarrier."""
    if y_win is None or y_win.size == 0:
        return 0.0
    line = y_win[y_win.shape[0] // 2, :].astype(np.float64)
    line = line - line.mean()
    n = len(line)
    if n < 8:
        return 0.0
    spec = np.fft.rfft(line)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_MHz)
    band = np.zeros_like(spec)
    mask = (freqs >= 3.0) & (freqs <= 4.0)
    band[mask] = spec[mask]
    filt = np.fft.irfft(band, n)
    return float(filt.max() - filt.min())


def _wedge_hv_symmetry(y_win):
    """Compare horizontal vs vertical luma modulation through the wedge
    center. The radial wedge has black/white wedges radiating from its
    centre; on a symmetric (aperture-balanced) decoder, the H and V
    cross-sections through the centre carry equal modulation energy. A
    decoder that applies more sharpening on one axis biases the ratio."""
    if y_win is None or y_win.size < 4:
        return None
    cy = y_win.shape[0] // 2
    cx = y_win.shape[1] // 2
    h_line = y_win[cy, :].astype(np.float32)
    v_line = y_win[:, cx].astype(np.float32)
    h_line = h_line - h_line.mean()
    v_line = v_line - v_line.mean()
    h_std = float(h_line.std())
    v_std = float(v_line.std())
    if v_std > 1e-6:
        hv_ratio = float(h_std / v_std)
    else:
        hv_ratio = None
    return {
        "h_modulation_std": h_std,
        "v_modulation_std": v_std,
        "hv_ratio":         hv_ratio,
    }


def measure(Y, U, V, affine):
    sample_rate = tp_chart.NTSC_SAMPLE_RATE_MHZ
    regions_out = {}
    for r in tp_chart.ARTIFACT_REGIONS:
        kind = r["artifact_kind"]
        box = _project_box(affine, r["ideal_box"])
        yw, uw, vw = _crop_yuv(Y, U, V, box)
        entry = {"artifact_kind": kind, "sample_box_capture": list(box)}
        if yw is None:
            entry["error"] = "out_of_frame"
        elif kind == "hanging_dots":
            pp = _hanging_dots_pp(yw)
            entry["metric_y10_pp"] = pp
            entry["metric_y10_pp_normalized"] = pp / 876.0
        elif kind == "dot_crawl":
            entry["chroma_rms"] = _chroma_rms(uw, vw)
        elif kind == "cross_color":
            entry["chroma_rms"] = _chroma_rms(uw, vw)
        elif kind == "cross_luma":
            entry["metric_y10_pp"] = _cross_luma_sc_amp(yw, sample_rate)
        elif kind == "zone_plate_chroma_leak":
            crms = _chroma_rms(uw, vw)
            entry["chroma_rms"] = crms
            entry["chroma_present"] = bool(crms > THRESHOLDS["T_zp_threshold"])
        elif kind == "wedge_hv_symmetry":
            sym = _wedge_hv_symmetry(yw)
            if sym is not None:
                entry.update(sym)
        regions_out[r["id"]] = entry
    return {
        "regions":    regions_out,
        "summary":    _summarize(regions_out),
        "thresholds": dict(THRESHOLDS),
    }


def _summarize(regions_out):
    def _max_metric(kind, key):
        vals = [r.get(key, 0.0) for r in regions_out.values()
                if r.get("artifact_kind") == kind and key in r]
        return float(max(vals)) if vals else 0.0

    zp = regions_out.get("ZP_CHROMA_LEAK", {}) or {}
    return {
        "max_hanging_dots_y10_pp":    _max_metric("hanging_dots", "metric_y10_pp"),
        "max_dot_crawl_chroma_rms":   _max_metric("dot_crawl", "chroma_rms"),
        "max_cross_color_chroma_rms": _max_metric("cross_color", "chroma_rms"),
        "max_cross_luma_y10_pp":      _max_metric("cross_luma", "metric_y10_pp"),
        "zone_plate_chroma_rms":      float(zp.get("chroma_rms", 0.0)),
        "zone_plate_chroma_present":  bool(zp.get("chroma_present", False)),
    }
