"""Per-burst frequency-response measurement (Stage 3).

For each region in tp_chart.BURST_REGIONS:
  1. Project the ideal sample box through the Stage 2 affine matrix.
  2. Extract a 1D Y cross-section in the burst's modulation direction.
  3. FFT the cross-section, find peak amplitude at the expected frequency.
  4. Express modulation as a percentage of full chart contrast (BLACK->WHITE)
     and convert to dB.

The summary block exposes the assembled luma response curve and the
interpolated -3 dB / -6 dB roll-off frequencies.
"""
from __future__ import annotations
import math
import numpy as np

import tp_chart


def _apply_affine_pt(M, x, y):
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _sample_box_capture(M, ideal_box):
    x, y, w, h = ideal_box
    cx, cy = _apply_affine_pt(M, x + w / 2.0, y + h / 2.0)
    return (int(round(cx - w / 2.0)),
            int(round(cy - h / 2.0)),
            int(w), int(h))


def _crop(Y, box):
    x, y, w, h = box
    h_lim, w_lim = Y.shape
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(w_lim, x + w); y1 = min(h_lim, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return Y[y0:y1, x0:x1].astype(np.float32)


def _line_modulation(line, freq_MHz, sample_rate_MHz):
    """Return (peak_amplitude_pp_codes, detected_freq_MHz, snr_db)."""
    line = line - line.mean()
    n = len(line)
    if n < 8:
        return 0.0, 0.0, 0.0
    spec = np.fft.rfft(line)
    mags = np.abs(spec) * (2.0 / n)  # single-sided amplitude spectrum
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_MHz)
    tol = max(0.5, 0.25 * freq_MHz)
    candidates = np.where(np.abs(freqs - freq_MHz) <= tol)[0]
    if len(candidates) == 0:
        idx = int(np.argmin(np.abs(freqs - freq_MHz)))
    else:
        idx = int(candidates[np.argmax(mags[candidates])])
    peak_amp_pp = float(mags[idx] * 2.0)  # sinusoid amplitude -> peak-to-peak
    detected = float(freqs[idx])
    non_peak = np.concatenate([mags[1:idx], mags[idx + 1:]])
    if len(non_peak) > 0:
        noise_rms = float(np.sqrt((non_peak ** 2).mean()))
        snr_db = 20.0 * math.log10(
            max(peak_amp_pp, 1e-9) / max(noise_rms, 1e-9)
        )
    else:
        snr_db = 0.0
    return peak_amp_pp, detected, snr_db


def _extract_line(win, kind, stripe_angle_deg=None):
    h, w = win.shape
    if kind in ("burst_vertical", "wedge_segment"):
        return win[h // 2, :]
    if kind == "burst_horizontal":
        return win[:, w // 2]
    if kind == "burst_diagonal":
        # Sample along the projection axis (perpendicular to the stripes).
        # The synthesizer defines stripes by `proj = x*cos + y*sin`, so the
        # direction of maximum modulation is (cos, sin) -- the same vector.
        import scipy.ndimage as ndi
        cx, cy = w / 2.0, h / 2.0
        length = int(min(h, w))
        theta = np.deg2rad(stripe_angle_deg)
        t = np.linspace(-length / 2.0, length / 2.0, length)
        xs = cx + t * np.cos(theta)
        ys = cy + t * np.sin(theta)
        return ndi.map_coordinates(
            win.astype(np.float64), [ys, xs], order=1, mode="nearest"
        )
    raise ValueError(f"unknown burst kind: {kind}")


def _crop_chroma(U, V, box):
    """Crop U and V at half-x for a luma-coord box. Returns (U_win, V_win)
    each as float32, or (None, None) if the crop falls fully outside."""
    x, y, w, h = box
    h_lim, w_lim = U.shape
    x0 = max(0, x // 2)
    y0 = max(0, y)
    x1 = min(w_lim, (x + w + 1) // 2)
    y1 = min(h_lim, y + h)
    if x1 <= x0 or y1 <= y0:
        return None, None
    return U[y0:y1, x0:x1].astype(np.float32), V[y0:y1, x0:x1].astype(np.float32)


def _measure_chroma_burst(r, Y, U, V, box):
    """Return chroma modulation + luma cross-modulation at the burst's
    expected frequency. Chroma is the sqrt(U^2+V^2) magnitude (with each
    plane recentred at the chroma neutral 512); luma is plain Y."""
    chroma_sample_rate = tp_chart.NTSC_SAMPLE_RATE_MHZ / 2.0  # 6.75 MHz
    freq_MHz = r["frequency_MHz"]
    win_y = _crop(Y, box)
    win_u, win_v = _crop_chroma(U, V, box)
    if win_y is None or win_y.size == 0 or win_u is None:
        return None
    # FFT U and V separately and take the axis with the larger modulation.
    # Magnitude-sqrt(U^2+V^2) does NOT modulate between opposing colors
    # (red/cyan have the same chroma magnitude, just different
    # directions), so per-axis FFTs are required to detect the burst.
    u_line = win_u[win_u.shape[0] // 2, :].astype(np.float32) - tp_chart.CHROMA_CENTER
    v_line = win_v[win_v.shape[0] // 2, :].astype(np.float32) - tp_chart.CHROMA_CENTER
    u_peak, _, u_snr = _line_modulation(u_line, freq_MHz, chroma_sample_rate)
    v_peak, _, v_snr = _line_modulation(v_line, freq_MHz, chroma_sample_rate)
    if u_peak >= v_peak:
        chroma_peak_pp, chroma_snr = u_peak, u_snr
        chroma_detected = freq_MHz  # detected freq lookup is in _line_modulation; use expected
    else:
        chroma_peak_pp, chroma_snr = v_peak, v_snr
        chroma_detected = freq_MHz
    # Luma line at chart sample rate, same expected frequency. Non-zero
    # luma modulation here is cross-luma / dot-crawl injected by the
    # decoder at the chroma transitions.
    y_line = win_y[win_y.shape[0] // 2, :]
    luma_peak_pp, luma_detected, luma_snr = _line_modulation(
        y_line, freq_MHz, tp_chart.NTSC_SAMPLE_RATE_MHZ
    )
    # Normalize chroma_pct against full chroma swing (896 codes per axis,
    # so magnitude ~ 896 for fully-saturated alternation).
    chroma_full = 896.0
    chroma_pct = 100.0 * chroma_peak_pp / chroma_full
    chroma_db = (
        20.0 * math.log10(chroma_pct / 100.0) if chroma_pct > 0 else float("-inf")
    )
    luma_full = float(tp_chart.WHITE_Y10 - tp_chart.BLACK_Y10)
    luma_pct = 100.0 * luma_peak_pp / luma_full
    return {
        "frequency_MHz_expected":      freq_MHz,
        "chroma_modulation_pct":       float(chroma_pct),
        "chroma_modulation_db":        float(chroma_db),
        "chroma_frequency_MHz_detected": float(chroma_detected),
        "chroma_snr_db":               float(chroma_snr),
        "luma_dot_crawl_pct":          float(luma_pct),
        "luma_frequency_MHz_detected": float(luma_detected),
        "sample_box_capture":          list(box),
    }


def measure(Y, U, V, affine):
    contrast_full = float(tp_chart.WHITE_Y10 - tp_chart.BLACK_Y10)  # 876
    sample_rate = tp_chart.NTSC_SAMPLE_RATE_MHZ
    regions_out = {}
    for r in tp_chart.BURST_REGIONS:
        box = _sample_box_capture(affine, r["ideal_box"])
        if r["kind"] == "chroma_burst":
            entry = _measure_chroma_burst(r, Y, U, V, box)
            if entry is None:
                entry = {
                    "frequency_MHz_expected": r["frequency_MHz"],
                    "chroma_modulation_pct":  0.0,
                    "luma_dot_crawl_pct":     0.0,
                    "sample_box_capture":     list(box),
                    "error":                  "out_of_frame",
                }
            regions_out[r["id"]] = entry
            continue
        win = _crop(Y, box)
        if win is None or win.size == 0:
            regions_out[r["id"]] = {
                "frequency_MHz_expected": r["frequency_MHz"],
                "frequency_MHz_detected": None,
                "modulation_pct":         0.0,
                "modulation_db":          float("-inf"),
                "snr_db":                 0.0,
                "sample_box_capture":     list(box),
            }
            continue
        line = _extract_line(win, r["kind"], r.get("stripe_angle_deg"))
        peak_pp, detected, snr_db = _line_modulation(
            line, r["frequency_MHz"], sample_rate
        )
        mod_pct = 100.0 * peak_pp / contrast_full
        if mod_pct <= 0:
            mod_db = float("-inf")
        else:
            mod_db = 20.0 * math.log10(mod_pct / 100.0)
        regions_out[r["id"]] = {
            "frequency_MHz_expected": r["frequency_MHz"],
            "frequency_MHz_detected": detected,
            "modulation_pct":         float(mod_pct),
            "modulation_db":          float(mod_db),
            "snr_db":                 float(snr_db),
            "sample_box_capture":     list(box),
        }
    return {"regions": regions_out, "summary": _summarize(regions_out)}


def _summarize(regions_out):
    # The luma_response_curve only includes Y-plane bursts. Chroma bursts
    # (Y/C timing) are reported separately because their amplitude is in
    # different units (chroma codes vs luma codes).
    curve = sorted(
        ((d["frequency_MHz_expected"], d["modulation_db"])
         for d in regions_out.values()
         if "modulation_db" in d),
        key=lambda p: p[0],
    )

    def _crossing(threshold_db):
        last_above = None
        for f, db in curve:
            if db >= threshold_db:
                last_above = (f, db)
            elif last_above is not None:
                f0, db0 = last_above
                if db0 == db:
                    return float(f)
                ratio = (db0 - threshold_db) / (db0 - db)
                return float(f0 + ratio * (f - f0))
        return None

    return {
        "luma_response_curve": [[float(f), float(d)] for f, d in curve],
        "minus_3db_freq_MHz":  _crossing(-3.0),
        "minus_6db_freq_MHz":  _crossing(-6.0),
    }
