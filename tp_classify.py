"""Decoder-class classifier (Stage 3).

Rule-based heuristic that consumes tp_freq.measure() + tp_artifacts.measure()
outputs and infers a discrete decoder class:

  - notch:                    high cross-color, low chroma BW, zone-plate
                              chroma leak present (whitepaper "Output from
                              simple NTSC notch").
  - line_comb:                low cross-color, high hanging-dots amplitude
                              (whitepaper "Hanging dots ... vertical
                              chrominance transition").
  - temporal_comb_or_adaptive: low cross-color, low hanging dots, low
                              cross-luma, no zone-plate leak (whitepaper
                              "Field comb decoded"). Distinguishing
                              temporal from adaptive requires zone-plate
                              motion analysis (Stage 3.5).
  - undetermined:             no rule's confidence reaches 0.5.

Confidence per rule is the geometric mean of per-metric confidences. Each
per-metric confidence is a linear interp between rule-threshold and
"hard-yes" extreme; clipped to [0, 1].
"""
from __future__ import annotations
import math

import tp_artifacts


def _ge_conf(value, threshold, hard_yes):
    """Confidence for "value >= threshold" rule. 0 at <= threshold,
    1 at >= hard_yes."""
    if hard_yes <= threshold:
        return 1.0 if value >= threshold else 0.0
    if value <= threshold:
        return 0.0
    if value >= hard_yes:
        return 1.0
    return (value - threshold) / (hard_yes - threshold)


def _le_conf(value, threshold, hard_yes):
    """Confidence for "value <= threshold" rule. 0 at >= threshold,
    1 at <= hard_yes."""
    if hard_yes >= threshold:
        return 1.0 if value <= threshold else 0.0
    if value >= threshold:
        return 0.0
    if value <= hard_yes:
        return 1.0
    return (threshold - value) / (threshold - hard_yes)


def _geom_mean(values):
    values = [max(v, 1e-6) for v in values]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def classify(freq, artifacts):
    T = tp_artifacts.THRESHOLDS
    s = (artifacts.get("summary") or {}) if isinstance(artifacts, dict) else {}
    xc = float(s.get("max_cross_color_chroma_rms", 0.0))
    hd = float(s.get("max_hanging_dots_y10_pp", 0.0))
    xl = float(s.get("max_cross_luma_y10_pp", 0.0))
    zp_present = bool(s.get("zone_plate_chroma_present", False))
    zp_rms = float(s.get("zone_plate_chroma_rms", 0.0))
    freq_regions = (freq.get("regions") or {}) if isinstance(freq, dict) else {}
    w5 = freq_regions.get("WEDGE_5MHz", {}) or {}
    w5_mod = float(w5.get("modulation_pct", 100.0))

    # Rule 1: notch / simple low-pass
    c_xc_high = _ge_conf(xc, T["T_xc_high"], T["T_xc_high"] * 2.0)
    c_w5_low = _le_conf(w5_mod, T["T_chroma_bw_low_pct"], T["T_chroma_bw_low_pct"] * 0.5)
    c_zp_present = 1.0 if zp_present else 0.0
    notch_conf = _geom_mean([c_xc_high, c_w5_low, c_zp_present])

    # Rule 2: line comb
    c_xc_low = _le_conf(xc, T["T_xc_low"], T["T_xc_low"] * 0.5)
    c_hd_high = _ge_conf(hd, T["T_hd_high"], T["T_hd_high"] * 2.0)
    line_comb_conf = _geom_mean([c_xc_low, c_hd_high])

    # Rule 3: temporal-comb or adaptive (cannot be subclassified without
    # zone-plate motion analysis -- deferred to Stage 3.5).
    c_hd_low = _le_conf(hd, T["T_hd_low"], T["T_hd_low"] * 0.5)
    c_xl_low = _le_conf(xl, T["T_xl_low"], T["T_xl_low"] * 0.5)
    c_zp_absent = 1.0 if not zp_present else 0.0
    tca_conf = _geom_mean([c_xc_low, c_hd_low, c_xl_low, c_zp_absent])

    candidates = [
        ("notch", notch_conf, {
            "max_cross_color_chroma_rms": xc,
            "WEDGE_5MHz_modulation_pct":  w5_mod,
            "zone_plate_chroma_present":  zp_present,
            "zone_plate_chroma_rms":      zp_rms,
        }),
        ("line_comb", line_comb_conf, {
            "max_cross_color_chroma_rms": xc,
            "max_hanging_dots_y10_pp":    hd,
        }),
        ("temporal_comb_or_adaptive", tca_conf, {
            "max_cross_color_chroma_rms": xc,
            "max_hanging_dots_y10_pp":    hd,
            "max_cross_luma_y10_pp":      xl,
            "zone_plate_chroma_present":  zp_present,
        }),
    ]
    candidates.sort(key=lambda c: c[1], reverse=True)
    best_class, best_conf, best_metrics = candidates[0]
    cand_confs = {c[0]: float(c[1]) for c in candidates}

    if best_conf < 0.5:
        return {
            "decoder_class": "undetermined",
            "confidence":    float(best_conf),
            "evidence": {
                "rule_fired":             None,
                "candidate_confidences":  cand_confs,
                "contributing_metrics":   best_metrics,
            },
            "thresholds_used": dict(T),
        }
    return {
        "decoder_class": best_class,
        "confidence":    float(best_conf),
        "evidence": {
            "rule_fired":             f"{best_class}_rule",
            "candidate_confidences":  cand_confs,
            "contributing_metrics":   best_metrics,
        },
        "thresholds_used": dict(T),
    }
