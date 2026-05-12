#!/usr/bin/env python3
"""Tests for tp_freq: per-burst frequency-response measurement."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_freq


def _identity_affine():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def test_freq_measure_returns_all_regions():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_freq.measure(Y, U, V, _identity_affine())
    assert set(out["regions"].keys()) == {r["id"] for r in tp_chart.BURST_REGIONS}


def test_freq_measure_synthesized_high_modulation():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_freq.measure(Y, U, V, _identity_affine())
    for rid, rdata in out["regions"].items():
        assert rdata["modulation_pct"] > 30.0, (
            f"{rid}: modulation_pct={rdata['modulation_pct']:.1f}"
        )


def test_freq_measure_band_limited_drops_high_freq():
    import cv2
    Y, U, V = tp_synthesize.synthesize(720, 486)
    Yf = cv2.boxFilter(Y.astype(np.float32), -1, (5, 1))
    Yblur = Yf.astype(np.uint16)
    base = tp_freq.measure(Y, U, V, _identity_affine())
    blur = tp_freq.measure(Yblur, U, V, _identity_affine())
    assert (blur["regions"]["WEDGE_5MHz"]["modulation_pct"]
            < base["regions"]["WEDGE_5MHz"]["modulation_pct"] - 10), (
        "expected band-limit to drop WEDGE_5MHz modulation"
    )


def test_freq_measure_summary_has_minus3db():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_freq.measure(Y, U, V, _identity_affine())
    s = out["summary"]
    assert "luma_response_curve" in s
    assert "minus_3db_freq_MHz" in s
    assert "minus_6db_freq_MHz" in s
    assert len(s["luma_response_curve"]) == len(tp_chart.BURST_REGIONS)


TESTS = [
    test_freq_measure_returns_all_regions,
    test_freq_measure_synthesized_high_modulation,
    test_freq_measure_band_limited_drops_high_freq,
    test_freq_measure_summary_has_minus3db,
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
