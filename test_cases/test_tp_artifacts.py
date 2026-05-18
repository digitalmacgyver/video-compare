#!/usr/bin/env python3
"""Tests for tp_artifacts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_artifacts


def _identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def test_artifacts_measure_returns_all_regions():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    assert set(out["regions"].keys()) == {r["id"] for r in tp_chart.ARTIFACT_REGIONS}


def test_zone_plate_chroma_leak_synthesized_is_low():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    zp = out["regions"]["ZP_CHROMA_LEAK"]
    assert zp["chroma_rms"] < 5.0, f"zp.chroma_rms={zp['chroma_rms']:.2f}"
    assert zp["chroma_present"] is False


def test_zone_plate_chroma_leak_detects_injected_chroma():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    U2 = U.copy(); V2 = V.copy()
    # Inject chroma into a slab inside the zone-plate region (cells 3,4-6,9).
    # U/V are half-x; inject at half-x indices [90..270] (covers x=180..540).
    U2[120:270, 100:260] = 600
    V2[120:270, 100:260] = 600
    out = tp_artifacts.measure(Y, U2, V2, _identity())
    zp = out["regions"]["ZP_CHROMA_LEAK"]
    assert zp["chroma_rms"] > 15.0
    assert zp["chroma_present"] is True


def test_cross_color_metric_zero_on_synthesized_bursts():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    for rid in ("XC_BURST_300TVL", "XC_BURST_400TVL",
                "XC_WEDGE_4MHz", "XC_WEDGE_5MHz"):
        r = out["regions"][rid]
        assert r["chroma_rms"] < 3.0, f"{rid} chroma_rms={r['chroma_rms']:.2f}"


def test_summary_keys_present():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    s = out["summary"]
    for key in ("max_hanging_dots_y10_pp", "max_dot_crawl_chroma_rms",
                "max_cross_color_chroma_rms", "max_cross_luma_y10_pp",
                "zone_plate_chroma_present", "zone_plate_chroma_rms"):
        assert key in s, f"missing summary key: {key}"


def test_wedge_hv_symmetry_balanced_on_synth_radial_wedge():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    w = out["regions"]["WEDGE_HV_SYMMETRY"]
    # The synth now renders a clean Siemens-star-style radial wedge,
    # so both H and V cross-sections carry strong luma modulation.
    # An unbiased decoder reports the H/V ratio close to 1.0.
    assert w.get("h_modulation_std", 0) > 50.0
    assert w.get("v_modulation_std", 0) > 50.0
    ratio = w.get("hv_ratio")
    assert ratio is not None and 0.7 < ratio < 1.4, w


def test_wedge_hv_symmetry_detects_h_only_pattern():
    """Inject vertical stripes (high H modulation, zero V) into the wedge
    region and confirm the detector reports hv_ratio >> 1."""
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Vertical stripes at 3 px period across the wedge box.
    x, y, w, h = (604, 384, 52, 44)
    cols = np.arange(w)
    pattern = np.where(cols % 4 < 2, 64, 940).astype(np.uint16)
    Y[y:y + h, x:x + w] = pattern[None, :]
    out = tp_artifacts.measure(Y, U, V, _identity())
    wsym = out["regions"]["WEDGE_HV_SYMMETRY"]
    assert wsym["h_modulation_std"] > 100, wsym
    assert wsym["v_modulation_std"] < 10, wsym
    # With v ~ 0 the ratio is either None (formally undefined) or very
    # large; either signals "H dominates V".
    ratio = wsym["hv_ratio"]
    assert ratio is None or ratio > 10, wsym


TESTS = [
    test_artifacts_measure_returns_all_regions,
    test_zone_plate_chroma_leak_synthesized_is_low,
    test_zone_plate_chroma_leak_detects_injected_chroma,
    test_cross_color_metric_zero_on_synthesized_bursts,
    test_summary_keys_present,
    test_wedge_hv_symmetry_balanced_on_synth_radial_wedge,
    test_wedge_hv_symmetry_detects_h_only_pattern,
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
