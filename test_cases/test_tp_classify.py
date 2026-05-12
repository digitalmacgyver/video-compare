#!/usr/bin/env python3
"""Tests for tp_classify."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tp_classify


def _clean_freq():
    return {
        "regions": {
            "WEDGE_3MHz": {"frequency_MHz_expected": 3.0, "modulation_pct": 92.0, "modulation_db": -0.7},
            "WEDGE_4MHz": {"frequency_MHz_expected": 4.0, "modulation_pct": 88.0, "modulation_db": -1.1},
            "WEDGE_5MHz": {"frequency_MHz_expected": 5.0, "modulation_pct": 65.0, "modulation_db": -3.7},
        },
        "summary": {"luma_response_curve": [], "minus_3db_freq_MHz": 4.5, "minus_6db_freq_MHz": 5.2},
    }


def _clean_art():
    return {
        "regions": {},
        "summary": {
            "max_hanging_dots_y10_pp":    4.0,
            "max_dot_crawl_chroma_rms":   1.5,
            "max_cross_color_chroma_rms": 3.0,
            "max_cross_luma_y10_pp":      6.0,
            "zone_plate_chroma_present":  False,
            "zone_plate_chroma_rms":      3.1,
        },
    }


def test_clean_signal_classifies_as_temporal_or_adaptive():
    out = tp_classify.classify(_clean_freq(), _clean_art())
    assert out["decoder_class"] == "temporal_comb_or_adaptive"
    assert out["confidence"] >= 0.5


def test_high_xc_and_zp_classifies_as_notch():
    freq = _clean_freq()
    freq["regions"]["WEDGE_5MHz"]["modulation_pct"] = 14.0  # band-limited
    art = _clean_art()
    art["summary"]["max_cross_color_chroma_rms"] = 35.0
    art["summary"]["zone_plate_chroma_present"] = True
    art["summary"]["zone_plate_chroma_rms"] = 32.0
    out = tp_classify.classify(freq, art)
    assert out["decoder_class"] == "notch"


def test_high_hd_classifies_as_line_comb():
    art = _clean_art()
    art["summary"]["max_hanging_dots_y10_pp"] = 45.0
    out = tp_classify.classify(_clean_freq(), art)
    assert out["decoder_class"] == "line_comb"


def test_ambiguous_returns_undetermined():
    freq = _clean_freq()
    art = _clean_art()
    art["summary"]["max_cross_color_chroma_rms"] = 16.0  # in the middle
    art["summary"]["max_hanging_dots_y10_pp"] = 18.0     # in the middle
    art["summary"]["max_cross_luma_y10_pp"] = 12.0       # near boundary
    out = tp_classify.classify(freq, art)
    assert out["decoder_class"] == "undetermined"


def test_classify_emits_candidate_confidences():
    out = tp_classify.classify(_clean_freq(), _clean_art())
    cands = out["evidence"]["candidate_confidences"]
    assert set(cands.keys()) == {"notch", "line_comb", "temporal_comb_or_adaptive"}
    for v in cands.values():
        assert 0.0 <= v <= 1.0


TESTS = [
    test_clean_signal_classifies_as_temporal_or_adaptive,
    test_high_xc_and_zp_classifies_as_notch,
    test_high_hd_classifies_as_line_comb,
    test_ambiguous_returns_undetermined,
    test_classify_emits_candidate_confidences,
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
