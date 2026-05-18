"""Synthetic tests for geometry.derived.summary (layperson-friendly block).

Each test calls tp_register.register_with_geometry on a synth/fixture
frame with known ground truth and asserts the summary fields land within
a small tolerance of the expected values.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tp_chart
import tp_fixtures
import tp_register
import tp_synthesize


def _summary(Y):
    return tp_register.register_with_geometry(Y)["geometry"]["derived"]["summary"]


def test_identity_synth_arrow_spacings_within_tolerance():
    """On a clean synth, every arrow spacing should land within 2 px of
    the chart-spec ideal (357 horizontal, 483 vertical)."""
    Y, _, _ = tp_synthesize.synthesize()
    s = _summary(Y)
    sp = s["arrow_spacings_px"]
    assert sp is not None
    for edge in ("top", "bottom"):
        assert abs(sp[edge]["delta"]) < 2.0, f"{edge} delta {sp[edge]['delta']}"
        assert sp[edge]["ideal"] == 357
    for edge in ("left", "right"):
        assert abs(sp[edge]["delta"]) < 2.0, f"{edge} delta {sp[edge]['delta']}"
        assert sp[edge]["ideal"] == 483


def test_identity_synth_center_offset_near_zero():
    Y, _, _ = tp_synthesize.synthesize()
    s = _summary(Y)
    off = s["picture_center_offset_px"]
    assert off is not None
    assert abs(off["dx"]) < 1.0, off["dx"]
    assert abs(off["dy"]) < 1.0, off["dy"]


def test_identity_synth_scale_near_100_percent():
    Y, _, _ = tp_synthesize.synthesize()
    s = _summary(Y)
    sc = s["picture_scale_pct"]
    assert sc is not None
    assert abs(sc["horizontal"] - 100.0) < 1.0, sc["horizontal"]
    assert abs(sc["vertical"]   - 100.0) < 1.0, sc["vertical"]


def test_identity_synth_keystone_near_zero():
    Y, _, _ = tp_synthesize.synthesize()
    s = _summary(Y)
    ks = s["keystone_px"]
    assert ks is not None
    assert abs(ks["horizontal_top_minus_bottom"]) < 2.0
    assert abs(ks["vertical_left_minus_right"])   < 2.0


def test_shift_synth_center_offset_matches_shift():
    """A 5-px down, 3-px right shift should show up as dy≈+5, dx≈+3."""
    Y, _, _, _ = tp_fixtures.synthesize_with_ground_truth(shift=(3, 5))
    s = _summary(Y)
    off = s["picture_center_offset_px"]
    assert off is not None
    assert abs(off["dx"] - 3.0) < 1.5, off["dx"]
    assert abs(off["dy"] - 5.0) < 1.5, off["dy"]


def test_par_elliptical_synth_round_in_display():
    """The synth ring is rendered PAR-elliptical (horizontal axis ×
    11/10). The reported displayed_circularity should land near 1.0."""
    Y, _, _ = tp_synthesize.synthesize()
    s = _summary(Y)
    circ = s["circle"]
    assert circ is not None
    dc = circ["displayed_circularity"]
    assert dc is not None
    # 0.05 tolerance covers the few-px bias from the ring's 3-px thickness
    # and antialias picked up by the bounding-box axis-extraction.
    assert abs(dc - 1.0) < 0.05, f"displayed_circularity = {dc}"


def test_par_elliptical_synth_h_v_axes_match_par():
    Y, _, _ = tp_synthesize.synthesize()
    s = _summary(Y)
    circ = s["circle"]
    assert circ["expected_h_over_v_for_round"] == tp_chart.NTSC_PAR_X_OVER_Y
    actual = circ["actual_h_over_v"]
    assert abs(actual - tp_chart.NTSC_PAR_X_OVER_Y) < 0.06, actual


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
