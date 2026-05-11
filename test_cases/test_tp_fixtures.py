#!/usr/bin/env python3
"""Tests for tp_fixtures.synthesize_with_ground_truth."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_fixtures


def test_fixture_identity_ground_truth_matches_chart_catalog():
    Y, U, V, gt = tp_fixtures.synthesize_with_ground_truth()
    assert Y.shape == (486, 720)
    assert U.shape == (486, 360)
    assert V.shape == (486, 360)

    # 12 grid intersections with ids L1..L12
    grids = gt["grid_intersections"]
    assert len(grids) == 12
    for lm in tp_chart.GRID_LANDMARKS:
        assert grids[lm["id"]] == (lm["ideal_x"], lm["ideal_y"])

    # 4 triangles
    tris = gt["triangles"]
    assert set(tris.keys()) == {"TL", "TR", "BL", "BR"}
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        entry = tris[tri["id"]]
        assert entry["back_corner_1"] == tri["ideal_back_corner_1"]
        assert entry["back_corner_2"] == tri["ideal_back_corner_2"]
        assert entry["back_midpoint"] == tri["ideal_back_midpoint"]
        assert entry["apex"]          == tri["ideal_apex"]
        assert entry["apex_visible"] is True

    # Cross
    rc = gt["cross"]
    assert rc["center"] == (tp_chart.REGISTRATION_CROSS["ideal_x"],
                            tp_chart.REGISTRATION_CROSS["ideal_y"])

    # Circle
    cc = gt["circle"]
    assert cc["center"] == (tp_chart.BLACK_CIRCLE["ideal_cx"],
                            tp_chart.BLACK_CIRCLE["ideal_cy"])
    assert cc["radius"] == tp_chart.BLACK_CIRCLE["expected_radius_px"]


TESTS = [
    test_fixture_identity_ground_truth_matches_chart_catalog,
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
