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


def test_fixture_shift_updates_ground_truth_and_frame():
    Y0, _, _, gt0 = tp_fixtures.synthesize_with_ground_truth()
    Y, U, V, gt = tp_fixtures.synthesize_with_ground_truth(shift=(5, 7))
    # Pick whichever L1 lives at — its ground truth must shift by (5,7).
    l1_orig = gt0["grid_intersections"]["L1"]
    assert gt["grid_intersections"]["L1"] == (l1_orig[0] + 5, l1_orig[1] + 7)
    # The pixel at the new position should be black (grid intersection).
    nx, ny = gt["grid_intersections"]["L1"]
    assert Y[ny, nx] == tp_chart.BLACK_Y10
    # Triangle TL back corner shifted by (5, 7).
    tl_orig = gt0["triangles"]["TL"]["back_corner_1"]
    assert gt["triangles"]["TL"]["back_corner_1"] == (tl_orig[0] + 5, tl_orig[1] + 7)
    # First 7 rows are the grey-background fill from the shift.
    assert (Y[:7, :] == tp_chart.GREY_BACKGROUND_Y10).all()
    # First 5 columns are the grey-background fill.
    assert (Y[:, :5] == tp_chart.GREY_BACKGROUND_Y10).all()


def test_fixture_rotation_updates_ground_truth_positions():
    import math
    Y, U, V, gt = tp_fixtures.synthesize_with_ground_truth(rotation_deg=1.5)
    # Rotation about picture center (360, 243). For cv2.getRotationMatrix2D
    # with positive angle theta and y-down coords, the matrix is:
    #   R = [[cos, sin, ...], [-sin, cos, ...]]
    # so a point (px, py) maps to (cx + (px-cx)*cos + (py-cy)*sin,
    #                              cy - (px-cx)*sin + (py-cy)*cos).
    theta = math.radians(1.5)
    cx, cy = 360.0, 243.0
    # Pick any landmark to test against.
    lm = tp_chart.GRID_LANDMARKS[0]
    px, py = float(lm["ideal_x"]), float(lm["ideal_y"])
    new_x = cx + (px - cx) * math.cos(theta) + (py - cy) * math.sin(theta)
    new_y = cy - (px - cx) * math.sin(theta) + (py - cy) * math.cos(theta)
    gx, gy = gt["grid_intersections"][lm["id"]]
    assert abs(gx - new_x) < 0.6, f"{lm['id']} ground truth x mismatch: {gx} vs {new_x}"
    assert abs(gy - new_y) < 0.6, f"{lm['id']} ground truth y mismatch: {gy} vs {new_y}"


TESTS = [
    test_fixture_identity_ground_truth_matches_chart_catalog,
    test_fixture_shift_updates_ground_truth_and_frame,
    test_fixture_rotation_updates_ground_truth_positions,
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
