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

    # Grid intersections — count and ids match whatever the catalog is.
    grids = gt["grid_intersections"]
    assert len(grids) == len(tp_chart.GRID_LANDMARKS)
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


def test_fixture_noise_only_changes_pixel_values():
    Y0, _, _, gt0 = tp_fixtures.synthesize_with_ground_truth()
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(noise_sigma=20.0)
    # Ground truth identical.
    assert gt == gt0
    # Y differs.
    diff = (Y.astype(np.int32) - Y0.astype(np.int32)).astype(np.float64)
    assert abs(diff.mean()) < 5.0, "noise should be zero-mean on average"
    assert 15.0 < diff.std() < 25.0, f"noise sigma ~20 expected, got {diff.std()}"


def test_fixture_blur_softens_edges_but_preserves_truth():
    Y0, _, _, gt0 = tp_fixtures.synthesize_with_ground_truth()
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(blur_sigma=1.5)
    assert gt == gt0
    # A grid-line pixel was black (Y10=64); after a 1.5σ blur over ~9 px it
    # rises but should still be substantially darker than grey.
    # Pick any grid landmark.
    lm = tp_chart.GRID_LANDMARKS[0]
    gx, gy = lm["ideal_x"], lm["ideal_y"]
    assert Y[gy, gx] > tp_chart.BLACK_Y10
    assert Y[gy, gx] < tp_chart.GREY_BACKGROUND_Y10 - 50


def test_fixture_clip_top_invalidates_TL_TR_apexes():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(clip_top=5)
    # Top 5 rows zeroed.
    assert (Y[:5, :] == 0).all()
    # TL apex was at y=3 -- now in the clipped region.
    assert gt["triangles"]["TL"]["apex_visible"] is False
    assert gt["triangles"]["TR"]["apex_visible"] is False
    # Back corners at y=30 still visible.
    tl_orig_bc1 = tp_chart.BOUNDARY_TRIANGLES[0]["ideal_back_corner_1"]
    assert gt["triangles"]["TL"]["back_corner_1"] == tl_orig_bc1
    # BL/BR unaffected.
    assert gt["triangles"]["BL"]["apex_visible"] is True
    assert gt["triangles"]["BR"]["apex_visible"] is True


def test_fixture_clip_bottom_invalidates_BL_BR_apexes():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(clip_bottom=5)
    assert (Y[-5:, :] == 0).all()
    # BL apex was at y=482 -- now in the clipped region.
    assert gt["triangles"]["BL"]["apex_visible"] is False
    assert gt["triangles"]["BR"]["apex_visible"] is False
    assert gt["triangles"]["TL"]["apex_visible"] is True


TESTS = [
    test_fixture_identity_ground_truth_matches_chart_catalog,
    test_fixture_shift_updates_ground_truth_and_frame,
    test_fixture_rotation_updates_ground_truth_positions,
    test_fixture_noise_only_changes_pixel_values,
    test_fixture_blur_softens_edges_but_preserves_truth,
    test_fixture_clip_top_invalidates_TL_TR_apexes,
    test_fixture_clip_bottom_invalidates_BL_BR_apexes,
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
