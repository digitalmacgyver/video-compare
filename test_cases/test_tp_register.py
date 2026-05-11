#!/usr/bin/env python3
"""Tests for tp_register: landmark detection + affine fit."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_register
import tp_fixtures


def approx(a, b, tol):
    return abs(a - b) <= tol


def test_detect_landmark_on_synthesized_ideal():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    for lm in tp_chart.GRID_LANDMARKS:
        result = tp_register.detect_landmark(
            Y, lm["ideal_x"], lm["ideal_y"], lm["search_window_px"],
        )
        assert result is not None, f"no detection at {lm['id']}"
        det_x, det_y, conf = result
        assert approx(det_x, lm["ideal_x"], 0.6), (
            f"{lm['id']} x: got {det_x:.2f}, want {lm['ideal_x']}"
        )
        assert approx(det_y, lm["ideal_y"], 0.6), (
            f"{lm['id']} y: got {det_y:.2f}, want {lm['ideal_y']}"
        )
        assert conf > 0.05


def test_detect_landmark_off_grid_returns_none():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    # Look in a flat-grey region: cell centre at (270, 189) -- 30 px from
    # the nearest grid line at x=240/300 and 27 px from y=162/216, well
    # outside any 24-px search window.
    result = tp_register.detect_landmark(Y, 270, 189, 24)
    assert result is None or result[2] < 0.05


def test_fit_affine_identity():
    pts = np.array([[100.0, 50.0], [300.0, 50.0], [200.0, 200.0],
                    [50.0, 300.0], [400.0, 250.0]], dtype=np.float32)
    result = tp_register.fit_affine(pts.copy(), pts.copy())
    M = result["affine_matrix"]
    assert M is not None
    # Identity: M = [[1, 0, 0], [0, 1, 0]]
    np.testing.assert_allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-3)
    assert result["residuals_px"]["mean"] < 1e-3
    assert result["inliers"] == len(pts)


def test_fit_affine_translation():
    ideal = np.array([[100.0, 50.0], [300.0, 50.0], [200.0, 200.0],
                      [50.0, 300.0], [400.0, 250.0]], dtype=np.float32)
    detected = ideal + np.array([5.0, 7.0], dtype=np.float32)
    result = tp_register.fit_affine(detected, ideal)
    M = result["affine_matrix"]
    assert M is not None
    np.testing.assert_allclose(M[:, 2], [5.0, 7.0], atol=0.1)
    np.testing.assert_allclose(M[:, :2], np.eye(2), atol=1e-3)
    assert result["residuals_px"]["max"] < 0.5


def test_fit_affine_rejects_outlier():
    ideal = np.array([[100.0, 50.0], [300.0, 50.0], [200.0, 200.0],
                      [50.0, 300.0], [400.0, 250.0], [550.0, 400.0]],
                     dtype=np.float32)
    # All shift by (3, 4) except the last is wildly off.
    detected = ideal + np.array([3.0, 4.0], dtype=np.float32)
    detected[-1] += np.array([50.0, 60.0], dtype=np.float32)
    result = tp_register.fit_affine(detected, ideal)
    assert result["inliers"] == 5
    assert result["total"] == 6
    np.testing.assert_allclose(result["affine_matrix"][:, 2], [3.0, 4.0], atol=0.1)


def test_register_identity_on_synthesized_ideal():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    result = tp_register.register(Y)
    assert result["affine_matrix"] is not None
    np.testing.assert_allclose(result["affine_matrix"], [[1, 0, 0], [0, 1, 0]], atol=0.5)
    assert result["residuals_px"]["mean"] < 0.6
    assert result["quality_flag"] == "ok"
    assert result["inliers"] >= len(tp_chart.GRID_LANDMARKS) - 1


def test_register_recovers_translation():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    # Shift the synthesized frame down-right by (4, 3) by rolling the array.
    shifted = np.roll(np.roll(Y, 4, axis=1), 3, axis=0)
    result = tp_register.register(shifted)
    assert result["affine_matrix"] is not None
    np.testing.assert_allclose(
        result["affine_matrix"][:, 2], [4.0, 3.0], atol=0.7
    )
    np.testing.assert_allclose(result["affine_matrix"][:, :2], np.eye(2), atol=0.05)
    assert result["quality_flag"] == "ok"


def test_register_quality_flag_failure_when_no_landmarks_detect():
    # All-grey frame: no grid -> no detections.
    Y = np.full((486, 720), tp_chart.GREY_BACKGROUND_Y10, dtype=np.uint16)
    result = tp_register.register(Y)
    assert result["quality_flag"] == "failed"
    assert result["affine_matrix"] is None


def test_detect_fiducial_dispatches_grid_intersection():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    lm = tp_chart.GRID_LANDMARKS[0]
    det = tp_register.detect_fiducial(Y, lm)
    assert det is not None
    # grid_intersection contract: returns (x, y, confidence)
    x, y, conf = det
    truth_x, truth_y = gt["grid_intersections"][lm["id"]]
    assert abs(x - truth_x) < 0.5
    assert abs(y - truth_y) < 0.5


def test_detect_fiducial_unknown_kind_raises():
    Y, _, _, _ = tp_fixtures.synthesize_with_ground_truth()
    raised = False
    try:
        tp_register.detect_fiducial(Y, {"id": "?", "kind": "no_such_thing"})
    except ValueError:
        raised = True
    assert raised


def test_detect_boundary_triangle_identity():
    """Identity variant: back corners within 0.5 px; apex within 1.0 px."""
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    truth = gt["triangles"]["TL"]
    bc1_x, bc1_y = result["back_corner_1"]
    assert abs(bc1_x - truth["back_corner_1"][0]) < 0.5
    assert abs(bc1_y - truth["back_corner_1"][1]) < 0.5
    bc2_x, bc2_y = result["back_corner_2"]
    assert abs(bc2_x - truth["back_corner_2"][0]) < 0.5
    assert abs(bc2_y - truth["back_corner_2"][1]) < 0.5
    apex_x, apex_y = result["apex_inferred"]
    assert abs(apex_x - truth["apex"][0]) < 1.0
    assert abs(apex_y - truth["apex"][1]) < 1.0


def test_detect_boundary_triangle_noise_sigma_20():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(noise_sigma=20.0)
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    truth = gt["triangles"]["TL"]
    for key in ("back_corner_1", "back_corner_2"):
        dx = result[key][0] - truth[key][0]
        dy = result[key][1] - truth[key][1]
        assert (dx ** 2 + dy ** 2) ** 0.5 < 1.0, f"{key} err > 1.0 px"


def test_detect_boundary_triangle_blur_sigma_1_5():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(blur_sigma=1.5)
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    truth = gt["triangles"]["TL"]
    for key in ("back_corner_1", "back_corner_2"):
        dx = result[key][0] - truth[key][0]
        dy = result[key][1] - truth[key][1]
        assert (dx ** 2 + dy ** 2) ** 0.5 < 1.5, f"{key} err > 1.5 px"


def test_detect_boundary_triangle_clip_top_apex_inferred_only():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(clip_top=5)
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    # Back corners still detected to threshold.
    truth = gt["triangles"]["TL"]
    for key in ("back_corner_1", "back_corner_2"):
        dx = result[key][0] - truth[key][0]
        dy = result[key][1] - truth[key][1]
        assert (dx ** 2 + dy ** 2) ** 0.5 < 0.5
    # apex_detected is None (clipped), apex_inferred is computed from
    # back_midpoint + chart offset.
    assert result["apex_detected"] is None
    assert result["apex_inferred"] is not None


TESTS = [
    test_detect_landmark_on_synthesized_ideal,
    test_detect_landmark_off_grid_returns_none,
    test_fit_affine_identity,
    test_fit_affine_translation,
    test_fit_affine_rejects_outlier,
    test_register_identity_on_synthesized_ideal,
    test_register_recovers_translation,
    test_register_quality_flag_failure_when_no_landmarks_detect,
    test_detect_fiducial_dispatches_grid_intersection,
    test_detect_fiducial_unknown_kind_raises,
    test_detect_boundary_triangle_identity,
    test_detect_boundary_triangle_noise_sigma_20,
    test_detect_boundary_triangle_blur_sigma_1_5,
    test_detect_boundary_triangle_clip_top_apex_inferred_only,
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
