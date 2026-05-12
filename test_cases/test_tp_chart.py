#!/usr/bin/env python3
"""Tests for tp_chart constants and YUV math."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tp_chart


def approx(actual, expected, tol):
    return abs(actual - expected) <= tol


def test_constants_exist():
    assert tp_chart.TP_CHART_VERSION == 1
    assert tp_chart.BLACK_Y10 == 64
    assert tp_chart.WHITE_Y10 == 940
    assert tp_chart.CHROMA_CENTER == 512
    # Grey background = 50% IRE in BT.601 limited range = 64 + 0.5*876 = 502
    assert tp_chart.GREY_BACKGROUND_Y10 == 502
    # 4-step grays at 20/40/60/80% IRE
    assert tp_chart.GRAY_IDEAL_Y10 == (239.2, 414.4, 589.6, 764.8)


def test_rgb_norm_to_yuv10_black():
    y, u, v = tp_chart.rgb_norm_to_yuv10(0.0, 0.0, 0.0)
    assert approx(y, 64.0, 0.1)
    assert approx(u, 512.0, 0.1)
    assert approx(v, 512.0, 0.1)


def test_rgb_norm_to_yuv10_white():
    y, u, v = tp_chart.rgb_norm_to_yuv10(1.0, 1.0, 1.0)
    assert approx(y, 940.0, 0.1)
    assert approx(u, 512.0, 0.1)
    assert approx(v, 512.0, 0.1)


def test_rgb_norm_to_yuv10_75_yellow():
    # R=G=0.75, B=0 -> Y = 0.299*0.75 + 0.587*0.75 = 0.6645
    # Y10 = 64 + 876 * 0.6645 = 646.1
    y, u, v = tp_chart.rgb_norm_to_yuv10(0.75, 0.75, 0.0)
    assert approx(y, 646.1, 0.5)
    assert approx(u, 176.0, 1.0)   # 512 + 896 * 0.5 * (-0.6645)/0.886
    assert approx(v, 566.7, 1.0)   # 512 + 896 * 0.5 * (0.75 - 0.6645)/0.701


def test_yuv10_to_rgb8_round_trip():
    y, u, v = tp_chart.rgb_norm_to_yuv10(0.5, 0.5, 0.5)
    r8, g8, b8 = tp_chart.yuv10_to_rgb8(y, u, v)
    assert approx(r8, 128, 1)
    assert approx(g8, 128, 1)
    assert approx(b8, 128, 1)


def test_tartan_regions_count_and_ids():
    ids = [r["id"] for r in tp_chart.TARTAN_REGIONS]
    assert len(ids) == 8
    assert set(ids) == {"YEL", "CYN", "BLU", "RED",
                        "MAG", "GRN", "RED2", "CYN2"}


def test_tartan_bottom_row_75_magenta_via_rec601():
    # Bottom row is full 75% saturation, same Rec.601 math as the top row.
    # Magenta RGB = (0.75, 0, 0.75)  ->  Y10 ~= 335.3, U10 ~= 734.7, V10 ~= 793.5
    mag = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "MAG")
    e = mag["expected"]
    assert approx(e["y10"], 335.3, 0.5)
    assert approx(e["u10"], 734.7, 1.0)
    assert approx(e["v10"], 793.5, 1.0)


def test_tartan_red2_equals_top_red():
    # The bottom-row RED2 has the same color as the top-row RED (intentional:
    # creates a no-chroma-change column to contrast with adjacent transitions).
    red = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "RED")
    red2 = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "RED2")
    assert red["expected"] == red2["expected"]


def test_tartan_region_record_shape():
    for r in tp_chart.TARTAN_REGIONS:
        assert "id" in r and "name" in r and "kind" in r
        assert r["kind"] == "tartan_rect"
        x, y, w, h = r["ideal_box"]
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= 720
        assert 0 <= y and y + h <= 486
        assert "expected" in r
        assert "y10" in r["expected"] and "u10" in r["expected"] and "v10" in r["expected"]
        assert r["sample"] == {"kind": "center_window", "size_frac": 0.2}


def test_tartan_75_yellow_expected():
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    e = yel["expected"]
    assert approx(e["y10"], 646.1, 0.5)
    assert approx(e["u10"], 176.0, 1.0)
    assert approx(e["v10"], 566.7, 1.0)


def test_gray_regions():
    assert len(tp_chart.GRAY_REGIONS) == 4
    ids = [r["id"] for r in tp_chart.GRAY_REGIONS]
    assert ids == ["G1", "G2", "G3", "G4"]
    for r, expected_y in zip(tp_chart.GRAY_REGIONS, tp_chart.GRAY_IDEAL_Y10):
        assert r["kind"] == "gray_step"
        assert approx(r["expected"]["y10"], expected_y, 0.05)
        assert r["expected"]["u10"] == 512
        assert r["expected"]["v10"] == 512
        assert r["sample"] == {"kind": "center_window", "size_frac": 0.2}


def test_grid_landmarks():
    lms = tp_chart.GRID_LANDMARKS
    # Refreshed May 2026 against operator chart-layout review: 11 anchors at
    # intersections clear of the moving zone plate, the SW2/NTSC merged-text
    # box, and the chart features that would confound projection-detection.
    assert len(lms) == 11, f"expected 11 anchors, got {len(lms)}"
    expected_ids = ["L1", "L3", "L4", "L7", "L9", "L10", "L12",
                    "L13", "L14", "L15", "L16"]
    assert [lm["id"] for lm in lms] == expected_ids
    for lm in lms:
        assert lm["kind"] == "grid_intersection"
        assert 0 < lm["ideal_x"] < 720
        assert 0 < lm["ideal_y"] < 486
        assert lm["search_window_px"] >= 16


def test_grid_landmark_distribution():
    # All anchors avoid: tartan/gray strip (upper-left), zone-plate reserved
    # area (cells (3,4)-(6,9) i.e. x in [180,540], y in [108,324]), chart
    # border, S&W banner. Some anchors sit on the zone-plate boundary at
    # x=180 or x=540 -- still safe because the search window (24 px) reaches
    # only 12 px into adjacent cells, and the actual moving pattern stays
    # well inside its container.
    for lm in tp_chart.GRID_LANDMARKS:
        assert 54 <= lm["ideal_y"] <= 432, lm
        assert 60 <= lm["ideal_x"] <= 660, lm
    # Distinct y-rows present (sorted).
    ys = sorted({lm["ideal_y"] for lm in tp_chart.GRID_LANDMARKS})
    assert ys == [54, 108, 162, 216, 270, 378], ys
    xs = sorted({lm["ideal_x"] for lm in tp_chart.GRID_LANDMARKS})
    assert xs == [180, 240, 480, 540], xs


def test_ideal_picture_box():
    box = tp_chart.IDEAL_PICTURE_BOX
    assert box == {"left": 0, "top": 0, "right": 719, "bottom": 485}
    corners = tp_chart.IDEAL_PICTURE_BOX_CORNERS
    # TL, TR, BL, BR
    assert corners == [(0, 0), (719, 0), (0, 485), (719, 485)]


def test_boundary_triangles_catalog():
    tris = tp_chart.BOUNDARY_TRIANGLES
    assert [t["id"] for t in tris] == ["TL", "TR", "BL", "BR"]
    expected_orient = {"TL": "apex_up", "TR": "apex_up",
                       "BL": "apex_down", "BR": "apex_down"}
    for t in tris:
        assert t["kind"] == "boundary_triangle"
        assert t["orientation"] == expected_orient[t["id"]]
        for key in ("ideal_back_corner_1", "ideal_back_corner_2",
                    "ideal_back_midpoint", "ideal_apex",
                    "search_window_px"):
            assert key in t, key
        bm = t["ideal_back_midpoint"]
        bc1 = t["ideal_back_corner_1"]
        bc2 = t["ideal_back_corner_2"]
        # back_midpoint is the midpoint of the two back corners.
        assert bm == ((bc1[0] + bc2[0]) / 2, (bc1[1] + bc2[1]) / 2)
        # apex sits 15 px from back_midpoint in the orientation direction
        # (operator-calibrated; was 27 before chart-layout review).
        if t["orientation"] == "apex_up":
            assert t["ideal_apex"] == (bm[0], bm[1] - 15)
        elif t["orientation"] == "apex_down":
            assert t["ideal_apex"] == (bm[0], bm[1] + 15)


def test_boundary_triangles_back_corner_spacing():
    # 16 px base across the back edge (operator-calibrated; was 20).
    for t in tp_chart.BOUNDARY_TRIANGLES:
        bc1, bc2 = t["ideal_back_corner_1"], t["ideal_back_corner_2"]
        spacing = ((bc1[0] - bc2[0]) ** 2 + (bc1[1] - bc2[1]) ** 2) ** 0.5
        assert abs(spacing - 16.0) < 0.5


def test_registration_cross_catalog():
    rc = tp_chart.REGISTRATION_CROSS
    assert rc["id"] == "RC"
    assert rc["kind"] == "registration_cross"
    # Cross is in the upper-right composite (operator-calibrated to (568, 36)).
    assert rc["ideal_x"] == 568
    assert rc["ideal_y"] == 36
    assert rc["ideal_arm_len_px"] == 17
    assert rc["ideal_arm_thickness_px"] == 3
    assert rc["box_size_px"] == 24
    assert rc["search_window_px"] >= 32


def test_black_circle_catalog():
    bc = tp_chart.BLACK_CIRCLE
    assert bc["id"] == "BC"
    assert bc["kind"] == "black_circle"
    assert bc["ideal_cx"] == 359
    assert bc["ideal_cy"] == 243
    # Chart-spec literal — diameter = picture_height = 486, radius = 243.
    # Real captures are elliptical (rx ≈ 266, ry ≈ 242) due to NTSC 10:11
    # PAR; the detector's circular annulus catches the y-axis side of
    # the ring and reports biased rx/ry. PAR-aware ellipse fit is future work.
    assert bc["expected_radius_px"] == 243
    assert bc["ring_thickness_px"] == 3
    assert bc["search_band_px"] >= 10


def test_burst_regions_catalog():
    ids = [r["id"] for r in tp_chart.BURST_REGIONS]
    assert len(ids) == 8
    expected = {"BURST_3p58", "BURST_4p43", "BURST_4p286_SECAM",
                "BURST_300TVL_DIAG", "BURST_400TVL_DIAG",
                "WEDGE_3MHz", "WEDGE_4MHz", "WEDGE_5MHz"}
    assert set(ids) == expected
    valid_kinds = {"burst_vertical", "burst_diagonal", "wedge_segment"}
    for r in tp_chart.BURST_REGIONS:
        assert r["kind"] in valid_kinds
        assert r["frequency_MHz"] > 0
        x, y, w, h = r["ideal_box"]
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= 720
        assert 0 <= y and y + h <= 486
        # 4:2:2 alignment: x and w must be even.
        assert x % 2 == 0
        assert w % 2 == 0
        if r["kind"] == "burst_diagonal":
            assert "stripe_angle_deg" in r


def test_artifact_regions_catalog():
    ids = [r["id"] for r in tp_chart.ARTIFACT_REGIONS]
    expected = {"HD_RED_TOP", "HD_MAGENTA_TOP", "DC_TARTAN_BELOW",
                "XC_BURST_300TVL", "XC_BURST_400TVL", "XC_WEDGE_4MHz",
                "XC_WEDGE_5MHz", "XL_RED_INTERIOR", "XL_MAGENTA_INTERIOR",
                "ZP_CHROMA_LEAK"}
    assert set(ids) == expected
    valid_kinds = {"hanging_dots", "dot_crawl", "cross_color",
                   "cross_luma", "zone_plate_chroma_leak"}
    for r in tp_chart.ARTIFACT_REGIONS:
        assert r["artifact_kind"] in valid_kinds
        x, y, w, h = r["ideal_box"]
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= 720
        assert 0 <= y and y + h <= 486
        assert x % 2 == 0 and w % 2 == 0


TESTS = [
    test_constants_exist,
    test_rgb_norm_to_yuv10_black,
    test_rgb_norm_to_yuv10_white,
    test_rgb_norm_to_yuv10_75_yellow,
    test_yuv10_to_rgb8_round_trip,
    test_tartan_regions_count_and_ids,
    test_tartan_bottom_row_75_magenta_via_rec601,
    test_tartan_red2_equals_top_red,
    test_tartan_region_record_shape,
    test_tartan_75_yellow_expected,
    test_gray_regions,
    test_grid_landmarks,
    test_grid_landmark_distribution,
    test_ideal_picture_box,
    test_boundary_triangles_catalog,
    test_boundary_triangles_back_corner_spacing,
    test_registration_cross_catalog,
    test_black_circle_catalog,
    test_burst_regions_catalog,
    test_artifact_regions_catalog,
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
