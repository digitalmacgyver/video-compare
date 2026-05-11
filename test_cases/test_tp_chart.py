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
    assert len(lms) == 12, f"expected 12 anchors, got {len(lms)}"
    ids = [lm["id"] for lm in lms]
    assert ids == [f"L{i+1}" for i in range(12)], ids
    for lm in lms:
        assert lm["kind"] == "grid_intersection"
        assert 0 < lm["ideal_x"] < 720
        assert 0 < lm["ideal_y"] < 486
        assert lm["search_window_px"] >= 16


def test_grid_landmark_distribution():
    # All anchors lie in the safe interior y ∈ [108, 270] (avoiding the busy
    # top of the chart and the lower-third uncertainty around the old L8).
    for lm in tp_chart.GRID_LANDMARKS:
        assert 108 <= lm["ideal_y"] <= 270, lm
        assert 180 <= lm["ideal_x"] <= 600, lm
    # 3 anchors per y-row for x-spread, 4 distinct y-rows.
    ys = sorted({lm["ideal_y"] for lm in tp_chart.GRID_LANDMARKS})
    assert ys == [108, 162, 216, 270]
    # Verify each y-row has exactly 3 anchors (balanced x-spread per row).
    import collections
    row_counts = collections.Counter(lm["ideal_y"] for lm in tp_chart.GRID_LANDMARKS)
    assert all(count == 3 for count in row_counts.values()), dict(row_counts)
    # x-values must avoid the burst-column region; exact set is intentional.
    xs = sorted({lm["ideal_x"] for lm in tp_chart.GRID_LANDMARKS})
    assert xs == [180, 360, 420, 540, 600], xs


def test_ideal_picture_box():
    box = tp_chart.IDEAL_PICTURE_BOX
    assert box == {"left": 0, "top": 0, "right": 719, "bottom": 485}
    corners = tp_chart.IDEAL_PICTURE_BOX_CORNERS
    # TL, TR, BL, BR
    assert corners == [(0, 0), (719, 0), (0, 485), (719, 485)]


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
