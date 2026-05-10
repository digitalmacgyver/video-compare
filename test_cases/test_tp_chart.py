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


TESTS = [
    test_constants_exist,
    test_rgb_norm_to_yuv10_black,
    test_rgb_norm_to_yuv10_white,
    test_rgb_norm_to_yuv10_75_yellow,
    test_yuv10_to_rgb8_round_trip,
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
