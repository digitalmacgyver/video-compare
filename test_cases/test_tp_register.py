#!/usr/bin/env python3
"""Tests for tp_register: landmark detection + affine fit."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_register


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


TESTS = [
    test_detect_landmark_on_synthesized_ideal,
    test_detect_landmark_off_grid_returns_none,
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
