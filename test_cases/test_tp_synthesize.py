#!/usr/bin/env python3
"""Tests for tp_synthesize."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize


def test_synthesize_shapes():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    assert Y.shape == (486, 720)
    assert U.shape == (486, 360)
    assert V.shape == (486, 360)
    assert Y.dtype == np.uint16


def test_synthesize_grey_background_dominates():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Most pixels (>= 50%) should be grey-background; we overlay tartan,
    # gray strip, and grid lines on top.
    grey_count = int((Y == tp_chart.GREY_BACKGROUND_Y10).sum())
    assert grey_count > Y.size * 0.5, (
        f"only {grey_count}/{Y.size} pixels at grey level"
    )


def test_synthesize_chroma_centred_off_color_regions():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # In the lower-right quadrant (no tartan/gray/triangle in stage 1),
    # chroma should be the centre value 512 everywhere except where grid
    # lines fall.
    u_mid = U[300:400, 250:300]
    v_mid = V[300:400, 250:300]
    assert (u_mid == tp_chart.CHROMA_CENTER).all()
    assert (v_mid == tp_chart.CHROMA_CENTER).all()


def test_synthesize_grid_intersections_dark():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for lm in tp_chart.GRID_LANDMARKS:
        x, y = lm["ideal_x"], lm["ideal_y"]
        # The 3x3 patch around a landmark should contain dark pixels
        patch = Y[y - 1:y + 2, x - 1:x + 2]
        assert int(patch.min()) < 100, (
            f"landmark {lm['id']} at ({x},{y}): no dark pixel "
            f"(min Y10={int(patch.min())})"
        )


TESTS = [
    test_synthesize_shapes,
    test_synthesize_grey_background_dominates,
    test_synthesize_chroma_centred_off_color_regions,
    test_synthesize_grid_intersections_dark,
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
