#!/usr/bin/env python3
"""Tests for tp_compare HTML rendering."""

import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tp_compare


def _make_capture_json(tag, residual_mean=0.4, quality="ok"):
    return {
        "_meta": {
            "capture": f"/path/{tag}.mov",
            "frame_index": 60,
            "raster_in": [720, 480],
            "raster_processed": [720, 486],
            "padding_offsets": {"top": 3, "bottom": 3, "left": 0, "right": 0},
            "field_order": "tb",
            "progressive_warning": False,
            "decode": {"deinterlace": "none-weave-only", "pix_fmt": "yuv422p10le"},
            "registration": {
                "affine": [[1.001, 0.0, 2.5], [0.0, 1.0, -1.0]],
                "residuals_px": {"mean": residual_mean, "max": residual_mean * 2},
                "inliers": 7, "total": 8,
                "landmarks_used": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
                "quality_flag": quality,
                "quality_reason": None,
            },
            "tool_version": "tp_measure 0.1",
            "tp_chart_version": 1,
            "ideal_frame_md5": "abc123",
        },
        "tartan": [
            {"id": "YEL", "name": "yellow_75", "ideal_yuv10": [646, 176, 567],
             "measured_yuv10": [640, 173, 569], "delta_yuv10": [-6, -3, 2],
             "sat_pct_vs_ideal": 76.5, "patch_size_px": [3, 3],
             "patch_center_capture_xy": [15, 13]},
        ],
        "grays": [
            {"id": "G1", "name": "gray_step_20", "ideal_y10": 239.2,
             "measured_y10": 234.0, "delta_y10": -5.2,
             "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6]},
        ],
    }


def test_render_registration_summary_contains_per_capture_data():
    a = _make_capture_json("alpha", residual_mean=0.4, quality="ok")
    b = _make_capture_json("beta", residual_mean=2.5, quality="warn")
    html = tp_compare.render_registration_summary([a, b])
    assert "alpha.mov" in html
    assert "beta.mov" in html
    assert "0.40" in html  # residual mean
    assert "warn" in html


TESTS = [
    test_render_registration_summary_contains_per_capture_data,
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
