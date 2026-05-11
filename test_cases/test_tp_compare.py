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


def test_render_tartan_deltas_contains_swatches_and_deltas():
    a = _make_capture_json("alpha")
    b = _make_capture_json("beta")
    # Mutate one delta in beta to verify it gets rendered.
    b["tartan"][0]["delta_yuv10"] = [-50.0, 5.0, -2.0]
    b["tartan"][0]["measured_yuv10"] = [596.0, 181.0, 565.0]
    html = tp_compare.render_tartan_deltas([a, b])
    assert "YEL" in html
    assert "-50" in html or "-50.0" in html
    # Inline swatch background-color
    assert "background-color: rgb(" in html or "background:rgb(" in html
    # Per-swatch role labels and CSS classes
    assert "swatch-ideal" in html
    assert "swatch-measured" in html
    assert ">ref<" in html
    assert ">cap<" in html


def test_render_luma_scale_analysis_reports_gain_and_pedestal_residuals():
    # Capture A: nearly-perfect (linear residual near zero, gain ~ 1.0)
    a = _make_capture_json("alpha")
    a["grays"] = [
        {"id": "G1", "ideal_y10": 239.2, "measured_y10": 239.0, "delta_y10": -0.2,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_20"},
        {"id": "G2", "ideal_y10": 414.4, "measured_y10": 414.0, "delta_y10": -0.4,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_40"},
        {"id": "G3", "ideal_y10": 589.6, "measured_y10": 589.2, "delta_y10": -0.4,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_60"},
        {"id": "G4", "ideal_y10": 764.8, "measured_y10": 764.0, "delta_y10": -0.8,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_80"},
    ]
    # Capture B: ~9% luma compression (snellld-like signature)
    b = _make_capture_json("beta")
    b["grays"] = [
        {"id": "G1", "ideal_y10": 239.2, "measured_y10": 220.0, "delta_y10": -19.2,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_20"},
        {"id": "G2", "ideal_y10": 414.4, "measured_y10": 379.0, "delta_y10": -35.4,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_40"},
        {"id": "G3", "ideal_y10": 589.6, "measured_y10": 538.0, "delta_y10": -51.6,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_60"},
        {"id": "G4", "ideal_y10": 764.8, "measured_y10": 698.0, "delta_y10": -66.8,
         "u10": 512.0, "v10": 512.0, "patch_size_px": [6, 6], "name": "gray_step_80"},
    ]
    html = tp_compare.render_luma_scale_analysis([a, b])
    assert "Luma Scale Analysis" in html
    assert "Slope" in html and "Intercept" in html
    assert "alpha.mov" in html and "beta.mov" in html
    # Verdict text differs for the two captures
    assert "essentially correct" in html
    assert "luma gain" in html
    # Pedestal hypothesis residuals are reported
    assert "Pedestal" in html


def test_render_gray_deltas_contains_table_and_chart_data():
    a = _make_capture_json("alpha")
    a["grays"] = [
        {"id": "G1", "name": "gray_step_20", "ideal_y10": 239.2,
         "measured_y10": 234.0, "delta_y10": -5.2, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
        {"id": "G2", "name": "gray_step_40", "ideal_y10": 414.4,
         "measured_y10": 406.0, "delta_y10": -8.4, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
        {"id": "G3", "name": "gray_step_60", "ideal_y10": 589.6,
         "measured_y10": 579.0, "delta_y10": -10.6, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
        {"id": "G4", "name": "gray_step_80", "ideal_y10": 764.8,
         "measured_y10": 752.0, "delta_y10": -12.8, "u10": 512.0, "v10": 512.0,
         "patch_size_px": [6, 6]},
    ]
    html = tp_compare.render_gray_deltas([a])
    assert "G1" in html and "G4" in html
    assert "-5.2" in html or "-5.20" in html
    assert "-12.8" in html or "-12.80" in html
    # Per-cell ref/cap swatches (same convention as the tartan section)
    assert "swatch-ideal" in html
    assert "swatch-measured" in html
    assert ">ref<" in html
    assert ">cap<" in html
    # Chart.js data block
    assert "Chart" in html
    assert "239.2" in html  # ideal Y10 for G1


def test_compare_cli_writes_html(tmp_dir):
    import subprocess
    a = _make_capture_json("alpha")
    b = _make_capture_json("beta")
    a_path = os.path.join(tmp_dir, "a.json")
    b_path = os.path.join(tmp_dir, "b.json")
    out_path = os.path.join(tmp_dir, "report.html")
    with open(a_path, "w") as f:
        json.dump(a, f)
    with open(b_path, "w") as f:
        json.dump(b, f)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        ["python", "tp_compare.py", a_path, b_path, "--output", out_path],
        check=True, cwd=project_root,
    )
    with open(out_path) as f:
        html = f.read()
    assert "<html" in html
    assert "Registration Summary" in html
    assert "Tartan Deltas" in html
    assert "Gray Step Deltas" in html
    assert "Chart.js" in html or "chart.js" in html  # CDN script reference


TESTS_NO_TMPDIR = [
    test_render_registration_summary_contains_per_capture_data,
    test_render_tartan_deltas_contains_swatches_and_deltas,
    test_render_gray_deltas_contains_table_and_chart_data,
    test_render_luma_scale_analysis_reports_gain_and_pedestal_residuals,
]
TESTS_TMPDIR = [test_compare_cli_writes_html]


def main():
    failed = 0
    for t in TESTS_NO_TMPDIR:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    tmp = tempfile.mkdtemp(prefix="tp_compare_test_")
    try:
        for t in TESTS_TMPDIR:
            try:
                t(tmp); print(f"PASS  {t.__name__}")
            except Exception as e:
                failed += 1; print(f"FAIL  {t.__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(TESTS_NO_TMPDIR) + len(TESTS_TMPDIR)
    if failed:
        print(f"\n{failed}/{total} tests failed"); sys.exit(1)
    print(f"\nAll {total} tests passed")


if __name__ == "__main__":
    main()
