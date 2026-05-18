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


def _make_capture_json_with_geometry(tag):
    cap = _make_capture_json(tag)
    cap["geometry"] = {
        "fiducials": {
            "triangles": {
                "TL": {"back_corner_1": [50, 30], "back_corner_2": [70, 30],
                       "back_midpoint": [60, 30], "apex_inferred": [60, 3],
                       "apex_detected": [60, 3], "confidence": 0.95,
                       "failure_reason": None},
                "TR": {"back_corner_1": [650, 30], "back_corner_2": [670, 30],
                       "back_midpoint": [660, 30], "apex_inferred": [660, 3],
                       "apex_detected": [660, 3], "confidence": 0.94,
                       "failure_reason": None},
                "BL": {"back_corner_1": [50, 456], "back_corner_2": [70, 456],
                       "back_midpoint": [60, 456], "apex_inferred": [60, 483],
                       "apex_detected": [60, 483], "confidence": 0.93,
                       "failure_reason": None},
                "BR": {"back_corner_1": [650, 456], "back_corner_2": [670, 456],
                       "back_midpoint": [660, 456], "apex_inferred": [660, 483],
                       "apex_detected": [660, 483], "confidence": 0.94,
                       "failure_reason": None},
            },
            "cross": {"center": [360.0, 243.0],
                      "h_arm_len_px": 17.0, "v_arm_len_px": 17.0,
                      "confidence": 0.98},
            "circle": {"center": [360.0, 243.0], "rx": 242.8, "ry": 243.1,
                       "rotation_deg": 0.0, "fit_rms": 0.41,
                       "confidence": 0.97},
        },
        "derived": {
            "active_picture_box": {"top": 3, "left": 60, "bottom": 483, "right": 660},
            "picture_extent_px": {"width": 600, "height": 480},
            "picture_offset_from_ideal": {"dx": 60, "dy": 3},
            "corner_skew_px": {"top_vs_bottom_width_diff": 0.0,
                                "left_vs_right_height_diff": 0.0},
            "arrow_tip_coords": {"TL": [60, 3], "TR": [660, 3],
                                  "BL": [60, 483], "BR": [660, 483]},
            "clip_detected": {
                "TL": {"apex_visible": True, "clip_px": 0,
                       "interpretation": "no clip detected"},
                "TR": {"apex_visible": True, "clip_px": 0,
                       "interpretation": "no clip detected"},
                "BL": {"apex_visible": True, "clip_px": 0,
                       "interpretation": "no clip detected"},
                "BR": {"apex_visible": True, "clip_px": 0,
                       "interpretation": "no clip detected"},
            },
            "cross_offset_from_ideal": [0.0, 0.0],
            "aperture_symmetry": 1.0,
            "aspect_ratio_check": 0.998,
            "diameter_vs_picture_height": 1.012,
            "circle_fit_rms": 0.41,
            "summary": {
                "arrow_spacings_px": {
                    "top":    {"actual": 600.0, "ideal": 357.0, "delta": 243.0},
                    "bottom": {"actual": 600.0, "ideal": 357.0, "delta": 243.0},
                    "left":   {"actual": 480.0, "ideal": 483.0, "delta": -3.0},
                    "right":  {"actual": 480.0, "ideal": 483.0, "delta": -3.0},
                },
                "picture_center_offset_px": {"dx": 0.0, "dy": 0.0},
                "picture_scale_pct": {"horizontal": 100.0, "vertical": 99.4},
                "keystone_px": {
                    "horizontal_top_minus_bottom": 0.0,
                    "vertical_left_minus_right":   0.0,
                },
                "circle": {
                    "horizontal_diameter_px": 532.0,
                    "vertical_diameter_px":   484.0,
                    "expected_h_over_v_for_round": 1.1,
                    "actual_h_over_v":             1.099,
                    "displayed_circularity":       0.999,
                    "rotation_deg":                0.0,
                },
            },
        },
        "registration_refit": {
            "inlier_count_initial": 11,
            "inlier_count_final":   20,
            "final_residuals_px": {"mean": 0.21, "max": 0.55},
            "anchors_added": ["TL.bc1", "TL.bc2", "TR.bc1", "TR.bc2",
                              "BL.bc1", "BL.bc2", "BR.bc1", "BR.bc2", "RC"],
        },
        "quality_flag":  "ok",
        "quality_reason": None,
    }
    return cap


def test_render_geometry_section_includes_picture_box_and_clip_status():
    a = _make_capture_json_with_geometry("alpha")
    html = tp_compare.render_geometry_section([a])
    assert "Geometry" in html
    # Arrowhead spacings table heading
    assert "Arrowhead spacing" in html
    # Picture displacement summary
    assert "Center shifted" in html
    assert "Horizontal scale" in html and "Keystone" in html
    # Registration cross (aperture symmetry)
    assert "aperture" in html
    # PAR-aware circle
    assert "Circle" in html and "Displayed circularity" in html
    # Clip status chips
    assert "no clip detected" in html or "TL: visible" in html
    # Refit benefit shows the inlier counts
    assert "11" in html  # initial
    assert "20" in html  # final


def test_render_geometry_section_shows_clip_when_apex_invisible():
    a = _make_capture_json_with_geometry("alpha")
    a["geometry"]["derived"]["clip_detected"]["TL"] = {
        "apex_visible": False, "clip_px": 3,
        "interpretation": "top edge clipped ~3 px",
    }
    a["geometry"]["derived"]["clip_detected"]["TR"] = {
        "apex_visible": False, "clip_px": 3,
        "interpretation": "top edge clipped ~3 px",
    }
    html = tp_compare.render_geometry_section([a])
    assert "top edge clipped" in html


def test_render_page_includes_geometry_section():
    a = _make_capture_json_with_geometry("alpha")
    b = _make_capture_json_with_geometry("beta")
    html = tp_compare.render_page([a, b])
    assert "Geometry" in html
    # Geometry section appears BEFORE tartan deltas in the page order.
    g_idx = html.index("Geometry")
    t_idx = html.index("Tartan Deltas")
    assert g_idx < t_idx


def test_render_geometry_section_handles_missing_block():
    """Older JSONs without a geometry block must render a placeholder."""
    a = _make_capture_json("alpha")
    # Note: _make_capture_json from the existing test file does NOT add a
    # geometry block, so this exercises the fallback path.
    html = tp_compare.render_geometry_section([a])
    assert "older JSON" in html or "no geometry" in html


def test_geometry_overview_table_contains_all_captures_and_is_sortable():
    a = _make_capture_json_with_geometry("alpha")
    b = _make_capture_json_with_geometry("beta")
    html = tp_compare.render_geometry_overview([a, b])
    assert "Geometry Overview" in html
    assert "overview-table" in html
    # Each capture's basename appears in a row
    assert "alpha.mov" in html
    assert "beta.mov" in html
    # Sortable headers carry the data-sort attribute the JS reads
    assert 'data-sort="num"' in html
    assert 'data-sort="text"' in html


def test_tartan_overview_includes_mean_de_and_worst_color():
    a = _make_capture_json_with_geometry("alpha")
    html = tp_compare.render_tartan_overview([a])
    assert "Color (Tartan) Overview" in html
    assert "Mean ΔE" in html
    assert "Worst color" in html
    # The single tartan patch in the fixture is YEL → it should be reported as worst
    assert "YEL" in html


def test_grayscale_overview_lists_black_floor_and_white_ceiling():
    a = _make_capture_json_with_geometry("alpha")
    # The fixture only has G1; build a richer one with G1–G4 so the
    # overview can compute the white-ceiling delta.
    a["grays"] = [
        {"id": "G1", "name": "gray_20", "ideal_y10": 239.2,
         "measured_y10": 248.0, "delta_y10": 8.8,
         "u10": 512, "v10": 512, "patch_size_px": [3, 3]},
        {"id": "G2", "name": "gray_40", "ideal_y10": 414.4,
         "measured_y10": 413.0, "delta_y10": -1.4,
         "u10": 512, "v10": 512, "patch_size_px": [3, 3]},
        {"id": "G3", "name": "gray_60", "ideal_y10": 589.6,
         "measured_y10": 591.0, "delta_y10": 1.4,
         "u10": 512, "v10": 512, "patch_size_px": [3, 3]},
        {"id": "G4", "name": "gray_80", "ideal_y10": 764.8,
         "measured_y10": 756.0, "delta_y10": -8.8,
         "u10": 514, "v10": 510, "patch_size_px": [3, 3]},
    ]
    html = tp_compare.render_grayscale_overview([a])
    assert "Grayscale Overview" in html
    assert "Black floor" in html and "White ceiling" in html
    # The G1 delta (+8.8) and G4 delta (-8.8) should both appear
    assert "+8.8" in html
    assert "-8.8" in html


def test_render_tartan_panel_shows_per_color_swatches_and_verdict():
    a = _make_capture_json_with_geometry("alpha")
    html = tp_compare.render_tartan_panels([a])
    assert "alpha.mov" in html
    # Per-color row pieces
    assert "swatch-inline" in html
    assert "ΔE" in html
    # Plain-language verdict appears
    assert "match" in html or "close" in html or "off" in html


def test_render_gray_panel_shows_per_step_swatches():
    a = _make_capture_json_with_geometry("alpha")
    html = tp_compare.render_gray_panels([a])
    assert "alpha.mov" in html
    assert "swatch-inline" in html
    # The fixture has G1; its label should appear
    assert "20% IRE" in html


def test_render_page_places_registration_summary_in_appendix():
    a = _make_capture_json_with_geometry("alpha")
    b = _make_capture_json_with_geometry("beta")
    html = tp_compare.render_page([a, b])
    # Appendix container exists and wraps the registration summary
    app_idx = html.index("Technical Appendix")
    reg_idx = html.index("Registration Summary")
    geo_overview_idx = html.index("Geometry Overview")
    # Overviews come before the appendix
    assert geo_overview_idx < app_idx
    # Registration Summary is inside the appendix region
    assert app_idx < reg_idx


TESTS_NO_TMPDIR = [
    test_render_registration_summary_contains_per_capture_data,
    test_render_tartan_deltas_contains_swatches_and_deltas,
    test_render_gray_deltas_contains_table_and_chart_data,
    test_render_luma_scale_analysis_reports_gain_and_pedestal_residuals,
    test_render_geometry_section_includes_picture_box_and_clip_status,
    test_render_geometry_section_shows_clip_when_apex_invisible,
    test_render_page_includes_geometry_section,
    test_render_geometry_section_handles_missing_block,
    test_geometry_overview_table_contains_all_captures_and_is_sortable,
    test_tartan_overview_includes_mean_de_and_worst_color,
    test_grayscale_overview_lists_black_floor_and_white_ceiling,
    test_render_tartan_panel_shows_per_color_swatches_and_verdict,
    test_render_gray_panel_shows_per_step_swatches,
    test_render_page_places_registration_summary_in_appendix,
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
