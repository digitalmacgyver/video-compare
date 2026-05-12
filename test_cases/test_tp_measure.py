#!/usr/bin/env python3
"""Tests for tp_measure: frame extraction, padding, sampling, JSON."""

import sys, os, json, tempfile, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_measure


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_synthesized_prores(path, width, height, frames=10):
    """Encode a synthesized SW2 frame as ProRes 422 HQ at the given raster.

    The synthesizer produces a single frame; we write `frames` copies so the
    extractor can pick frame 0 (or any small index).
    """
    Y, U, V = tp_synthesize.synthesize(width, height)
    raw = (Y.astype("<u2").tobytes()
           + U.astype("<u2").tobytes()
           + V.astype("<u2").tobytes())
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "-s", f"{width}x{height}",
        "-r", "30000/1001",
        "-i", "pipe:0",
        "-frames:v", str(frames),
        "-c:v", "prores_ks", "-profile:v", "3",
        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for _ in range(frames):
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0


def test_extract_frame_yuv422p10le_720x486(tmp_dir):
    path = os.path.join(tmp_dir, "ideal.mov")
    _write_synthesized_prores(path, 720, 486)
    Y, U, V, meta = tp_measure.extract_frame(path, frame_index=0)
    assert Y.shape == (486, 720)
    assert U.shape == (486, 360)
    assert V.shape == (486, 360)
    assert meta["raster_in"] == [720, 486]
    # Center pixel of YEL tartan box should be at the expected Y10.
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    x, y, w, h = yel["ideal_box"]
    cx, cy = x + w // 2, y + h // 2
    assert abs(int(Y[cy, cx]) - yel["expected"]["y10"]) < 5


def test_pad_to_486_centers_grey():
    Y_in = np.full((480, 720), 200, dtype=np.uint16)
    U_in = np.full((480, 360), 512, dtype=np.uint16)
    V_in = np.full((480, 360), 512, dtype=np.uint16)
    Y, U, V, offsets = tp_measure.pad_to_486(Y_in, U_in, V_in)
    assert Y.shape == (486, 720)
    assert offsets == {"top": 3, "bottom": 3, "left": 0, "right": 0}
    # Top 3 rows are grey-background; rows 3..483 are 200; bottom 3 are grey.
    assert (Y[:3, :] == tp_chart.GREY_BACKGROUND_Y10).all()
    assert (Y[3:483, :] == 200).all()
    assert (Y[483:, :] == tp_chart.GREY_BACKGROUND_Y10).all()


def test_extract_pad_720x480_dvd_like(tmp_dir):
    path = os.path.join(tmp_dir, "dvd.mov")
    Y, U, V = tp_synthesize.synthesize(720, 486)
    Y480 = Y[3:483, :]
    U480 = U[3:483, :]
    V480 = V[3:483, :]
    raw = (Y480.astype("<u2").tobytes()
           + U480.astype("<u2").tobytes()
           + V480.astype("<u2").tobytes())
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "-s", "720x480", "-r", "30000/1001",
        "-i", "pipe:0", "-frames:v", "5",
        "-c:v", "prores_ks", "-profile:v", "3",
        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for _ in range(5):
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()
    Y, U, V, meta = tp_measure.extract_frame(path, frame_index=0)
    Y_padded, U_padded, V_padded, offsets = tp_measure.pad_to_486(Y, U, V)
    assert Y_padded.shape == (486, 720)
    assert offsets["top"] == 3 and offsets["bottom"] == 3


def test_sample_region_on_synthesized_identity():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    measurement = tp_measure.sample_region(Y, U, V, yel, identity)
    assert abs(measurement["measured_yuv10"][0] - yel["expected"]["y10"]) < 1.0
    assert abs(measurement["measured_yuv10"][1] - yel["expected"]["u10"]) < 1.0
    assert abs(measurement["measured_yuv10"][2] - yel["expected"]["v10"]) < 1.0
    assert abs(measurement["delta_yuv10"][0]) < 1.0
    assert abs(measurement["delta_yuv10"][1]) < 1.0
    assert abs(measurement["delta_yuv10"][2]) < 1.0


def test_sample_region_with_translation():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Translate ideal -> capture by (5, 7): a YEL pixel at ideal (cx,cy) is
    # actually at (cx+5, cy+7) in the capture. So affine = [[1,0,5],[0,1,7]].
    M = np.array([[1, 0, 5], [0, 1, 7]], dtype=np.float32)
    # Roll the synthesized frame by (5, 7) to simulate a shifted capture.
    Y_shifted = np.roll(np.roll(Y, 5, axis=1), 7, axis=0)
    U_shifted = np.roll(np.roll(U, 5 // 2, axis=1), 7, axis=0)
    V_shifted = np.roll(np.roll(V, 5 // 2, axis=1), 7, axis=0)
    yel = next(r for r in tp_chart.TARTAN_REGIONS if r["id"] == "YEL")
    measurement = tp_measure.sample_region(Y_shifted, U_shifted, V_shifted, yel, M)
    assert abs(measurement["delta_yuv10"][0]) < 5.0


def test_pad_to_486_raises_on_oversized_height():
    Y = np.zeros((576, 720), dtype=np.uint16)
    U = np.zeros((576, 360), dtype=np.uint16)
    V = np.zeros((576, 360), dtype=np.uint16)
    raised = False
    try:
        tp_measure.pad_to_486(Y, U, V)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for height > 486"


def test_measure_end_to_end_zero_deltas(tmp_dir):
    capture_path = os.path.join(tmp_dir, "ideal.mov")
    json_path = os.path.join(tmp_dir, "out.json")
    _write_synthesized_prores(capture_path, 720, 486, frames=5)
    cmd = [
        "python", "tp_measure.py", capture_path,
        "--frame", "0",
        "--output", json_path,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    with open(json_path) as f:
        data = json.load(f)
    assert data["_meta"]["raster_in"] == [720, 486]
    assert data["_meta"]["registration"]["quality_flag"] == "ok"
    assert len(data["tartan"]) == 8
    assert len(data["grays"]) == 4
    # On the synthesized ideal, all deltas should be tiny.
    for patch in data["tartan"]:
        assert abs(patch["delta_yuv10"][0]) < 2.0, patch
    # Gray-delta tolerance widened to 2.0 to absorb ProRes 422 HQ round-trip
    # quantization (the synthesized frame goes through encode -> decode before
    # being measured). Pure ideal-frame sampling (test_sample_region_on_synthesized_identity)
    # stays sub-pixel.
    for g in data["grays"]:
        assert abs(g["delta_y10"]) < 2.0, g
    # The luma_scale block should be present with sensible identity-ish values.
    assert "luma_scale" in data and data["luma_scale"] is not None
    ls = data["luma_scale"]
    for key in ("slope", "intercept", "gain_pct_loss",
                "predicted_y10_at_black", "predicted_y10_at_white",
                "rms_linear", "rms_pedestal_a", "rms_pedestal_b"):
        assert key in ls, key
    assert abs(ls["slope"] - 1.0) < 0.01
    assert abs(ls["intercept"]) < 2.0
    assert ls["rms_linear"] < 2.0


def test_fit_gray_ramp_unit_pure_gain_signature():
    # 9% pure luma gain reduction (snellld-like signature).
    grays = [
        {"id": "G1", "ideal_y10": 239.2, "measured_y10": 64 + 0.91 * (239.2 - 64)},
        {"id": "G2", "ideal_y10": 414.4, "measured_y10": 64 + 0.91 * (414.4 - 64)},
        {"id": "G3", "ideal_y10": 589.6, "measured_y10": 64 + 0.91 * (589.6 - 64)},
        {"id": "G4", "ideal_y10": 764.8, "measured_y10": 64 + 0.91 * (764.8 - 64)},
    ]
    fit = tp_measure.fit_gray_ramp(grays)
    assert fit is not None
    assert abs(fit["slope"] - 0.91) < 0.001
    assert fit["rms_linear"] < 0.5
    # Pedestal hypotheses should fit much worse than the linear model.
    assert fit["rms_pedestal_a"] > 10 * fit["rms_linear"]
    assert fit["rms_pedestal_b"] > 10 * fit["rms_linear"]


def test_fit_gray_ramp_returns_none_on_malformed_input():
    assert tp_measure.fit_gray_ramp([]) is None
    assert tp_measure.fit_gray_ramp([{"id": "G1", "ideal_y10": 100, "measured_y10": 100}]) is None


def test_measure_end_to_end_dvd_padding(tmp_dir):
    capture_path = os.path.join(tmp_dir, "dvd.mov")
    json_path = os.path.join(tmp_dir, "dvd.json")
    Y, U, V = tp_synthesize.synthesize(720, 486)
    Y480, U480, V480 = Y[3:483, :], U[3:483, :], V[3:483, :]
    raw = (Y480.astype("<u2").tobytes()
           + U480.astype("<u2").tobytes()
           + V480.astype("<u2").tobytes())
    enc_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "-s", "720x480", "-r", "30000/1001",
        "-i", "pipe:0", "-frames:v", "5",
        "-c:v", "prores_ks", "-profile:v", "3",
        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
        capture_path,
    ]
    proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    for _ in range(5):
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0
    cmd = [
        "python", "tp_measure.py", capture_path,
        "--frame", "0", "--output", json_path,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    with open(json_path) as f:
        data = json.load(f)
    assert data["_meta"]["raster_in"] == [720, 480]
    assert data["_meta"]["padding_offsets"] == {"top": 3, "bottom": 3, "left": 0, "right": 0}
    assert data["_meta"]["registration"]["quality_flag"] in ("ok", "warn")


def test_measure_end_to_end_includes_geometry_block(tmp_dir):
    capture_path = os.path.join(tmp_dir, "ideal.mov")
    json_path = os.path.join(tmp_dir, "out.json")
    _write_synthesized_prores(capture_path, 720, 486, frames=5)
    subprocess.run(
        ["python", "tp_measure.py", capture_path,
         "--frame", "0", "--output", json_path],
        check=True, cwd=PROJECT_ROOT,
    )
    with open(json_path) as f:
        data = json.load(f)
    assert "geometry" in data
    g = data["geometry"]
    assert "fiducials" in g
    assert "derived" in g
    assert set(g["fiducials"]["triangles"].keys()) == {"TL", "TR", "BL", "BR"}
    assert g["fiducials"]["cross"] is not None
    assert g["fiducials"]["circle"] is not None
    assert g["derived"]["active_picture_box"] is not None
    # On the synthesized ideal, no clipping detected.
    for tid in ("TL", "TR", "BL", "BR"):
        assert g["derived"]["clip_detected"][tid]["apex_visible"] is True
    # Quality flag present.
    assert g["quality_flag"] in ("ok", "warn", "partial", "failed")
    # Refit benefit reporting.
    rrf = g["registration_refit"]
    assert "inlier_count_initial" in rrf
    assert "inlier_count_final" in rrf
    assert rrf["inlier_count_final"] >= rrf["inlier_count_initial"]


def _make_geom(n_tris=4, cross=True, circle_rms=None):
    tris = {tid: ({"dummy": True} if i < n_tris else None)
            for i, tid in enumerate(("TL", "TR", "BL", "BR"))}
    rc = {"dummy": True} if cross else None
    bc = None if circle_rms is None else {"fit_rms": circle_rms}
    return {"fiducials": {"triangles": tris, "cross": rc, "circle": bc}}


def test_geometry_quality_ok_with_4_tris_cross_no_circle():
    geom = _make_geom(n_tris=4, cross=True, circle_rms=None)
    final = {"residuals_px": {"mean": 0.4}}
    flag, _ = tp_measure._compute_geometry_quality(geom, final)
    assert flag == "ok"


def test_geometry_quality_ok_when_circle_fits_poorly():
    # PAR-elliptical ring: circle_fit_rms is inflated, but triangles+cross
    # are clean and residuals are tight -> quality stays "ok".
    geom = _make_geom(n_tris=4, cross=True, circle_rms=4.5)
    final = {"residuals_px": {"mean": 0.4}}
    flag, _ = tp_measure._compute_geometry_quality(geom, final)
    assert flag == "ok"


def test_geometry_quality_warn_when_residual_above_1px():
    geom = _make_geom(n_tris=4, cross=True, circle_rms=None)
    final = {"residuals_px": {"mean": 1.3}}
    flag, _ = tp_measure._compute_geometry_quality(geom, final)
    assert flag == "warn"


def test_geometry_quality_failed_when_residual_above_2px():
    geom = _make_geom(n_tris=4, cross=True, circle_rms=None)
    final = {"residuals_px": {"mean": 2.5}}
    flag, _ = tp_measure._compute_geometry_quality(geom, final)
    assert flag == "failed"


def test_geometry_quality_partial_when_cross_missing():
    geom = _make_geom(n_tris=4, cross=False, circle_rms=1.0)
    final = {"residuals_px": {"mean": 0.4}}
    flag, _ = tp_measure._compute_geometry_quality(geom, final)
    assert flag == "partial"


def test_geometry_quality_partial_when_only_2_triangles():
    geom = _make_geom(n_tris=2, cross=True, circle_rms=1.0)
    final = {"residuals_px": {"mean": 0.4}}
    flag, _ = tp_measure._compute_geometry_quality(geom, final)
    assert flag == "partial"


def test_geometry_quality_failed_when_cross_and_tris_all_missing():
    geom = _make_geom(n_tris=1, cross=False, circle_rms=None)
    final = {"residuals_px": {"mean": 0.4}}
    flag, _ = tp_measure._compute_geometry_quality(geom, final)
    assert flag == "failed"


def test_measure_end_to_end_includes_artifacts(tmp_dir):
    capture_path = os.path.join(tmp_dir, "ideal.mov")
    json_path = os.path.join(tmp_dir, "out_art.json")
    _write_synthesized_prores(capture_path, 720, 486, frames=5)
    subprocess.run(
        ["python", "tp_measure.py", capture_path,
         "--frame", "0", "--output", json_path],
        check=True, cwd=PROJECT_ROOT,
    )
    with open(json_path) as f:
        data = json.load(f)
    assert "artifacts" in data
    art = data["artifacts"]
    assert art is not None
    assert "regions" in art
    assert "summary" in art
    assert "ZP_CHROMA_LEAK" in art["regions"]
    assert art["summary"]["zone_plate_chroma_present"] is False


def test_measure_end_to_end_includes_frequency_response(tmp_dir):
    capture_path = os.path.join(tmp_dir, "ideal.mov")
    json_path = os.path.join(tmp_dir, "out_freq.json")
    _write_synthesized_prores(capture_path, 720, 486, frames=5)
    subprocess.run(
        ["python", "tp_measure.py", capture_path,
         "--frame", "0", "--output", json_path],
        check=True, cwd=PROJECT_ROOT,
    )
    with open(json_path) as f:
        data = json.load(f)
    assert "frequency_response" in data
    fr = data["frequency_response"]
    assert fr is not None
    assert "regions" in fr
    assert "summary" in fr
    assert "BURST_3p58" in fr["regions"]
    assert fr["regions"]["BURST_3p58"]["modulation_pct"] > 30.0


TESTS_TMPDIR = [
    test_extract_frame_yuv422p10le_720x486,
    test_extract_pad_720x480_dvd_like,
    test_measure_end_to_end_zero_deltas,
    test_measure_end_to_end_dvd_padding,
    test_measure_end_to_end_includes_geometry_block,
    test_measure_end_to_end_includes_frequency_response,
    test_measure_end_to_end_includes_artifacts,
]
TESTS_NO_TMPDIR = [
    test_pad_to_486_centers_grey,
    test_sample_region_on_synthesized_identity,
    test_sample_region_with_translation,
    test_pad_to_486_raises_on_oversized_height,
    test_fit_gray_ramp_unit_pure_gain_signature,
    test_fit_gray_ramp_returns_none_on_malformed_input,
    test_geometry_quality_ok_with_4_tris_cross_no_circle,
    test_geometry_quality_ok_when_circle_fits_poorly,
    test_geometry_quality_warn_when_residual_above_1px,
    test_geometry_quality_failed_when_residual_above_2px,
    test_geometry_quality_partial_when_cross_missing,
    test_geometry_quality_partial_when_only_2_triangles,
    test_geometry_quality_failed_when_cross_and_tris_all_missing,
]


def main():
    failed = 0
    for t in TESTS_NO_TMPDIR:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    tmp = tempfile.mkdtemp(prefix="tp_measure_test_")
    try:
        for t in TESTS_TMPDIR:
            try:
                t(tmp)
                print(f"PASS  {t.__name__}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {t.__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(TESTS_NO_TMPDIR) + len(TESTS_TMPDIR)
    if failed:
        print(f"\n{failed}/{total} tests failed")
        sys.exit(1)
    print(f"\nAll {total} tests passed")


if __name__ == "__main__":
    main()
