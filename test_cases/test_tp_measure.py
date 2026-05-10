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
    for g in data["grays"]:
        assert abs(g["delta_y10"]) < 1.0, g


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


TESTS_TMPDIR = [
    test_extract_frame_yuv422p10le_720x486,
    test_extract_pad_720x480_dvd_like,
    test_measure_end_to_end_zero_deltas,
    test_measure_end_to_end_dvd_padding,
]
TESTS_NO_TMPDIR = [
    test_pad_to_486_centers_grey,
    test_sample_region_on_synthesized_identity,
    test_sample_region_with_translation,
    test_pad_to_486_raises_on_oversized_height,
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
