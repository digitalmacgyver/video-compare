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


def test_synthesize_tartan_centers():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for r in tp_chart.TARTAN_REGIONS:
        x, y, w, h = r["ideal_box"]
        cx, cy = x + w // 2, y + h // 2
        # Sample a 3x3 patch at the box centre to avoid grid edges.
        y_sample = float(Y[cy - 1:cy + 2, cx - 1:cx + 2].mean())
        u_sample = float(U[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        v_sample = float(V[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        assert abs(y_sample - r["expected"]["y10"]) < 2.0, (
            f"{r['id']} Y10: got {y_sample:.1f}, want {r['expected']['y10']:.1f}"
        )
        assert abs(u_sample - r["expected"]["u10"]) < 2.0, (
            f"{r['id']} U10: got {u_sample:.1f}, want {r['expected']['u10']:.1f}"
        )
        assert abs(v_sample - r["expected"]["v10"]) < 2.0, (
            f"{r['id']} V10: got {v_sample:.1f}, want {r['expected']['v10']:.1f}"
        )


def test_synthesize_gray_strip_centers():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for r in tp_chart.GRAY_REGIONS:
        x, y, w, h = r["ideal_box"]
        cx, cy = x + w // 2, y + h // 2
        y_sample = float(Y[cy - 1:cy + 2, cx - 1:cx + 2].mean())
        u_sample = float(U[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        v_sample = float(V[cy - 1:cy + 2, cx // 2 - 1:cx // 2 + 2].mean())
        assert abs(y_sample - r["expected"]["y10"]) < 1.0, (
            f"{r['id']} Y10: got {y_sample:.1f}, want {r['expected']['y10']:.1f}"
        )
        assert int(round(u_sample)) == 512, (
            f"{r['id']} U10: got {u_sample:.1f}, want 512"
        )
        assert int(round(v_sample)) == 512, (
            f"{r['id']} V10: got {v_sample:.1f}, want 512"
        )



def test_synthesize_renders_all_four_boundary_triangles():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        bm = tri["ideal_back_midpoint"]
        apex = tri["ideal_apex"]
        # The back midpoint and apex (clamped to inside the frame) must be black.
        assert Y[int(bm[1]), int(bm[0])] == tp_chart.BLACK_Y10, \
            f"back midpoint of {tri['id']} not black"
        ax, ay = int(apex[0]), max(0, min(485, int(apex[1])))
        assert Y[ay, ax] == tp_chart.BLACK_Y10, f"apex of {tri['id']} not black"


def test_synthesize_back_corners_of_each_triangle_are_black():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        for corner_key in ("ideal_back_corner_1", "ideal_back_corner_2"):
            cx, cy = tri[corner_key]
            assert Y[int(cy), int(cx)] == tp_chart.BLACK_Y10, \
                f"{tri['id']}.{corner_key} not black at ({cx},{cy})"


def test_synthesize_old_placeholder_cell_no_longer_triangle():
    """The Stage 1 placeholder lived at Y[81:108, 0:30] and was a filled
    triangle with ~200 dark pixels. The Stage 2 synthesizer no longer
    draws there; only the grid lines passing through that cell remain.
    """
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    cell = Y[81:108, 0:30]
    black_count = int((cell == tp_chart.BLACK_Y10).sum())
    # Only the y=108 grid line (3 px tall, full 30 wide = 90 px) + the x=0
    # grid line (54 px tall, 3 wide minus overlap with the y=108 line) ≈
    # 90 + ~75 - overlap = ~155 dark pixels at most. The old triangle filled
    # ~330 dark pixels. So < 200 means no triangle.
    assert black_count < 200, \
        f"old placeholder cell still has {black_count} black pixels"


def test_cli_writes_png(tmp_dir):
    import subprocess
    out = os.path.join(tmp_dir, "ideal.png")
    cmd = ["python", "tp_synthesize.py", "--raster", "720x486", "--output", out]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists(out)
    # PNG should be 720x486
    import cv2
    img = cv2.imread(out)
    assert img.shape == (486, 720, 3)


def test_cli_writes_yuv(tmp_dir):
    import subprocess
    out = os.path.join(tmp_dir, "ideal.yuv")
    cmd = ["python", "tp_synthesize.py", "--raster", "720x486", "--output", out]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists(out)
    # yuv422p10le: (720*486 + 2 * 360*486) * 2 bytes
    expected_size = (720 * 486 + 2 * 360 * 486) * 2
    assert os.path.getsize(out) == expected_size


def test_fill_box_rejects_odd_x():
    Y = np.zeros((100, 100), dtype=np.uint16)
    U = np.zeros((100, 50), dtype=np.uint16)
    V = np.zeros((100, 50), dtype=np.uint16)
    raised = False
    try:
        tp_synthesize._fill_box_yuv422(Y, U, V, (1, 0, 10, 10), 512, 512, 512)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError for odd x"


def test_fill_box_rejects_odd_w():
    Y = np.zeros((100, 100), dtype=np.uint16)
    U = np.zeros((100, 50), dtype=np.uint16)
    V = np.zeros((100, 50), dtype=np.uint16)
    raised = False
    try:
        tp_synthesize._fill_box_yuv422(Y, U, V, (0, 0, 11, 10), 512, 512, 512)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError for odd w"


def test_synthesize_renders_registration_cross():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    rc = tp_chart.REGISTRATION_CROSS
    cx, cy = rc["ideal_x"], rc["ideal_y"]
    # Cross center is white.
    assert Y[cy, cx] >= 800
    # Pixels 5-6 px from center along the horizontal arm are white.
    for dx in (-6, -5, 5, 6):
        assert Y[cy, cx + dx] >= 800, f"horiz arm at dx={dx} not bright"
    # Pixels 5-6 px from center along the vertical arm are white.
    for dy in (-6, -5, 5, 6):
        assert Y[cy + dy, cx] >= 800, f"vert arm at dy={dy} not bright"
    # The black box surrounds the cross: 3 px below center and 5 px right
    # of center is outside both arms (horiz arm is on the cy row +/- arm_th,
    # vert arm is on the cx column +/- arm_th), so should be box-black.
    assert Y[cy - 3, cx + 5] == tp_chart.BLACK_Y10


def test_synthesize_renders_black_circle_dark_ring_at_expected_radius():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    bc = tp_chart.BLACK_CIRCLE
    cx, cy = bc["ideal_cx"], bc["ideal_cy"]
    # Synth now renders the ring as a PAR-elliptical ring (horizontal
    # semi-axis = vertical * 11/10) to match real NTSC captures. Sample
    # along that ellipse.
    import math
    ry = bc["expected_radius_px"]
    rx = ry * tp_chart.NTSC_PAR_X_OVER_Y
    dark_count = 0
    sampled = 0
    for i in range(32):
        theta = 2 * math.pi * i / 32
        x = int(round(cx + rx * math.cos(theta)))
        y = int(round(cy + ry * math.sin(theta)))
        if 0 <= x < 720 and 0 <= y < 486:
            sampled += 1
            if Y[y, x] < tp_chart.GREY_BACKGROUND_Y10 - 100:
                dark_count += 1
    assert sampled >= 24, f"only {sampled} ring samples landed in-frame"
    assert dark_count >= int(0.75 * sampled), \
        f"only {dark_count}/{sampled} ring samples are dark"


def test_synthesized_red_block_is_saturated():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    win_y = Y[445:475, 600:660].astype(np.float32).mean()
    # 100% red in BT.601 limited range: Y10 ~ 64 + 0.299*876 ≈ 326.
    assert 250 < win_y < 400, f"red Y mean={win_y:.0f}, want ~326"
    # V is half-x sampled; red at full sat -> V well above center (512).
    win_v = V[445:475, 300:330].astype(np.float32).mean()
    assert win_v > 700, f"red V mean={win_v:.0f}, want >700"


def test_synthesized_magenta_steps_monotonic():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    ys = [Y[450:470, x0:x0 + 30].astype(np.float32).mean() for x0 in (15, 75, 135)]
    assert ys[0] < ys[1] < ys[2], f"magenta steps not monotonic: {ys}"


def test_synthesized_bursts_have_expected_modulation():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for r in tp_chart.BURST_REGIONS:
        x, y, w, h = r["ideal_box"]
        win = Y[y:y + h, x:x + w].astype(np.float32)
        amp = float(win.max() - win.min())
        assert amp > 600, f"{r['id']}: amp={amp:.0f}, want >600"


import tempfile
import shutil

TESTS_NO_TMPDIR = [
    test_synthesize_shapes,
    test_synthesize_grey_background_dominates,
    test_synthesize_chroma_centred_off_color_regions,
    test_synthesize_grid_intersections_dark,
    test_synthesize_tartan_centers,
    test_synthesize_gray_strip_centers,
    test_synthesize_renders_all_four_boundary_triangles,
    test_synthesize_back_corners_of_each_triangle_are_black,
    test_synthesize_old_placeholder_cell_no_longer_triangle,
    test_synthesize_renders_registration_cross,
    test_synthesize_renders_black_circle_dark_ring_at_expected_radius,
    test_fill_box_rejects_odd_x,
    test_fill_box_rejects_odd_w,
    test_synthesized_bursts_have_expected_modulation,
    test_synthesized_red_block_is_saturated,
    test_synthesized_magenta_steps_monotonic,
]
TESTS_TMPDIR = [
    test_cli_writes_png,
    test_cli_writes_yuv,
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
    tmp = tempfile.mkdtemp(prefix="tp_synth_test_")
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
