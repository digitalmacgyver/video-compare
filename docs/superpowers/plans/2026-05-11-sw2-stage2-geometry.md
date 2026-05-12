# SW2 Stage 2 — Geometry / Picture-in-Raster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add boundary-triangle / registration-cross / black-circle detection, a sequential-with-feedback affine re-fit, picture-in-raster geometry measurements, and an extended comparison HTML — all driven by a new test-only fixture module with explicit per-detector accuracy thresholds.

**Architecture:** Stage 2 changes existing files (no new product modules). A new `tp_fixtures.py` (test-only) provides `synthesize_with_ground_truth(...)` that produces frames with controlled degradations and per-fiducial ground truth positions. Detectors live in `tp_register.py` behind a generic `detect_fiducial(Y, fid)` dispatcher that routes by `fid["kind"]`. `register_with_geometry()` runs Stage 1 registration first, then Stage 2 detection in registered coords, then a single re-fit of the affine using all anchors — so Stage 1 tartan/gray sampling picks up the tighter fit transparently.

**Tech Stack:** Python 3, NumPy, OpenCV (cv2.estimateAffine2D RANSAC, cv2.matchTemplate, cv2.fitEllipse, cv2.circle), ffmpeg (test fixture encoding via ProRes422 HQ), plain-Python test harness (sys.exit on fail, NOT pytest, following the `test_metrics.py` / `test_tp_*.py` convention already in `test_cases/`).

**Reference spec:** `docs/superpowers/specs/2026-05-11-sw2-stage2-geometry-design.md`

---

## File Map

```
tp_chart.py                      MODIFY  catalogs (grids, triangles, cross, circle, IDEAL_PICTURE_BOX)
tp_synthesize.py                 MODIFY  render all 4 triangles, cross, circle
tp_register.py                   MODIFY  dispatcher + 3 new detectors + orchestrator + sequential-with-feedback
tp_measure.py                    MODIFY  call register_with_geometry, embed geometry block in JSON
tp_compare.py                    MODIFY  render_geometry_section + CSS additions
tp_calibrate.py                  MODIFY  stage2-fiducials preset

tp_fixtures.py                   NEW     synthesize_with_ground_truth (test-only)

test_cases/test_tp_chart.py      MODIFY  catalog assertions
test_cases/test_tp_synthesize.py MODIFY  render assertions for triangles/cross/circle
test_cases/test_tp_fixtures.py   NEW     fixture ground-truth + degradation tests
test_cases/test_tp_register.py   MODIFY  per-detector + dispatcher + orchestrator tests (fixture-driven)
test_cases/test_tp_measure.py    MODIFY  geometry block in JSON
test_cases/test_tp_compare.py    MODIFY  geometry section in HTML
```

Every product module gains responsibility within its existing scope. `tp_fixtures.py` is the only new module; it is imported only by test files.

---

## Convention reminders for the implementer

- Tests use the **plain-Python convention** (no pytest). Each test file ends with a `TESTS_*` list + a `main()` that iterates, catches exceptions, prints PASS/FAIL, and `sys.exit(1)` on any failure. See `test_cases/test_tp_chart.py` for the canonical shape.
- New tests are appended to existing `TESTS_NO_TMPDIR` / `TESTS_TMPDIR` lists. Tests that need a tmpdir take a `tmp_dir` positional arg; the harness allocates it.
- Run all tests after each task:
  ```bash
  source venv/bin/activate
  for f in test_cases/test_tp_*.py; do python "$f" || exit 1; done
  ```
- Commit after every passing task. Use the existing `feat:` / `refactor:` / `test:` prefixes consistent with recent commits.

---

## Task 1: Refresh GRID_LANDMARKS catalog (drop L8, relocate top row, add y=216)

**Files:**
- Modify: `tp_chart.py:187-203`
- Test:   `test_cases/test_tp_chart.py:114-129`

Diagnostic finding: L8 (360, 378) is universally undetectable; L1/L2/L3 at y=54 are noisy. Catalog tune: 12 anchors at y ∈ {108, 162, 216, 270}, x ∈ {180, 360, 420, 540, 600} biased to avoid tartan/gray strip and the burst columns.

- [ ] **Step 1: Update the test_grid_landmarks assertions to expect 12 anchors at the new positions**

Replace `test_grid_landmarks` and `test_grid_landmark_distribution` in `test_cases/test_tp_chart.py`:

```python
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
```

- [ ] **Step 2: Run tests; verify failure**

Run: `python test_cases/test_tp_chart.py`
Expected: FAIL on `test_grid_landmarks` / `test_grid_landmark_distribution`.

- [ ] **Step 3: Update the GRID_LANDMARKS section in tp_chart.py**

Replace lines 177-203 of `tp_chart.py` (the `# REGISTRATION LANDMARK CATALOG` section through the `GRID_LANDMARKS = [...]` list) with:

```python
# =====================================================================
# REGISTRATION LANDMARK CATALOG
# =====================================================================
#
# Black grid intersections on the grey background. 12 anchors in the safe
# interior of the chart (avoiding the tartan/gray strip on the upper-left,
# the busy top of the chart, and the chart border).
#
# Diagnostic context: the Stage 1 catalog (8 anchors including L8 at
# (360, 378) and L1-L3 at y=54) gave only 4 of 8 surviving RANSAC -- L8 has
# no clean intersection, the top row was near the boundary triangles and
# tartan and showed 1-4 px noise. The y∈{108..270} interior gives sub-px
# residuals against a single-affine fit on the real captures.

_LANDMARK_GRID = [
    # (id, ideal_x, ideal_y)
    ("L1",  180, 108), ("L2",  360, 108), ("L3",  540, 108),
    ("L4",  180, 162), ("L5",  420, 162), ("L6",  600, 162),
    ("L7",  180, 216), ("L8",  360, 216), ("L9",  540, 216),
    ("L10", 180, 270), ("L11", 420, 270), ("L12", 600, 270),
]

GRID_LANDMARKS = [
    {
        "id": rid,
        "kind": "grid_intersection",
        "ideal_x": x,
        "ideal_y": y,
        "search_window_px": 24,
    }
    for rid, x, y in _LANDMARK_GRID
]
```

- [ ] **Step 4: Run tests; verify pass**

Run: `python test_cases/test_tp_chart.py`
Expected: All tests pass (13 tests).

Also run `python test_cases/test_tp_register.py` to make sure detection still works on synthesized identity (it should — the synthesizer draws the grid).

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "$(cat <<'EOF'
feat(tp_chart): refresh GRID_LANDMARKS to 12 anchors in safe interior

Drops L8 (360,378 — no clean intersection at that position) and the
y=54 top row (near busy upper features). New catalog: 12 anchors at
y∈{108,162,216,270}, x∈{180..600}. Adds 'kind' field for the upcoming
Stage 2 fiducial dispatcher.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add IDEAL_PICTURE_BOX constant

**Files:**
- Modify: `tp_chart.py` (after `GRAY_REGIONS` block, before the landmark catalog)
- Test:   `test_cases/test_tp_chart.py`

The geometry-derivation step projects the chart's ideal picture corners through the final affine to compare with the apex-derived active picture box.

- [ ] **Step 1: Write the failing test**

Add to `test_cases/test_tp_chart.py` (and append to TESTS list):

```python
def test_ideal_picture_box():
    box = tp_chart.IDEAL_PICTURE_BOX
    assert box == {"left": 0, "top": 0, "right": 719, "bottom": 485}
    corners = tp_chart.IDEAL_PICTURE_BOX_CORNERS
    # TL, TR, BL, BR
    assert corners == [(0, 0), (719, 0), (0, 485), (719, 485)]
```

Append `test_ideal_picture_box` to the `TESTS` list.

- [ ] **Step 2: Run; verify failure** — `python test_cases/test_tp_chart.py`
Expected: FAIL — `AttributeError: module 'tp_chart' has no attribute 'IDEAL_PICTURE_BOX'`.

- [ ] **Step 3: Add the constant to tp_chart.py**

Insert after line 174 (the `GRAY_REGIONS = _build_gray_regions()` line), before the existing `# REGISTRATION LANDMARK CATALOG` comment:

```python


# =====================================================================
# IDEAL PICTURE BOX
# =====================================================================
#
# Active picture extent in ideal 720x486 coords (full SDI raster). Used by
# detect_geometry to compare the apex-derived active picture box (in
# capture coords) against where the chart's ideal active picture would
# appear in capture coords after the final affine.

IDEAL_PICTURE_BOX = {"left": 0, "top": 0, "right": 719, "bottom": 485}
IDEAL_PICTURE_BOX_CORNERS = [
    (IDEAL_PICTURE_BOX["left"],  IDEAL_PICTURE_BOX["top"]),     # TL
    (IDEAL_PICTURE_BOX["right"], IDEAL_PICTURE_BOX["top"]),     # TR
    (IDEAL_PICTURE_BOX["left"],  IDEAL_PICTURE_BOX["bottom"]),  # BL
    (IDEAL_PICTURE_BOX["right"], IDEAL_PICTURE_BOX["bottom"]),  # BR
]
```

- [ ] **Step 4: Run; verify pass**

Run: `python test_cases/test_tp_chart.py`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(tp_chart): add IDEAL_PICTURE_BOX + corners for Stage 2 geometry derivation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add BOUNDARY_TRIANGLES catalog

**Files:**
- Modify: `tp_chart.py`
- Test:   `test_cases/test_tp_chart.py`

4 corner-region triangles. TL/TR `apex_up`; BL/BR `apex_down`. Each entry exposes the chart-spec offset (apex − back_midpoint) so the detector can fall back to inferring the apex when it's clipped.

Canonical chart-spec positions (refined by `tp_calibrate.py --preset stage2-fiducials` against real captures in Task 22):

- TL: back midpoint (60, 30), back corners (50, 30) & (70, 30), apex (60, 3).
- TR: back midpoint (660, 30), back corners (650, 30) & (670, 30), apex (660, 3).
- BL: back midpoint (60, 456), back corners (50, 456) & (70, 456), apex (60, 483).
- BR: back midpoint (660, 456), back corners (650, 456) & (670, 456), apex (660, 483).

(Base width 20 px; height 27 px; back-row offset 30 px from the corresponding chart edge.)

- [ ] **Step 1: Write the failing test**

Add to `test_cases/test_tp_chart.py`:

```python
def test_boundary_triangles_catalog():
    tris = tp_chart.BOUNDARY_TRIANGLES
    assert [t["id"] for t in tris] == ["TL", "TR", "BL", "BR"]
    expected_orient = {"TL": "apex_up", "TR": "apex_up",
                       "BL": "apex_down", "BR": "apex_down"}
    for t in tris:
        assert t["kind"] == "boundary_triangle"
        assert t["orientation"] == expected_orient[t["id"]]
        for key in ("ideal_back_corner_1", "ideal_back_corner_2",
                    "ideal_back_midpoint", "ideal_apex",
                    "search_window_px"):
            assert key in t, key
        bm = t["ideal_back_midpoint"]
        bc1 = t["ideal_back_corner_1"]
        bc2 = t["ideal_back_corner_2"]
        # back_midpoint is the midpoint of the two back corners.
        assert bm == ((bc1[0] + bc2[0]) / 2, (bc1[1] + bc2[1]) / 2)
        # apex sits 27 px from back_midpoint in the orientation direction.
        if t["orientation"] == "apex_up":
            assert t["ideal_apex"] == (bm[0], bm[1] - 27)
        elif t["orientation"] == "apex_down":
            assert t["ideal_apex"] == (bm[0], bm[1] + 27)


def test_boundary_triangles_back_corner_spacing():
    # 20 px base across the back edge.
    for t in tp_chart.BOUNDARY_TRIANGLES:
        bc1, bc2 = t["ideal_back_corner_1"], t["ideal_back_corner_2"]
        spacing = ((bc1[0] - bc2[0]) ** 2 + (bc1[1] - bc2[1]) ** 2) ** 0.5
        assert abs(spacing - 20.0) < 0.5
```

Append both names to the `TESTS` list.

- [ ] **Step 2: Run; verify failure** — Expected: FAIL on missing `BOUNDARY_TRIANGLES`.

- [ ] **Step 3: Add BOUNDARY_TRIANGLES to tp_chart.py**

Insert after the `GRID_LANDMARKS` block:

```python


# =====================================================================
# BOUNDARY TRIANGLES (Stage 2)
# =====================================================================
#
# 4 corner-region triangles that mark the picture boundaries. TL/TR point
# UP (apex toward top edge); BL/BR point DOWN (apex toward bottom edge).
# The base of each triangle (back edge, furthest from the picture edge it
# marks) sits 30 px inside the chart from the corresponding edge.
#
# Registration uses the back corners (clip-resistant). The apex can be
# either detected from data or inferred from back_midpoint + chart-spec
# offset (apex - back_midpoint).

_BASE_HALF = 10   # half the back-edge width (20 px total)
_HEIGHT    = 27   # apex distance from back_midpoint (perpendicular)
_BACK_INSET = 30  # back edge sits 30 px inside the chart from the edge


def _make_triangle(rid, orientation, x_center, edge):
    """Build a triangle entry. `edge` is the chart edge it marks
    ('top' or 'bottom'); the back edge sits `_BACK_INSET` px inside
    that edge; the apex sits at the edge."""
    if edge == "top":
        back_y = _BACK_INSET
        apex_y = back_y - _HEIGHT
    elif edge == "bottom":
        back_y = 485 - _BACK_INSET
        apex_y = back_y + _HEIGHT
    else:
        raise ValueError(edge)
    bc1 = (x_center - _BASE_HALF, back_y)
    bc2 = (x_center + _BASE_HALF, back_y)
    bm  = (x_center, back_y)
    apex = (x_center, apex_y)
    return {
        "id": rid,
        "kind": "boundary_triangle",
        "orientation": orientation,
        "ideal_back_corner_1": bc1,
        "ideal_back_corner_2": bc2,
        "ideal_back_midpoint": bm,
        "ideal_apex": apex,
        "search_window_px": 40,
    }


BOUNDARY_TRIANGLES = [
    _make_triangle("TL", "apex_up",   60, "top"),
    _make_triangle("TR", "apex_up",  660, "top"),
    _make_triangle("BL", "apex_down", 60, "bottom"),
    _make_triangle("BR", "apex_down", 660, "bottom"),
]
```

- [ ] **Step 4: Run; verify pass**

Run: `python test_cases/test_tp_chart.py`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(tp_chart): add BOUNDARY_TRIANGLES catalog (TL,TR,BL,BR)

Each triangle records back-corner anchors, back_midpoint, and apex per
the chart spec. Registration uses back corners; apex_inferred derives
from back_midpoint + chart offset for clip detection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add REGISTRATION_CROSS catalog

**Files:**
- Modify: `tp_chart.py`
- Test:   `test_cases/test_tp_chart.py`

Picture-center registration feature: black box with a centered white "+". 200 ns at 13.5 MHz ≈ 2.7 samples → 3 px arm width. Box is 24×24 px, white arms are 17 px tip-to-tip (centered).

- [ ] **Step 1: Write the failing test**

```python
def test_registration_cross_catalog():
    rc = tp_chart.REGISTRATION_CROSS
    assert rc["id"] == "RC"
    assert rc["kind"] == "registration_cross"
    assert rc["ideal_x"] == 360
    assert rc["ideal_y"] == 243
    assert rc["ideal_arm_len_px"] == 17
    assert rc["ideal_arm_thickness_px"] == 3
    assert rc["box_size_px"] == 24
    assert rc["search_window_px"] >= 32
```

Append to TESTS list.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Add the catalog to tp_chart.py**

Insert after BOUNDARY_TRIANGLES:

```python


# =====================================================================
# REGISTRATION CROSS (Stage 2)
# =====================================================================
#
# Picture-center sub-pixel registration feature: a black box containing a
# centered white "+". Used for sub-pixel registration accuracy and for
# detecting directionally-biased aperture / sharpening filters via the
# horizontal-vs-vertical arm-length asymmetry.

REGISTRATION_CROSS = {
    "id": "RC",
    "kind": "registration_cross",
    "ideal_x": 360,
    "ideal_y": 243,
    "ideal_arm_len_px": 17,    # tip-to-tip length of each arm
    "ideal_arm_thickness_px": 3,
    "box_size_px": 24,
    "search_window_px": 40,
}
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(tp_chart): add REGISTRATION_CROSS catalog

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add BLACK_CIRCLE catalog

**Files:**
- Modify: `tp_chart.py`
- Test:   `test_cases/test_tp_chart.py`

Black ring centered on the picture. Diameter ≈ picture height (486 px) → radius ≈ 243 px. Line thickness 168 ns at 13.5 MHz ≈ 2.3 samples → 3 px ring.

- [ ] **Step 1: Write the failing test**

```python
def test_black_circle_catalog():
    bc = tp_chart.BLACK_CIRCLE
    assert bc["id"] == "BC"
    assert bc["kind"] == "black_circle"
    assert bc["ideal_cx"] == 360
    assert bc["ideal_cy"] == 243
    assert bc["expected_radius_px"] == 243
    assert bc["ring_thickness_px"] == 3
    assert bc["search_band_px"] >= 10
```

Append to TESTS list.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Add the catalog to tp_chart.py**

```python


# =====================================================================
# BLACK CIRCLE (Stage 2)
# =====================================================================
#
# Black ring centered on the picture. Diameter = picture height (486 px),
# ring thickness 168 ns @ 13.5 MHz ~ 3 samples. Used to measure aspect
# ratio (rx vs ry) and any picture-vs-spec scaling drift.

BLACK_CIRCLE = {
    "id": "BC",
    "kind": "black_circle",
    "ideal_cx": 360,
    "ideal_cy": 243,
    "expected_radius_px": 243,
    "ring_thickness_px": 3,
    "search_band_px": 15,
}
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(tp_chart): add BLACK_CIRCLE catalog

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Synthesize all 4 boundary triangles

**Files:**
- Modify: `tp_synthesize.py:92-118`
- Test:   `test_cases/test_tp_synthesize.py`

Replace the Stage 1 single-triangle placeholder with all four triangles driven by `tp_chart.BOUNDARY_TRIANGLES`. Each triangle is a filled black triangle with the back-edge corners and the apex from the catalog.

- [ ] **Step 1: Write the failing test**

Add to `test_cases/test_tp_synthesize.py` (look at the existing file for shape; if it doesn't exist as a separate file, the assertions can go in test_tp_chart_renders.py — but per the existing structure, append here):

```python
import numpy as np
import tp_chart
import tp_synthesize


def test_synthesize_renders_all_four_boundary_triangles():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        bm = tri["ideal_back_midpoint"]
        apex = tri["ideal_apex"]
        # The back midpoint and the apex (or just inside it) must be black.
        assert Y[int(bm[1]), int(bm[0])] == tp_chart.BLACK_Y10, \
            f"back midpoint of {tri['id']} not black"
        # Apex pixel (capped to frame) — clamp to inside the frame for the test.
        ax, ay = int(apex[0]), max(0, min(485, int(apex[1])))
        assert Y[ay, ax] == tp_chart.BLACK_Y10, f"apex of {tri['id']} not black"


def test_synthesize_back_corners_of_each_triangle_are_black():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        for corner_key in ("ideal_back_corner_1", "ideal_back_corner_2"):
            cx, cy = tri[corner_key]
            assert Y[int(cy), int(cx)] == tp_chart.BLACK_Y10, \
                f"{tri['id']}.{corner_key} not black at ({cx},{cy})"


def test_synthesize_no_stray_dark_outside_triangle_for_TL():
    # 5 px above TL apex should be grey background (or otherwise non-triangle).
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    tl = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    apex_y = tl["ideal_apex"][1]
    # apex_up: above apex (y < apex_y) should be grey background.
    # The apex is at y=3, so y=0..2 should be grey (above the triangle).
    if apex_y >= 1:
        # Pixel at (apex_x, apex_y - 1) -- 1 row above apex -- should be grey
        # unless rendering bleed-over. Allow either grey or the upper-most row
        # if that's outside the triangle's drawn rows.
        pass  # apex is at y=3, so y=0,1,2 -- but those rows might intersect the
              # grid line at y=0 which is also black. Use y=2 column at apex_x:
        assert (Y[2, int(tl["ideal_apex"][0])] != tp_chart.BLACK_Y10
                or Y[2, 100] == Y[2, int(tl["ideal_apex"][0])]), \
            "TL apex should not bleed above into non-grid background"
```

(The last test is intentionally loose because the top grid line itself sits at y=0..1 and is black; what we're really checking is "we don't paint random extra pixels". A cleaner test would inspect the count of black pixels.)

Append the three test names to `TESTS_NO_TMPDIR` (or `TESTS` if that's the convention in the existing file).

If `test_cases/test_tp_synthesize.py` does not yet exist, create it with the standard harness shape from `test_cases/test_tp_chart.py` (imports, TESTS list, main()).

- [ ] **Step 2: Run; verify failure**

Run: `python test_cases/test_tp_synthesize.py`
Expected: FAIL — the existing placeholder triangle is at (3,84)-(3,105)-(24,94), nothing else; new tests fail.

- [ ] **Step 3: Replace the Stage 1 triangle code with all four**

In `tp_synthesize.py`, **delete** lines 92-108 (the entire `_draw_boundary_triangle_upper_left` function) and replace with:

```python
def _fill_triangle_y(Y: np.ndarray, p1, p2, p3, y10: int = None) -> None:
    """Fill a triangle on the Y plane only (chroma not affected — boundary
    triangles are black on the chroma-center-grey background).

    Uses cv2.fillPoly which handles the geometry. p1/p2/p3 are (x, y)
    integer coords."""
    import cv2
    if y10 is None:
        y10 = tp_chart.BLACK_Y10
    pts = np.array([[p1, p2, p3]], dtype=np.int32)
    cv2.fillPoly(Y, pts, int(y10))


def _draw_boundary_triangles(Y: np.ndarray) -> None:
    """Render all 4 boundary triangles from tp_chart.BOUNDARY_TRIANGLES."""
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        bc1 = tuple(int(v) for v in tri["ideal_back_corner_1"])
        bc2 = tuple(int(v) for v in tri["ideal_back_corner_2"])
        apex = tuple(int(v) for v in tri["ideal_apex"])
        # cv2.fillPoly handles out-of-frame apex coords (clips them).
        _fill_triangle_y(Y, bc1, bc2, apex)
```

Update `synthesize()` body (currently around lines 110-119): replace the `_draw_boundary_triangle_upper_left(Y)` call with `_draw_boundary_triangles(Y)`:

```python
def synthesize(width: int = 720, height: int = 486) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ideal SW2 frame as (Y, U, V) uint16 planes (yuv422p10le)."""
    if width % 2 != 0:
        raise ValueError(f"width must be even for yuv422p; got {width}")
    Y, U, V = _make_grey_planes(width, height)
    _draw_grid(Y)
    _draw_tartan(Y, U, V)
    _draw_gray_strip(Y, U, V)
    _draw_boundary_triangles(Y)
    return Y, U, V
```

- [ ] **Step 4: Run; verify pass**

Run: `python test_cases/test_tp_synthesize.py`
Expected: All synthesize tests pass.

Also re-run `python test_cases/test_tp_measure.py` to confirm the synthesizer change didn't break Stage 1 end-to-end. Expected: all pass (the new triangles don't overlap the tartan/gray sample regions or grid landmarks at the new positions).

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(tp_synthesize): render all 4 boundary triangles from catalog

Replaces the Stage 1 single upper-left placeholder. Each triangle is a
filled black triangle defined by back corners and apex per the
BOUNDARY_TRIANGLES catalog. cv2.fillPoly handles clipping when an apex
falls outside the raster (BL/BR apexes near y=483 may be at the edge).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Synthesize the registration cross

**Files:**
- Modify: `tp_synthesize.py`
- Test:   `test_cases/test_tp_synthesize.py`

A 24×24 black box centered at (360, 243), with a centered white "+" of 17 px tip-to-tip and 3 px arm thickness.

- [ ] **Step 1: Write the failing test**

```python
def test_synthesize_renders_registration_cross():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    rc = tp_chart.REGISTRATION_CROSS
    cx, cy = rc["ideal_x"], rc["ideal_y"]
    # Cross center is white.
    assert Y[cy, cx] >= 800
    # Pixels 4-7 px from center along the horizontal arm are white.
    for dx in (-6, -5, 5, 6):
        assert Y[cy, cx + dx] >= 800, f"horiz arm at dx={dx} not bright"
    # Pixels 4-7 px from center along the vertical arm are white.
    for dy in (-6, -5, 5, 6):
        assert Y[cy + dy, cx] >= 800, f"vert arm at dy={dy} not bright"
    # The black box surrounds the cross: 10 px above center and 5 px right
    # of arm is in the box (not arm), must be black.
    # arm half-thickness 1, arm half-length 8; box half 12. Check (cy-3, cx+5):
    # 5 px right of center, 3 px below — outside the horizontal arm (vertical
    # arm only covers cx-1..cx+1; horizontal arm covers cy-1..cy+1). So
    # (cy-3, cx+5) is box-black.
    assert Y[cy - 3, cx + 5] == tp_chart.BLACK_Y10
```

Append `test_synthesize_renders_registration_cross` to `TESTS_NO_TMPDIR`.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Add the renderer**

Append to `tp_synthesize.py` after `_draw_boundary_triangles`:

```python
def _draw_registration_cross(Y: np.ndarray) -> None:
    """Render the registration cross from tp_chart.REGISTRATION_CROSS.

    A black square (box_size_px wide) with a white plus inside it
    (ideal_arm_len_px tip-to-tip, ideal_arm_thickness_px wide).
    """
    rc = tp_chart.REGISTRATION_CROSS
    cx, cy = rc["ideal_x"], rc["ideal_y"]
    box = rc["box_size_px"]
    arm_len = rc["ideal_arm_len_px"]
    arm_th  = rc["ideal_arm_thickness_px"]
    half_box = box // 2
    half_len = arm_len // 2
    half_th  = arm_th // 2
    # Black box
    Y[cy - half_box:cy + half_box, cx - half_box:cx + half_box] = tp_chart.BLACK_Y10
    # White cross arms
    Y[cy - half_th:cy + half_th + 1, cx - half_len:cx + half_len + 1] = tp_chart.WHITE_Y10
    Y[cy - half_len:cy + half_len + 1, cx - half_th:cx + half_th + 1] = tp_chart.WHITE_Y10
```

Update `synthesize()` to call it after `_draw_boundary_triangles`:

```python
    _draw_boundary_triangles(Y)
    _draw_registration_cross(Y)
    return Y, U, V
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(tp_synthesize): render the registration cross at picture center

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Synthesize the black circle

**Files:**
- Modify: `tp_synthesize.py`
- Test:   `test_cases/test_tp_synthesize.py`

Black ring with diameter ≈ picture height, thickness 3 px. Uses `cv2.circle` with thickness=3.

- [ ] **Step 1: Write the failing test**

```python
def test_synthesize_renders_black_circle_dark_ring_at_expected_radius():
    Y, _, _ = tp_synthesize.synthesize(720, 486)
    bc = tp_chart.BLACK_CIRCLE
    cx, cy = bc["ideal_cx"], bc["ideal_cy"]
    r = bc["expected_radius_px"]
    # Sample 32 angles around the circle at radius r; the ring pixels should
    # all be black (Y10 == 64).
    import math
    for i in range(32):
        theta = 2 * math.pi * i / 32
        x = int(round(cx + r * math.cos(theta)))
        y = int(round(cy + r * math.sin(theta)))
        if 0 <= x < 720 and 0 <= y < 486:
            assert Y[y, x] == tp_chart.BLACK_Y10, \
                f"ring sample angle={i} at ({x},{y}) not black"
    # 20 px outside the ring (at radius r+20) is OUTSIDE the picture for most
    # angles; just confirm 20 px INSIDE the ring (r-20) is grey/background or
    # whatever-is-there (not the ring).
    x = int(round(cx + (r - 20) * math.cos(0)))
    y = int(round(cy + (r - 20) * math.sin(0)))
    # At angle 0 (rightward), (cx+r-20, cy) should not be the ring; either grey
    # background or grid intersection. Assert not the ring (Y10 != 64 unless it
    # happens to land on a grid line — verify by checking it's not in any
    # grid-line position):
    assert (Y[y, x] != tp_chart.BLACK_Y10
            or x % 60 < 2 or x % 60 > 58
            or y % 54 < 2 or y % 54 > 52), \
        "interior of circle should be grey (unless on a grid line)"
```

Append to TESTS.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Add the renderer**

Append to `tp_synthesize.py`:

```python
def _draw_black_circle(Y: np.ndarray) -> None:
    """Render the black circle ring from tp_chart.BLACK_CIRCLE."""
    import cv2
    bc = tp_chart.BLACK_CIRCLE
    cv2.circle(
        Y,
        center=(bc["ideal_cx"], bc["ideal_cy"]),
        radius=bc["expected_radius_px"],
        color=int(tp_chart.BLACK_Y10),
        thickness=bc["ring_thickness_px"],
        lineType=cv2.LINE_AA,
    )
```

Update `synthesize()` to call it before the registration cross (so the cross overdraws the ring at the center — though they don't overlap at this radius):

```python
    _draw_boundary_triangles(Y)
    _draw_black_circle(Y)
    _draw_registration_cross(Y)
    return Y, U, V
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(tp_synthesize): render the picture-bounding black circle

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Create tp_fixtures.synthesize_with_ground_truth (identity baseline)

**Files:**
- Create: `tp_fixtures.py`
- Create: `test_cases/test_tp_fixtures.py`

Test-only module producing fixture frames with analytic ground-truth fiducial positions. This task implements the **identity** variant (no degradations) — subsequent tasks add shift/rotation/noise/blur/clip support.

- [ ] **Step 1: Write the failing test**

Create `test_cases/test_tp_fixtures.py` with the harness shape:

```python
#!/usr/bin/env python3
"""Tests for tp_fixtures.synthesize_with_ground_truth."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_fixtures


def test_fixture_identity_ground_truth_matches_chart_catalog():
    Y, U, V, gt = tp_fixtures.synthesize_with_ground_truth()
    assert Y.shape == (486, 720)
    assert U.shape == (486, 360)
    assert V.shape == (486, 360)

    # 12 grid intersections with ids L1..L12
    grids = gt["grid_intersections"]
    assert len(grids) == 12
    for lm in tp_chart.GRID_LANDMARKS:
        assert grids[lm["id"]] == (lm["ideal_x"], lm["ideal_y"])

    # 4 triangles
    tris = gt["triangles"]
    assert set(tris.keys()) == {"TL", "TR", "BL", "BR"}
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        entry = tris[tri["id"]]
        assert entry["back_corner_1"] == tri["ideal_back_corner_1"]
        assert entry["back_corner_2"] == tri["ideal_back_corner_2"]
        assert entry["back_midpoint"] == tri["ideal_back_midpoint"]
        assert entry["apex"]          == tri["ideal_apex"]
        assert entry["apex_visible"] is True

    # Cross
    rc = gt["cross"]
    assert rc["center"] == (tp_chart.REGISTRATION_CROSS["ideal_x"],
                            tp_chart.REGISTRATION_CROSS["ideal_y"])

    # Circle
    cc = gt["circle"]
    assert cc["center"] == (tp_chart.BLACK_CIRCLE["ideal_cx"],
                            tp_chart.BLACK_CIRCLE["ideal_cy"])
    assert cc["radius"] == tp_chart.BLACK_CIRCLE["expected_radius_px"]


TESTS = [
    test_fixture_identity_ground_truth_matches_chart_catalog,
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
```

- [ ] **Step 2: Run; verify failure** — `python test_cases/test_tp_fixtures.py`
Expected: FAIL — module `tp_fixtures` does not exist.

- [ ] **Step 3: Create tp_fixtures.py**

```python
"""tp_fixtures: test-only fixture generator with ground-truth fiducial
positions for Stage 2 detector validation.

NOT imported by production code. Tests import this to validate per-
detector accuracy against per-variant thresholds (see Stage 2 design
spec).
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

import tp_chart
import tp_synthesize


def _build_identity_ground_truth() -> Dict[str, Any]:
    """Ground truth for the ideal frame at identity (no degradation)."""
    grids = {
        lm["id"]: (lm["ideal_x"], lm["ideal_y"])
        for lm in tp_chart.GRID_LANDMARKS
    }
    triangles = {}
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        triangles[tri["id"]] = {
            "back_corner_1": tri["ideal_back_corner_1"],
            "back_corner_2": tri["ideal_back_corner_2"],
            "back_midpoint": tri["ideal_back_midpoint"],
            "apex":          tri["ideal_apex"],
            "apex_visible":  True,
        }
    cross = {
        "center": (tp_chart.REGISTRATION_CROSS["ideal_x"],
                   tp_chart.REGISTRATION_CROSS["ideal_y"]),
    }
    circle = {
        "center": (tp_chart.BLACK_CIRCLE["ideal_cx"],
                   tp_chart.BLACK_CIRCLE["ideal_cy"]),
        "radius": tp_chart.BLACK_CIRCLE["expected_radius_px"],
    }
    return {
        "grid_intersections": grids,
        "triangles": triangles,
        "cross": cross,
        "circle": circle,
    }


def synthesize_with_ground_truth(
    width: int = 720,
    height: int = 486,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synthesize the ideal SW2 frame and return per-fiducial ground truth.

    Identity baseline. Future tasks add shift/rotation/noise/blur/clip
    keyword arguments.
    """
    Y, U, V = tp_synthesize.synthesize(width, height)
    gt = _build_identity_ground_truth()
    return Y, U, V, gt
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_fixtures.py test_cases/test_tp_fixtures.py
git commit -m "feat(tp_fixtures): identity-variant fixture with ground-truth fiducials

Test-only module. Subsequent tasks add shift/rotation/noise/blur/clip
degradations for per-detector accuracy validation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Add shift support to tp_fixtures

**Files:**
- Modify: `tp_fixtures.py`
- Test:   `test_cases/test_tp_fixtures.py`

`shift=(dx, dy)` translates the frame globally (with grey-background fill on the wraparound rows/cols) and updates every ground-truth position by `(dx, dy)`.

- [ ] **Step 1: Write the failing test**

```python
def test_fixture_shift_updates_ground_truth_and_frame():
    Y0, _, _, gt0 = tp_fixtures.synthesize_with_ground_truth()
    Y, U, V, gt = tp_fixtures.synthesize_with_ground_truth(shift=(5, 7))
    # Grid intersection in shifted frame: original (180, 108) -> (185, 115)
    assert gt["grid_intersections"]["L1"] == (185, 115)
    # The pixel at the new position should be black (grid intersection).
    assert Y[115, 185] == tp_chart.BLACK_Y10
    # Triangle TL back corner shifted by (5, 7).
    tl_bc1 = gt["triangles"]["TL"]["back_corner_1"]
    assert tl_bc1 == (55, 37)
    # The first 7 rows are the grey-background fill from the shift.
    assert (Y[:7, :] == tp_chart.GREY_BACKGROUND_Y10).all()
    # The first 5 columns are the grey-background fill.
    assert (Y[:, :5] == tp_chart.GREY_BACKGROUND_Y10).all()
```

Append to TESTS.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement shift**

Update `tp_fixtures.py`:

```python
def _shift_planes(Y, U, V, dx, dy):
    """Roll planes by (dx, dy) with grey-background fill on the exposed
    rows/columns. dx is in Y-coords (full-rate); chroma dx is dx // 2 since
    yuv422p10le subsamples horizontally."""
    grey_y = tp_chart.GREY_BACKGROUND_Y10
    grey_c = tp_chart.CHROMA_CENTER

    def _shift(plane, dx_p, dy_p, fill):
        h, w = plane.shape
        out = np.full_like(plane, fill)
        # Clip the source region; place at the shifted destination.
        sx0 = max(0, -dx_p); sx1 = min(w, w - dx_p)
        sy0 = max(0, -dy_p); sy1 = min(h, h - dy_p)
        dx0 = max(0, dx_p); dy0 = max(0, dy_p)
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = plane[sy0:sy1, sx0:sx1]
        return out

    Y = _shift(Y, dx, dy, grey_y)
    U = _shift(U, dx // 2, dy, grey_c)
    V = _shift(V, dx // 2, dy, grey_c)
    return Y, U, V


def _apply_shift_to_ground_truth(gt, dx, dy):
    """Translate every ground-truth position by (dx, dy)."""
    def _t(pt):
        return (pt[0] + dx, pt[1] + dy)
    out = {
        "grid_intersections": {k: _t(v) for k, v in gt["grid_intersections"].items()},
        "triangles": {},
        "cross": {"center": _t(gt["cross"]["center"])},
        "circle": {
            "center": _t(gt["circle"]["center"]),
            "radius": gt["circle"]["radius"],
        },
    }
    for tid, t in gt["triangles"].items():
        out["triangles"][tid] = {
            "back_corner_1": _t(t["back_corner_1"]),
            "back_corner_2": _t(t["back_corner_2"]),
            "back_midpoint": _t(t["back_midpoint"]),
            "apex":          _t(t["apex"]),
            "apex_visible":  t["apex_visible"],
        }
    return out
```

Update the public function signature:

```python
def synthesize_with_ground_truth(
    width: int = 720,
    height: int = 486,
    shift: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synthesize the ideal SW2 frame, apply `shift`, and return per-fiducial
    ground truth in the resulting frame's coords."""
    Y, U, V = tp_synthesize.synthesize(width, height)
    gt = _build_identity_ground_truth()
    dx, dy = shift
    if dx != 0 or dy != 0:
        Y, U, V = _shift_planes(Y, U, V, dx, dy)
        gt = _apply_shift_to_ground_truth(gt, dx, dy)
    return Y, U, V, gt
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_fixtures.py test_cases/test_tp_fixtures.py
git commit -m "feat(tp_fixtures): add shift support with grey-fill on exposed rows

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Add rotation support to tp_fixtures

**Files:**
- Modify: `tp_fixtures.py`
- Test:   `test_cases/test_tp_fixtures.py`

`rotation_deg=θ` rotates the frame about the picture center using `cv2.warpAffine` with grey-background fill, then rotates every ground-truth position analytically.

- [ ] **Step 1: Write the failing test**

```python
def test_fixture_rotation_updates_ground_truth_positions():
    import math
    Y, U, V, gt = tp_fixtures.synthesize_with_ground_truth(rotation_deg=1.5)
    # Rotation about picture center (360, 243). A point at (180, 108) is
    # (180-360, 108-243) = (-180, -135) from center. Rotating by 1.5° clockwise
    # gives:  (-180*cos - -135*sin, -180*sin + -135*cos)
    # cv2 uses screen coords (y grows down), so positive angle = counter-
    # clockwise in math but our implementation rotates the IMAGE by +θ
    # (counter-clockwise on screen); ground truth must match the image.
    theta = math.radians(1.5)
    cx, cy = 360.0, 243.0
    px, py = 180.0, 108.0
    # Rotation matrix (positive θ = CCW with y-down: R = [[cos, sin], [-sin, cos]])
    new_x = cx + (px - cx) * math.cos(theta) + (py - cy) * math.sin(theta)
    new_y = cy - (px - cx) * math.sin(theta) + (py - cy) * math.cos(theta)
    gx, gy = gt["grid_intersections"]["L1"]
    assert abs(gx - new_x) < 0.6, f"L1 ground truth x mismatch: {gx} vs {new_x}"
    assert abs(gy - new_y) < 0.6, f"L1 ground truth y mismatch: {gy} vs {new_y}"
```

Append to TESTS.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement rotation**

Append to `tp_fixtures.py`:

```python
def _rotation_matrix_about_center(theta_deg, cx=360.0, cy=243.0):
    """2D rotation matrix as a 2x3 affine for use with cv2.warpAffine and
    for transforming ground-truth points. Uses cv2's convention: positive
    angle rotates the image CCW on screen (y-down)."""
    import cv2
    return cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)


def _rotate_planes(Y, U, V, theta_deg):
    import cv2
    grey_y = int(tp_chart.GREY_BACKGROUND_Y10)
    grey_c = int(tp_chart.CHROMA_CENTER)
    h, w = Y.shape
    M = _rotation_matrix_about_center(theta_deg, cx=w / 2.0, cy=h / 2.0)
    Y2 = cv2.warpAffine(Y, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=grey_y, flags=cv2.INTER_LINEAR)
    h2, w2 = U.shape
    Mc = _rotation_matrix_about_center(theta_deg, cx=w2 / 2.0, cy=h2 / 2.0)
    U2 = cv2.warpAffine(U, Mc, (w2, h2), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=grey_c, flags=cv2.INTER_LINEAR)
    V2 = cv2.warpAffine(V, Mc, (w2, h2), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=grey_c, flags=cv2.INTER_LINEAR)
    return Y2, U2, V2


def _apply_rotation_to_ground_truth(gt, theta_deg, cx=360.0, cy=243.0):
    M = _rotation_matrix_about_center(theta_deg, cx, cy)
    def _t(pt):
        x, y = pt
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return (float(nx), float(ny))
    out = {
        "grid_intersections": {k: _t(v) for k, v in gt["grid_intersections"].items()},
        "triangles": {},
        "cross": {"center": _t(gt["cross"]["center"])},
        "circle": {
            "center": _t(gt["circle"]["center"]),
            "radius": gt["circle"]["radius"],  # rotation preserves radius
        },
    }
    for tid, t in gt["triangles"].items():
        out["triangles"][tid] = {
            "back_corner_1": _t(t["back_corner_1"]),
            "back_corner_2": _t(t["back_corner_2"]),
            "back_midpoint": _t(t["back_midpoint"]),
            "apex":          _t(t["apex"]),
            "apex_visible":  t["apex_visible"],
        }
    return out
```

Extend the public function signature and call site:

```python
def synthesize_with_ground_truth(
    width: int = 720,
    height: int = 486,
    shift: Tuple[int, int] = (0, 0),
    rotation_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    Y, U, V = tp_synthesize.synthesize(width, height)
    gt = _build_identity_ground_truth()
    if rotation_deg != 0.0:
        Y, U, V = _rotate_planes(Y, U, V, rotation_deg)
        gt = _apply_rotation_to_ground_truth(gt, rotation_deg,
                                             cx=width / 2.0, cy=height / 2.0)
    dx, dy = shift
    if dx != 0 or dy != 0:
        Y, U, V = _shift_planes(Y, U, V, dx, dy)
        gt = _apply_shift_to_ground_truth(gt, dx, dy)
    return Y, U, V, gt
```

(Rotation is applied first, then shift — keeping the geometry composable and matching the test's expectation that the shift offsets are applied to the rotated coords.)

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_fixtures.py test_cases/test_tp_fixtures.py
git commit -m "feat(tp_fixtures): add rotation_deg support via cv2.warpAffine

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Add noise + blur support to tp_fixtures

**Files:**
- Modify: `tp_fixtures.py`
- Test:   `test_cases/test_tp_fixtures.py`

`noise_sigma=σ` adds Gaussian noise to Y; `blur_sigma=σ` applies Gaussian blur to Y. Ground truth is unchanged by these degradations (they don't move fiducial centers).

- [ ] **Step 1: Write the failing test**

```python
def test_fixture_noise_only_changes_pixel_values():
    Y0, _, _, gt0 = tp_fixtures.synthesize_with_ground_truth()
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(noise_sigma=20.0)
    # Ground truth identical.
    assert gt == gt0
    # Y differs.
    diff = (Y.astype(np.int32) - Y0.astype(np.int32)).astype(np.float64)
    assert abs(diff.mean()) < 5.0, "noise should be zero-mean on average"
    assert 15.0 < diff.std() < 25.0, f"noise sigma ~20 expected, got {diff.std()}"


def test_fixture_blur_softens_edges_but_preserves_truth():
    Y0, _, _, gt0 = tp_fixtures.synthesize_with_ground_truth()
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(blur_sigma=1.5)
    assert gt == gt0
    # A grid-line pixel was black (Y10=64); after a 1.5σ blur over ~9 px it
    # rises but should still be the local minimum of its neighborhood.
    grid_y = 108
    grid_x = 180
    assert Y[grid_y, grid_x] > tp_chart.BLACK_Y10
    # The blurred grid pixel should still be substantially darker than the
    # grey background, just not full black.
    assert Y[grid_y, grid_x] < tp_chart.GREY_BACKGROUND_Y10 - 50
```

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement noise + blur**

Append to `tp_fixtures.py`:

```python
def _add_noise(Y, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=12345)
    noise = rng.normal(0.0, sigma, size=Y.shape).astype(np.float32)
    out = Y.astype(np.float32) + noise
    np.clip(out, 0, 1023, out=out)
    return out.astype(Y.dtype)


def _blur_y(Y, sigma):
    import cv2
    # Kernel size: 6σ + 1, must be odd
    k = int(round(6 * sigma)) | 1
    if k < 3:
        k = 3
    return cv2.GaussianBlur(Y, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
```

Extend the signature & body:

```python
def synthesize_with_ground_truth(
    width: int = 720,
    height: int = 486,
    shift: Tuple[int, int] = (0, 0),
    rotation_deg: float = 0.0,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    Y, U, V = tp_synthesize.synthesize(width, height)
    gt = _build_identity_ground_truth()
    if rotation_deg != 0.0:
        Y, U, V = _rotate_planes(Y, U, V, rotation_deg)
        gt = _apply_rotation_to_ground_truth(gt, rotation_deg,
                                             cx=width / 2.0, cy=height / 2.0)
    dx, dy = shift
    if dx != 0 or dy != 0:
        Y, U, V = _shift_planes(Y, U, V, dx, dy)
        gt = _apply_shift_to_ground_truth(gt, dx, dy)
    if blur_sigma > 0:
        Y = _blur_y(Y, blur_sigma)
    if noise_sigma > 0:
        Y = _add_noise(Y, noise_sigma)
    return Y, U, V, gt
```

(Blur applied before noise, so the noise stays raw — matches how real-capture noise sits on top of an already-bandlimited signal.)

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_fixtures.py test_cases/test_tp_fixtures.py
git commit -m "feat(tp_fixtures): add noise_sigma + blur_sigma degradation knobs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Add edge-clip support to tp_fixtures

**Files:**
- Modify: `tp_fixtures.py`
- Test:   `test_cases/test_tp_fixtures.py`

`clip_top=N` / `clip_bottom=N` / `clip_left=N` / `clip_right=N` zero out the top/bottom/left/right N rows or columns. Updates `apex_visible` on every triangle whose `apex` is in the clipped region.

- [ ] **Step 1: Write the failing test**

```python
def test_fixture_clip_top_invalidates_TL_TR_apexes():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(clip_top=5)
    # Top 5 rows zeroed.
    assert (Y[:5, :] == 0).all()
    # TL apex was at (60, 3) -- now in the clipped region.
    assert gt["triangles"]["TL"]["apex_visible"] is False
    assert gt["triangles"]["TR"]["apex_visible"] is False
    # Back corners at y=30 are still visible.
    assert gt["triangles"]["TL"]["back_corner_1"] == (50, 30)
    # BL/BR unaffected.
    assert gt["triangles"]["BL"]["apex_visible"] is True
    assert gt["triangles"]["BR"]["apex_visible"] is True


def test_fixture_clip_bottom_invalidates_BL_BR_apexes():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(clip_bottom=5)
    assert (Y[-5:, :] == 0).all()
    # BL apex was at (60, 483) -- now in the clipped region.
    assert gt["triangles"]["BL"]["apex_visible"] is False
    assert gt["triangles"]["BR"]["apex_visible"] is False
    assert gt["triangles"]["TL"]["apex_visible"] is True
```

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement clipping**

Append to `tp_fixtures.py`:

```python
def _apply_clip_to_planes(Y, U, V, clip_top, clip_bottom, clip_left, clip_right):
    h, w = Y.shape
    if clip_top > 0:
        Y[:clip_top, :] = 0
        U[:clip_top, :] = 0
        V[:clip_top, :] = 0
    if clip_bottom > 0:
        Y[h - clip_bottom:, :] = 0
        U[h - clip_bottom:, :] = 0
        V[h - clip_bottom:, :] = 0
    if clip_left > 0:
        Y[:, :clip_left] = 0
        U[:, :clip_left // 2] = 0
        V[:, :clip_left // 2] = 0
    if clip_right > 0:
        Y[:, w - clip_right:] = 0
        U[:, (w - clip_right) // 2:] = 0
        V[:, (w - clip_right) // 2:] = 0
    return Y, U, V


def _apply_clip_to_ground_truth(gt, clip_top, clip_bottom, clip_left, clip_right,
                                width, height):
    """Mark a triangle's apex_visible = False if its apex coord falls in any
    clipped region."""
    out = dict(gt)
    new_tris = {}
    for tid, t in gt["triangles"].items():
        ax, ay = t["apex"]
        visible = t["apex_visible"]
        if visible:
            if clip_top > 0 and ay < clip_top:
                visible = False
            elif clip_bottom > 0 and ay >= height - clip_bottom:
                visible = False
            elif clip_left > 0 and ax < clip_left:
                visible = False
            elif clip_right > 0 and ax >= width - clip_right:
                visible = False
        new_tris[tid] = dict(t, apex_visible=visible)
    out["triangles"] = new_tris
    return out
```

Update the public function signature:

```python
def synthesize_with_ground_truth(
    width: int = 720,
    height: int = 486,
    shift: Tuple[int, int] = (0, 0),
    rotation_deg: float = 0.0,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
    clip_top: int = 0,
    clip_bottom: int = 0,
    clip_left: int = 0,
    clip_right: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    Y, U, V = tp_synthesize.synthesize(width, height)
    gt = _build_identity_ground_truth()
    if rotation_deg != 0.0:
        Y, U, V = _rotate_planes(Y, U, V, rotation_deg)
        gt = _apply_rotation_to_ground_truth(gt, rotation_deg,
                                             cx=width / 2.0, cy=height / 2.0)
    dx, dy = shift
    if dx != 0 or dy != 0:
        Y, U, V = _shift_planes(Y, U, V, dx, dy)
        gt = _apply_shift_to_ground_truth(gt, dx, dy)
    if blur_sigma > 0:
        Y = _blur_y(Y, blur_sigma)
    if noise_sigma > 0:
        Y = _add_noise(Y, noise_sigma)
    if clip_top or clip_bottom or clip_left or clip_right:
        Y, U, V = _apply_clip_to_planes(Y, U, V, clip_top, clip_bottom,
                                        clip_left, clip_right)
        gt = _apply_clip_to_ground_truth(gt, clip_top, clip_bottom,
                                         clip_left, clip_right, width, height)
    return Y, U, V, gt
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_fixtures.py test_cases/test_tp_fixtures.py
git commit -m "feat(tp_fixtures): add edge-clip degradations with apex_visible updates

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Introduce detect_fiducial dispatcher (refactor)

**Files:**
- Modify: `tp_register.py:12-77` (rename `detect_landmark` → `_detect_grid_intersection`; add dispatcher)
- Modify: `tp_register.py:156-200` (use the dispatcher in `register`)
- Modify: `test_cases/test_tp_register.py` (new dispatcher tests; existing `detect_landmark` tests adjusted if needed)

This task is a refactor with no behavioral change. The existing detect_landmark becomes the `grid_intersection` implementation of a new dispatcher. The catalog records now carry a `kind` field (Task 1 already added this for grid landmarks); the dispatcher routes by it.

- [ ] **Step 1: Write the failing dispatcher tests**

Add to `test_cases/test_tp_register.py`:

```python
import numpy as np
import tp_chart
import tp_fixtures
import tp_register


def test_detect_fiducial_dispatches_grid_intersection():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    lm = tp_chart.GRID_LANDMARKS[0]  # L1 at (180, 108)
    det = tp_register.detect_fiducial(Y, lm)
    assert det is not None
    # Existing grid_intersection contract: returns (x, y, confidence)
    x, y, conf = det
    truth_x, truth_y = gt["grid_intersections"]["L1"]
    assert abs(x - truth_x) < 0.5
    assert abs(y - truth_y) < 0.5


def test_detect_fiducial_unknown_kind_raises():
    Y, _, _, _ = tp_fixtures.synthesize_with_ground_truth()
    raised = False
    try:
        tp_register.detect_fiducial(Y, {"id": "?", "kind": "no_such_thing"})
    except ValueError:
        raised = True
    assert raised
```

If `test_cases/test_tp_register.py` does not yet exist, create it with the standard harness shape. Append the new test names to the `TESTS` list.

- [ ] **Step 2: Run; verify failure** — Expected: FAIL — `detect_fiducial` does not exist.

- [ ] **Step 3: Refactor tp_register.py to add the dispatcher**

In `tp_register.py`, **rename** `detect_landmark` to `_detect_grid_intersection` and change its signature to take the fiducial dict directly:

```python
def _detect_grid_intersection(Y, fid):
    """Existing grid-intersection detector. fid carries ideal_x, ideal_y,
    search_window_px."""
    return _grid_intersection_impl(
        Y, fid["ideal_x"], fid["ideal_y"], fid["search_window_px"]
    )


def _grid_intersection_impl(Y, ideal_x, ideal_y, search_window_px):
    """Implementation extracted verbatim from the original detect_landmark."""
    # [exact body of the old detect_landmark function, lines 36-76 of the
    #  current tp_register.py — column/row projection, sub-pixel centroid]
    h, w = Y.shape
    half = search_window_px // 2
    x0 = max(0, ideal_x - half)
    y0 = max(0, ideal_y - half)
    x1 = min(w, ideal_x + half)
    y1 = min(h, ideal_y + half)
    win = Y[y0:y1, x0:x1].astype(np.float32)
    if win.size == 0:
        return None
    threshold = 0.3 * tp_chart.GREY_BACKGROUND_Y10
    dark_mask = win < threshold
    dark_count = int(dark_mask.sum())
    if dark_count < 4:
        return None
    confidence = dark_count / win.size
    col_proj = dark_mask.sum(axis=0).astype(np.float32)
    row_proj = dark_mask.sum(axis=1).astype(np.float32)
    col_med = float(np.median(col_proj))
    row_med = float(np.median(row_proj))
    col_high = col_proj > col_med
    row_high = row_proj > row_med
    if col_high.sum() < 1 or row_high.sum() < 1:
        return None
    col_idxs = np.where(col_high)[0].astype(np.float32)
    cx = float((col_proj[col_high] * col_idxs).sum() / col_proj[col_high].sum())
    row_idxs = np.where(row_high)[0].astype(np.float32)
    cy = float((row_proj[row_high] * row_idxs).sum() / row_proj[row_high].sum())
    return (x0 + cx, y0 + cy, confidence)


def detect_fiducial(Y, fid):
    """Dispatch by fid['kind'] to the right detector implementation.
    Returns a detector-specific value (kind-dependent shape) or None."""
    kind = fid["kind"]
    if kind == "grid_intersection":
        return _detect_grid_intersection(Y, fid)
    # Stage 2 detectors added in subsequent tasks (Task 15-17).
    raise ValueError(f"unknown fiducial kind: {kind}")
```

Remove the old top-level `detect_landmark` definition (lines 12-76); the rest of `tp_register.py` (fit_affine, register, etc.) stays unchanged for now.

Update `register()` (around line 156) to call the dispatcher:

```python
def register(Y):
    detected = []
    ideal = []
    detected_lm_ids = []
    for lm in tp_chart.GRID_LANDMARKS:
        det = detect_fiducial(Y, lm)
        if det is None:
            continue
        dx, dy, _ = det
        detected.append((dx, dy))
        ideal.append((lm["ideal_x"], lm["ideal_y"]))
        detected_lm_ids.append(lm["id"])
    # [rest of register() unchanged]
```

(Diff: just the call from `detect_landmark(...)` to `detect_fiducial(Y, lm)`.)

- [ ] **Step 4: Run all tests; verify pass**

```bash
for f in test_cases/test_tp_*.py; do python "$f" || exit 1; done
```

Expected: All pass. The dispatcher tests pass, and `register()` still works on synthesized identity (existing measure tests pass).

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "refactor(tp_register): introduce detect_fiducial dispatcher

Renames detect_landmark to _detect_grid_intersection (now routed through
the dispatcher by fid['kind']). Behavioral no-op for Stage 1; prepares
for Stage 2 boundary-triangle / cross / circle detectors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Implement _detect_boundary_triangle

**Files:**
- Modify: `tp_register.py`
- Test:   `test_cases/test_tp_register.py`

Back-corner-first triangle detector. Returns `{back_corner_1, back_corner_2, back_midpoint, apex_inferred, apex_detected, confidence}` or `None`.

- [ ] **Step 1: Write the failing tests (fixture-driven accuracy contract)**

```python
def test_detect_boundary_triangle_identity():
    """Identity variant: back corners within 0.5 px; apex within 1.0 px."""
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    truth = gt["triangles"]["TL"]
    bc1_x, bc1_y = result["back_corner_1"]
    assert abs(bc1_x - truth["back_corner_1"][0]) < 0.5
    assert abs(bc1_y - truth["back_corner_1"][1]) < 0.5
    bc2_x, bc2_y = result["back_corner_2"]
    assert abs(bc2_x - truth["back_corner_2"][0]) < 0.5
    assert abs(bc2_y - truth["back_corner_2"][1]) < 0.5
    apex_x, apex_y = result["apex_inferred"]
    assert abs(apex_x - truth["apex"][0]) < 1.0
    assert abs(apex_y - truth["apex"][1]) < 1.0


def test_detect_boundary_triangle_noise_sigma_20():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(noise_sigma=20.0)
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    truth = gt["triangles"]["TL"]
    for key in ("back_corner_1", "back_corner_2"):
        dx = result[key][0] - truth[key][0]
        dy = result[key][1] - truth[key][1]
        assert (dx ** 2 + dy ** 2) ** 0.5 < 1.0, f"{key} err > 1.0 px"


def test_detect_boundary_triangle_blur_sigma_1_5():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(blur_sigma=1.5)
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    truth = gt["triangles"]["TL"]
    for key in ("back_corner_1", "back_corner_2"):
        dx = result[key][0] - truth[key][0]
        dy = result[key][1] - truth[key][1]
        assert (dx ** 2 + dy ** 2) ** 0.5 < 1.5, f"{key} err > 1.5 px"


def test_detect_boundary_triangle_clip_top_apex_inferred_only():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(clip_top=5)
    tri = next(t for t in tp_chart.BOUNDARY_TRIANGLES if t["id"] == "TL")
    result = tp_register.detect_fiducial(Y, tri)
    assert result is not None
    # Back corners still detected to threshold.
    truth = gt["triangles"]["TL"]
    for key in ("back_corner_1", "back_corner_2"):
        dx = result[key][0] - truth[key][0]
        dy = result[key][1] - truth[key][1]
        assert (dx ** 2 + dy ** 2) ** 0.5 < 0.5
    # apex_detected is None (clipped), apex_inferred is computed from
    # back_midpoint + chart offset.
    assert result["apex_detected"] is None
    assert result["apex_inferred"] is not None
```

Append all four to `TESTS`.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement _detect_boundary_triangle**

Append to `tp_register.py`:

```python
def _detect_boundary_triangle(Y, fid):
    """Detect a boundary triangle by its back corners.

    fid carries: ideal_back_corner_1, ideal_back_corner_2,
                 ideal_back_midpoint, ideal_apex, orientation,
                 search_window_px.

    Algorithm:
      1. Crop a search window centered on the ideal back_midpoint.
      2. Threshold dark pixels: Y10 < 0.3 * GREY_BACKGROUND_Y10.
      3. Reject if dark_count < 50.
      4. Find the back edge based on orientation:
           apex_up    -> back row = bottom-most row containing dark pixels
           apex_down  -> back row = top-most row
           apex_left  -> back col = right-most column
           apex_right -> back col = left-most column
      5. Find the two extreme dark pixels along the back edge -> back corners.
         Sub-pixel: weighted-centroid the dark pixels in a 3-px-wide strip
         along the back edge near each extreme.
      6. back_midpoint = midpoint of back corners.
      7. apex_inferred = back_midpoint + (fid.ideal_apex - fid.ideal_back_midpoint).
      8. apex_detected: extreme dark pixel in the apex-direction; None if it
         lies outside chart_spec_distance +/- 2 px.
    """
    h, w = Y.shape
    bm_x, bm_y = fid["ideal_back_midpoint"]
    half = fid["search_window_px"] // 2
    x0 = max(0, int(bm_x) - half)
    y0 = max(0, int(bm_y) - half)
    x1 = min(w, int(bm_x) + half)
    y1 = min(h, int(bm_y) + half)
    win = Y[y0:y1, x0:x1].astype(np.float32)
    if win.size == 0:
        return None

    threshold = 0.3 * tp_chart.GREY_BACKGROUND_Y10
    dark_mask = win < threshold
    dark_count = int(dark_mask.sum())
    if dark_count < 50:
        return None
    confidence = min(1.0, dark_count / win.size)

    orient = fid["orientation"]
    dark_ys, dark_xs = np.where(dark_mask)
    if len(dark_ys) == 0:
        return None

    if orient == "apex_up":
        # Back = bottom row containing dark pixels.
        back_local_y = int(dark_ys.max())
        apex_dir = -1  # apex is at smaller y (up)
        # Strip: rows back_local_y-1 .. back_local_y+1 in the window.
        strip = np.zeros_like(dark_mask)
        strip[max(0, back_local_y - 1):back_local_y + 1] = dark_mask[max(0, back_local_y - 1):back_local_y + 1]
    elif orient == "apex_down":
        back_local_y = int(dark_ys.min())
        apex_dir = +1
        strip = np.zeros_like(dark_mask)
        strip[back_local_y:back_local_y + 2] = dark_mask[back_local_y:back_local_y + 2]
    else:
        raise ValueError(f"unsupported orientation: {orient}")

    # Back corners: leftmost and rightmost dark pixels in the strip,
    # sub-pixel via weighted centroid in a 3 px horizontal window.
    strip_ys, strip_xs = np.where(strip)
    if len(strip_xs) < 2:
        return None
    left_extreme = int(strip_xs.min())
    right_extreme = int(strip_xs.max())

    def _subpixel_x(target_x):
        x_lo = max(0, target_x - 1)
        x_hi = min(strip.shape[1], target_x + 2)
        local = strip[:, x_lo:x_hi]
        ys, xs = np.where(local)
        if len(xs) == 0:
            return float(target_x)
        return float(xs.mean() + x_lo)

    bc1_lx = _subpixel_x(left_extreme)
    bc2_lx = _subpixel_x(right_extreme)
    bc1_ly = float(back_local_y)
    bc2_ly = float(back_local_y)

    bc1 = (x0 + bc1_lx, y0 + bc1_ly)
    bc2 = (x0 + bc2_lx, y0 + bc2_ly)
    back_midpoint = ((bc1[0] + bc2[0]) / 2.0, (bc1[1] + bc2[1]) / 2.0)

    # Apex inferred: back_midpoint + chart-spec offset
    ideal_apex = fid["ideal_apex"]
    ideal_bm = fid["ideal_back_midpoint"]
    offset = (ideal_apex[0] - ideal_bm[0], ideal_apex[1] - ideal_bm[1])
    apex_inferred = (back_midpoint[0] + offset[0],
                     back_midpoint[1] + offset[1])

    # Apex detected: extreme dark pixel in apex direction.
    if orient == "apex_up":
        candidate_local_y = int(dark_ys.min())
    else:  # apex_down
        candidate_local_y = int(dark_ys.max())
    chart_spec_dist = abs(offset[1])
    observed_dist = abs(candidate_local_y - back_local_y)
    if abs(observed_dist - chart_spec_dist) > 2:
        apex_detected = None
    else:
        # Sub-pixel x of the apex via centroid of dark pixels at the apex row.
        apex_strip = (dark_ys == candidate_local_y)
        if apex_strip.sum() == 0:
            apex_detected = None
        else:
            apex_lx = float(dark_xs[apex_strip].mean())
            apex_detected = (x0 + apex_lx, y0 + float(candidate_local_y))

    return {
        "back_corner_1": bc1,
        "back_corner_2": bc2,
        "back_midpoint": back_midpoint,
        "apex_inferred": apex_inferred,
        "apex_detected": apex_detected,
        "confidence": confidence,
    }
```

Extend the dispatcher (replace the `raise ValueError`):

```python
def detect_fiducial(Y, fid):
    kind = fid["kind"]
    if kind == "grid_intersection":
        return _detect_grid_intersection(Y, fid)
    if kind == "boundary_triangle":
        return _detect_boundary_triangle(Y, fid)
    raise ValueError(f"unknown fiducial kind: {kind}")
```

- [ ] **Step 4: Run; verify pass**

If a test fails on accuracy, examine which knob (sub-pixel weighting, strip width) needs tightening. Don't loosen the thresholds — they're part of the design contract.

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(tp_register): add _detect_boundary_triangle (back-corner-first)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Implement _detect_registration_cross

**Files:**
- Modify: `tp_register.py`
- Test:   `test_cases/test_tp_register.py`

Template-match the synthesized 24×24 cross template against the search window; do parabolic sub-pixel interpolation on the SQDIFF minimum.

- [ ] **Step 1: Write the failing tests**

```python
def test_detect_registration_cross_identity():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    rc = tp_chart.REGISTRATION_CROSS
    result = tp_register.detect_fiducial(Y, rc)
    assert result is not None
    truth_x, truth_y = gt["cross"]["center"]
    assert abs(result["x"] - truth_x) < 0.2
    assert abs(result["y"] - truth_y) < 0.2
    assert result["confidence"] > 0.6


def test_detect_registration_cross_subpixel_shift():
    """Use a 5-px integer shift since the synthesizer paints into integer
    pixels; with a 5-px shift the cross is still in the search window and the
    detector should track it within 0.2 px."""
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(shift=(5, 7))
    rc = tp_chart.REGISTRATION_CROSS
    result = tp_register.detect_fiducial(Y, rc)
    assert result is not None
    truth_x, truth_y = gt["cross"]["center"]
    assert abs(result["x"] - truth_x) < 0.5
    assert abs(result["y"] - truth_y) < 0.5


def test_detect_registration_cross_noise_sigma_20():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(noise_sigma=20.0)
    rc = tp_chart.REGISTRATION_CROSS
    result = tp_register.detect_fiducial(Y, rc)
    assert result is not None
    truth_x, truth_y = gt["cross"]["center"]
    err = ((result["x"] - truth_x) ** 2 + (result["y"] - truth_y) ** 2) ** 0.5
    assert err < 0.6, f"noise σ=20 produced err {err:.2f} px > 0.6 px threshold"
```

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement _detect_registration_cross**

Append to `tp_register.py`:

```python
_RC_TEMPLATE_CACHE = None


def _build_registration_cross_template():
    """Build a 24x24 template matching tp_chart.REGISTRATION_CROSS: black box
    with centered white + (3 px arm thickness, 17 px arm length)."""
    rc = tp_chart.REGISTRATION_CROSS
    size = rc["box_size_px"]
    t = np.full((size, size), 0, dtype=np.float32)  # black box
    half = size // 2
    half_len = rc["ideal_arm_len_px"] // 2
    half_th = rc["ideal_arm_thickness_px"] // 2
    # White arms (scaled 0..255 for matchTemplate).
    t[half - half_th:half + half_th + 1, half - half_len:half + half_len + 1] = 255.0
    t[half - half_len:half + half_len + 1, half - half_th:half + half_th + 1] = 255.0
    return t


def _registration_cross_template():
    global _RC_TEMPLATE_CACHE
    if _RC_TEMPLATE_CACHE is None:
        _RC_TEMPLATE_CACHE = _build_registration_cross_template()
    return _RC_TEMPLATE_CACHE


def _detect_registration_cross(Y, fid):
    import cv2
    h, w = Y.shape
    half = fid["search_window_px"] // 2
    cx, cy = fid["ideal_x"], fid["ideal_y"]
    x0 = max(0, cx - half); y0 = max(0, cy - half)
    x1 = min(w, cx + half); y1 = min(h, cy + half)
    win = Y[y0:y1, x0:x1].astype(np.float32)
    # Normalize win to 0..255 for matchTemplate consistency with the template.
    win_n = np.clip((win - tp_chart.BLACK_Y10) / (tp_chart.WHITE_Y10 - tp_chart.BLACK_Y10), 0, 1) * 255.0
    win_n = win_n.astype(np.float32)
    template = _registration_cross_template()
    if win_n.shape[0] < template.shape[0] or win_n.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(win_n, template, cv2.TM_SQDIFF_NORMED)
    min_val, _max, min_loc, _maxloc = cv2.minMaxLoc(result)
    confidence = float(1.0 - min_val)
    if confidence < 0.4:
        return None
    # Parabolic sub-pixel on the 3x3 around min_loc.
    px, py = min_loc
    rh, rw = result.shape
    if 1 <= px < rw - 1 and 1 <= py < rh - 1:
        dx = _parabolic_subpixel(result[py, px - 1], result[py, px], result[py, px + 1])
        dy = _parabolic_subpixel(result[py - 1, px], result[py, px], result[py + 1, px])
    else:
        dx = 0.0
        dy = 0.0
    template_h, template_w = template.shape
    # The matched position is the top-left of the template; the cross center
    # is template_w/2, template_h/2 px below + right.
    cross_x_local = px + dx + template_w / 2.0
    cross_y_local = py + dy + template_h / 2.0
    cross_x = x0 + cross_x_local
    cross_y = y0 + cross_y_local
    # Measure arm lengths along center lines.
    h_arm_len = _count_bright_run(Y[int(round(cross_y)), :], int(round(cross_x)))
    v_arm_len = _count_bright_run(Y[:, int(round(cross_x))], int(round(cross_y)))
    return {
        "x": float(cross_x),
        "y": float(cross_y),
        "h_arm_len_px": float(h_arm_len),
        "v_arm_len_px": float(v_arm_len),
        "confidence": confidence,
    }


def _parabolic_subpixel(left, center, right):
    """1D parabolic fit; returns offset in [-1, +1] from center."""
    denom = (left + right - 2.0 * center)
    if abs(denom) < 1e-9:
        return 0.0
    return float(0.5 * (left - right) / denom)


def _count_bright_run(line, center_idx):
    """Walk left and right from center_idx along `line`, counting pixels with
    Y10 > 700 contiguously. Returns total run length including center."""
    n = len(line)
    threshold = 700
    count = 1 if line[center_idx] > threshold else 0
    for i in range(center_idx - 1, -1, -1):
        if line[i] > threshold:
            count += 1
        else:
            break
    for i in range(center_idx + 1, n):
        if line[i] > threshold:
            count += 1
        else:
            break
    return count
```

Extend the dispatcher:

```python
def detect_fiducial(Y, fid):
    kind = fid["kind"]
    if kind == "grid_intersection":
        return _detect_grid_intersection(Y, fid)
    if kind == "boundary_triangle":
        return _detect_boundary_triangle(Y, fid)
    if kind == "registration_cross":
        return _detect_registration_cross(Y, fid)
    raise ValueError(f"unknown fiducial kind: {kind}")
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(tp_register): add _detect_registration_cross with sub-pixel match

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Implement _detect_black_circle

**Files:**
- Modify: `tp_register.py`
- Test:   `test_cases/test_tp_register.py`

Annular search → threshold → `cv2.fitEllipse` on the dark-pixel point cloud.

- [ ] **Step 1: Write the failing tests**

```python
def test_detect_black_circle_identity():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    bc = tp_chart.BLACK_CIRCLE
    result = tp_register.detect_fiducial(Y, bc)
    assert result is not None
    tcx, tcy = gt["circle"]["center"]
    assert abs(result["cx"] - tcx) < 0.5
    assert abs(result["cy"] - tcy) < 0.5
    truth_r = gt["circle"]["radius"]
    assert abs(result["rx"] - truth_r) < 0.5
    assert abs(result["ry"] - truth_r) < 0.5
    assert result["fit_rms"] < 1.0


def test_detect_black_circle_noise_sigma_20():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(noise_sigma=20.0)
    bc = tp_chart.BLACK_CIRCLE
    result = tp_register.detect_fiducial(Y, bc)
    assert result is not None
    tcx, tcy = gt["circle"]["center"]
    err = ((result["cx"] - tcx) ** 2 + (result["cy"] - tcy) ** 2) ** 0.5
    assert err < 1.0
    truth_r = gt["circle"]["radius"]
    assert abs(result["rx"] - truth_r) < 1.5
    assert abs(result["ry"] - truth_r) < 1.5
```

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement _detect_black_circle**

Append to `tp_register.py`:

```python
def _detect_black_circle(Y, fid):
    import cv2
    h, w = Y.shape
    cx_ideal = fid["ideal_cx"]
    cy_ideal = fid["ideal_cy"]
    r_ideal = fid["expected_radius_px"]
    band = fid["search_band_px"]
    # Build an annular mask: pixels with radius in [r-band, r+band].
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx_ideal) ** 2 + (yy - cy_ideal) ** 2)
    annulus = (dist >= (r_ideal - band)) & (dist <= (r_ideal + band))
    # Threshold dark pixels within annulus.
    threshold = 0.3 * tp_chart.GREY_BACKGROUND_Y10
    dark = (Y < threshold) & annulus
    dark_count = int(dark.sum())
    if dark_count < 100:
        return None
    ys, xs = np.where(dark)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) < 5:
        return None
    (cx, cy), (axis_a, axis_b), rot_deg = cv2.fitEllipse(pts)
    rx = min(axis_a, axis_b) / 2.0
    ry = max(axis_a, axis_b) / 2.0
    # fit_rms: mean perpendicular distance from each point to the fitted
    # ellipse. Approximate via the radial residual at each point's angle.
    theta = np.arctan2(ys - cy, xs - cx)
    rot_rad = np.deg2rad(rot_deg)
    # Express points in the ellipse's local frame.
    cos_t = np.cos(theta - rot_rad)
    sin_t = np.sin(theta - rot_rad)
    expected_r = (rx * ry) / np.sqrt((ry * cos_t) ** 2 + (rx * sin_t) ** 2 + 1e-9)
    actual_r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    fit_rms = float(np.sqrt(((actual_r - expected_r) ** 2).mean()))
    if fit_rms > 5.0:
        return None
    confidence = max(0.0, 1.0 - fit_rms / 10.0)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "rx": float(rx),
        "ry": float(ry),
        "rotation_deg": float(rot_deg),
        "fit_rms": fit_rms,
        "confidence": confidence,
    }
```

Extend the dispatcher:

```python
def detect_fiducial(Y, fid):
    kind = fid["kind"]
    if kind == "grid_intersection":
        return _detect_grid_intersection(Y, fid)
    if kind == "boundary_triangle":
        return _detect_boundary_triangle(Y, fid)
    if kind == "registration_cross":
        return _detect_registration_cross(Y, fid)
    if kind == "black_circle":
        return _detect_black_circle(Y, fid)
    raise ValueError(f"unknown fiducial kind: {kind}")
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(tp_register): add _detect_black_circle via cv2.fitEllipse

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Implement detect_geometry orchestrator (+ derivation calcs)

**Files:**
- Modify: `tp_register.py`
- Test:   `test_cases/test_tp_register.py`

Runs all Stage 2 detectors and computes derived measurements (active_picture_box, picture_offset_from_ideal, corner_skew, arrow_tip_coords, clip_detected, cross_offset_from_ideal, aperture_symmetry, aspect_ratio_check, diameter_vs_picture_height, circle_fit_rms).

- [ ] **Step 1: Write the failing test**

```python
def test_detect_geometry_returns_full_block_on_clean_fixture():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    M_identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    geom = tp_register.detect_geometry(Y, M_identity)
    assert geom is not None
    assert "fiducials" in geom
    assert "derived" in geom
    # Triangles
    tris = geom["fiducials"]["triangles"]
    assert set(tris.keys()) == {"TL", "TR", "BL", "BR"}
    # Derived box should match the chart's ideal picture box closely.
    box = geom["derived"]["active_picture_box"]
    assert abs(box["top"] - 3.0) < 1.5      # apex_inferred for TL/TR ~ y=3
    assert abs(box["bottom"] - 483.0) < 1.5
    assert abs(box["left"] - 60.0) < 1.5    # apex_inferred for TL/BL ~ x=60
    assert abs(box["right"] - 660.0) < 1.5
    # Aspect check should be near 1.0 (square circle).
    assert abs(geom["derived"]["aspect_ratio_check"] - 1.0) < 0.005
    # Aperture symmetry near 1.0 (cross is symmetric).
    assert abs(geom["derived"]["aperture_symmetry"] - 1.0) < 0.1


def test_detect_geometry_clip_top_marks_apexes_invisible():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth(clip_top=5)
    M_identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    geom = tp_register.detect_geometry(Y, M_identity)
    assert geom["derived"]["clip_detected"]["TL"]["apex_visible"] is False
    assert geom["derived"]["clip_detected"]["TR"]["apex_visible"] is False
    assert geom["derived"]["clip_detected"]["BL"]["apex_visible"] is True
```

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement detect_geometry**

Append to `tp_register.py`:

```python
def _apply_affine_pt(M, x, y):
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def detect_geometry(Y, M_initial):
    """Run all Stage 2 detectors and compute derived geometry.

    M_initial: 2x3 affine mapping ideal -> capture coords. Used to project
               each fiducial's ideal search-window center into capture
               coords for the detector's search.
    """
    h, w = Y.shape
    fiducials = {"triangles": {}, "cross": None, "circle": None}

    # Triangles
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        # Shift the ideal back_midpoint through M_initial so the detector
        # searches the right region in capture coords.
        bm = tri["ideal_back_midpoint"]
        proj_x, proj_y = _apply_affine_pt(M_initial, bm[0], bm[1])
        tri_capture = dict(tri, ideal_back_midpoint=(int(round(proj_x)),
                                                    int(round(proj_y))),
                           # also shift ideal_apex for the apex_detected check
                           ideal_apex=tuple(_apply_affine_pt(M_initial,
                                                            tri["ideal_apex"][0],
                                                            tri["ideal_apex"][1])))
        result = _detect_boundary_triangle(Y, tri_capture)
        fiducials["triangles"][tri["id"]] = result

    # Cross
    rc = tp_chart.REGISTRATION_CROSS
    proj_x, proj_y = _apply_affine_pt(M_initial, rc["ideal_x"], rc["ideal_y"])
    rc_capture = dict(rc, ideal_x=int(round(proj_x)), ideal_y=int(round(proj_y)))
    fiducials["cross"] = _detect_registration_cross(Y, rc_capture)

    # Circle
    bc = tp_chart.BLACK_CIRCLE
    proj_x, proj_y = _apply_affine_pt(M_initial, bc["ideal_cx"], bc["ideal_cy"])
    bc_capture = dict(bc, ideal_cx=float(proj_x), ideal_cy=float(proj_y))
    fiducials["circle"] = _detect_black_circle(Y, bc_capture)

    derived = _derive_geometry(fiducials, M_initial, w, h)
    return {"fiducials": fiducials, "derived": derived}


def _derive_geometry(fiducials, M, width, height):
    tris = fiducials["triangles"]
    derived = {}

    def _apex_inferred(tid):
        t = tris.get(tid)
        return t["apex_inferred"] if t is not None else None

    TL = _apex_inferred("TL"); TR = _apex_inferred("TR")
    BL = _apex_inferred("BL"); BR = _apex_inferred("BR")

    if all(p is not None for p in (TL, TR, BL, BR)):
        active_picture_box = {
            "top":    (TL[1] + TR[1]) / 2.0,
            "bottom": (BL[1] + BR[1]) / 2.0,
            "left":   (TL[0] + BL[0]) / 2.0,
            "right":  (TR[0] + BR[0]) / 2.0,
        }
        derived["active_picture_box"] = active_picture_box
        derived["picture_extent_px"] = {
            "width":  active_picture_box["right"]  - active_picture_box["left"],
            "height": active_picture_box["bottom"] - active_picture_box["top"],
        }
        # Compare to ideal picture corners projected through M.
        corners = tp_chart.IDEAL_PICTURE_BOX_CORNERS
        ideal_in_cap = [_apply_affine_pt(M, x, y) for x, y in corners]
        ideal_left = min(p[0] for p in ideal_in_cap)
        ideal_top  = min(p[1] for p in ideal_in_cap)
        derived["picture_offset_from_ideal"] = {
            "dx": active_picture_box["left"] - ideal_left,
            "dy": active_picture_box["top"]  - ideal_top,
        }
        top_w = TR[0] - TL[0]
        bot_w = BR[0] - BL[0]
        left_h = BL[1] - TL[1]
        right_h = BR[1] - TR[1]
        derived["corner_skew_px"] = {
            "top_vs_bottom_width_diff":  abs(top_w - bot_w),
            "left_vs_right_height_diff": abs(left_h - right_h),
        }
        derived["arrow_tip_coords"] = {
            "TL": TL, "TR": TR, "BL": BL, "BR": BR,
        }
    else:
        for key in ("active_picture_box", "picture_extent_px",
                    "picture_offset_from_ideal", "corner_skew_px",
                    "arrow_tip_coords"):
            derived[key] = None

    # Clip detection per triangle.
    clip = {}
    for tid in ("TL", "TR", "BL", "BR"):
        t = tris.get(tid)
        if t is None:
            clip[tid] = {"apex_visible": False, "clip_px": None,
                         "interpretation": "triangle not detected"}
            continue
        apex_visible = t["apex_detected"] is not None
        clip_px = 0.0
        interp = "no clip detected"
        ax, ay = t["apex_inferred"]
        orient = next(tt["orientation"] for tt in tp_chart.BOUNDARY_TRIANGLES
                      if tt["id"] == tid)
        if not apex_visible:
            if orient == "apex_up":
                clip_px = max(0.0, -ay)
                # Also flag clipping if apex is within 1 px of edge.
                if ay < 1:
                    clip_px = max(clip_px, 1.0 - ay)
                interp = f"top edge clipped ~{clip_px:.0f} px" if clip_px > 0 else "apex not detected"
            elif orient == "apex_down":
                clip_px = max(0.0, ay - (height - 1))
                if ay > height - 2:
                    clip_px = max(clip_px, ay - (height - 2))
                interp = f"bottom edge clipped ~{clip_px:.0f} px" if clip_px > 0 else "apex not detected"
        clip[tid] = {
            "apex_visible": apex_visible,
            "clip_px": float(clip_px),
            "interpretation": interp,
        }
    derived["clip_detected"] = clip

    # Cross-derived.
    rc = fiducials["cross"]
    if rc is not None:
        ideal_cx, ideal_cy = _apply_affine_pt(
            M, tp_chart.REGISTRATION_CROSS["ideal_x"],
               tp_chart.REGISTRATION_CROSS["ideal_y"])
        derived["cross_offset_from_ideal"] = [rc["x"] - ideal_cx,
                                              rc["y"] - ideal_cy]
        h_arm = rc["h_arm_len_px"]; v_arm = rc["v_arm_len_px"]
        if max(h_arm, v_arm) > 0:
            derived["aperture_symmetry"] = float(min(h_arm, v_arm) / max(h_arm, v_arm))
        else:
            derived["aperture_symmetry"] = None
    else:
        derived["cross_offset_from_ideal"] = None
        derived["aperture_symmetry"] = None

    # Circle-derived.
    bc = fiducials["circle"]
    if bc is not None:
        rx, ry = bc["rx"], bc["ry"]
        derived["aspect_ratio_check"] = float(min(rx, ry) / max(rx, ry))
        if derived.get("picture_extent_px") is not None:
            derived["diameter_vs_picture_height"] = float(
                2.0 * max(rx, ry) / derived["picture_extent_px"]["height"]
            )
        else:
            derived["diameter_vs_picture_height"] = None
        derived["circle_fit_rms"] = float(bc["fit_rms"])
    else:
        derived["aspect_ratio_check"] = None
        derived["diameter_vs_picture_height"] = None
        derived["circle_fit_rms"] = None

    return derived
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(tp_register): add detect_geometry orchestrator + derivation calcs

Computes active_picture_box, picture_offset, corner_skew, clip_detected,
cross/circle derived metrics from the 4 boundary triangles + cross +
circle detections.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Implement register_with_geometry (sequential-with-feedback)

**Files:**
- Modify: `tp_register.py`
- Test:   `test_cases/test_tp_register.py`

Runs Stage 1 `register()`, then `detect_geometry()`, then re-fits the affine using GRID_LANDMARKS + the 8 boundary-triangle back corners + cross center as anchors. Reports both inlier counts and the residuals from each fit.

- [ ] **Step 1: Write the failing test**

```python
def test_register_with_geometry_increases_inlier_count():
    Y, _, _, gt = tp_fixtures.synthesize_with_ground_truth()
    result = tp_register.register_with_geometry(Y)
    assert result["initial"]["affine_matrix"] is not None
    assert result["geometry"] is not None
    assert result["final"]["affine_matrix"] is not None
    # The final fit has more inliers than the initial fit.
    assert result["final"]["inliers"] >= result["initial"]["inliers"]
    # Residuals on the final fit are no worse than the initial fit's.
    assert (result["final"]["residuals_px"]["mean"]
            <= result["initial"]["residuals_px"]["mean"] + 0.1)


def test_register_with_geometry_preserves_identity_on_synthesized():
    Y, _, _, _ = tp_fixtures.synthesize_with_ground_truth()
    result = tp_register.register_with_geometry(Y)
    # Final affine is close to identity.
    M = result["final"]["affine_matrix"]
    assert abs(M[0, 0] - 1.0) < 0.005
    assert abs(M[1, 1] - 1.0) < 0.005
    assert abs(M[0, 1]) < 0.005
    assert abs(M[1, 0]) < 0.005
    assert abs(M[0, 2]) < 0.5
    assert abs(M[1, 2]) < 0.5
```

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement register_with_geometry**

Append to `tp_register.py`:

```python
def register_with_geometry(Y):
    """Sequential-with-feedback registration.

    Returns:
        {
            "initial":  <Stage 1 register() result>,
            "geometry": <detect_geometry() result>,
            "final":    <fit_affine() result on grids+geometry anchors>,
            "anchors_added": [<ids appended after Stage 1>],
        }
    """
    initial = register(Y)
    if initial["affine_matrix"] is None:
        return {"initial": initial, "geometry": None, "final": initial,
                "anchors_added": []}
    M_initial = initial["affine_matrix"]
    geometry = detect_geometry(Y, M_initial)

    # Collect all anchors: grids that survived initial + triangle back corners
    # + cross.
    detected_pts = []
    ideal_pts = []
    anchors_added = []
    # Initial grids -- re-detect to capture in capture coords (we don't have
    # them cached from register()). Simpler: re-run grid detection and use
    # the survivors.
    for lm in tp_chart.GRID_LANDMARKS:
        det = detect_fiducial(Y, lm)
        if det is None:
            continue
        dx, dy, _ = det
        detected_pts.append((dx, dy))
        ideal_pts.append((lm["ideal_x"], lm["ideal_y"]))

    # Triangle back corners.
    tris = geometry["fiducials"]["triangles"]
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        t = tris.get(tri["id"])
        if t is None:
            continue
        for cap_key, ideal_key, suffix in (
            ("back_corner_1", "ideal_back_corner_1", "bc1"),
            ("back_corner_2", "ideal_back_corner_2", "bc2"),
        ):
            detected_pts.append(t[cap_key])
            ideal_pts.append(tri[ideal_key])
            anchors_added.append(f"{tri['id']}.{suffix}")

    # Cross center.
    rc = geometry["fiducials"]["cross"]
    if rc is not None:
        detected_pts.append((rc["x"], rc["y"]))
        ideal_pts.append((tp_chart.REGISTRATION_CROSS["ideal_x"],
                          tp_chart.REGISTRATION_CROSS["ideal_y"]))
        anchors_added.append("RC")

    final = fit_affine(np.asarray(detected_pts, dtype=np.float32),
                       np.asarray(ideal_pts, dtype=np.float32))
    final["total"] = len(detected_pts)
    # Quality flag follows the existing rules.
    if final["affine_matrix"] is None or final["inliers"] < MIN_INLIERS:
        final["quality_flag"] = "failed"
        final["quality_reason"] = "RANSAC failed or too few inliers"
    elif (final["residuals_px"]["mean"] > RESIDUAL_OK_MEAN_PX
          or final["residuals_px"]["max"] > RESIDUAL_OK_MAX_PX):
        final["quality_flag"] = "warn"
        final["quality_reason"] = "residuals exceed threshold"
    else:
        final["quality_flag"] = "ok"
        final["quality_reason"] = None

    return {
        "initial": initial,
        "geometry": geometry,
        "final": final,
        "anchors_added": anchors_added,
    }
```

- [ ] **Step 4: Run; verify pass**

- [ ] **Step 5: Commit**

```bash
git add tp_register.py test_cases/test_tp_register.py
git commit -m "feat(tp_register): add register_with_geometry sequential-with-feedback

Runs Stage 1 register(), then detect_geometry(), then re-fits affine
with grids + triangle back corners + cross center as anchors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: tp_measure: call register_with_geometry; embed geometry block in JSON

**Files:**
- Modify: `tp_measure.py:264-322`
- Test:   `test_cases/test_tp_measure.py`

Replace the single `tp_register.register()` call with `register_with_geometry()`. Use the **final** affine for sampling. Embed the geometry block at top-level of the JSON. The existing `_meta.registration` field now reflects the **final** fit (matching what was used for sampling).

- [ ] **Step 1: Write the failing test**

Add to `test_cases/test_tp_measure.py`:

```python
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
```

Append to `TESTS_TMPDIR`.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Modify tp_measure.py**

In `tp_measure.py`, replace the `register` call and JSON-building section. Replace lines 277-322 (the `tp_register.register(Y_p)` call through `return ...`) with:

```python
    reg_full = tp_register.register_with_geometry(Y_p)
    reg = reg_full["final"]
    meta["registration"] = {
        "affine": reg["affine_matrix"].tolist() if reg["affine_matrix"] is not None else None,
        "residuals_px": reg["residuals_px"],
        "inliers": reg["inliers"],
        "total": reg["total"],
        "landmarks_used": reg_full["initial"].get("landmarks_used", []),
        "quality_flag": reg["quality_flag"],
        "quality_reason": reg.get("quality_reason"),
    }

    if reg["affine_matrix"] is None:
        raise RuntimeError(
            f"registration failed for {capture_path}: "
            f"{reg.get('quality_reason', 'no affine')}"
        )

    M = reg["affine_matrix"]
    tartan = [sample_region(Y_p, U_p, V_p, r, M) for r in tp_chart.TARTAN_REGIONS]
    grays_raw = [sample_region(Y_p, U_p, V_p, r, M) for r in tp_chart.GRAY_REGIONS]
    grays = []
    for r, raw in zip(tp_chart.GRAY_REGIONS, grays_raw):
        grays.append({
            "id": r["id"],
            "name": r["name"],
            "ideal_y10": r["expected"]["y10"],
            "measured_y10": raw["measured_yuv10"][0],
            "delta_y10": raw["delta_yuv10"][0],
            "u10": raw["measured_yuv10"][1],
            "v10": raw["measured_yuv10"][2],
            "patch_size_px": raw["patch_size_px"],
        })

    meta["ideal_frame_md5"] = _ideal_frame_md5()
    meta["tp_chart_version"] = tp_chart.TP_CHART_VERSION
    meta["tool_version"] = _TOOL_VERSION

    luma_scale = fit_gray_ramp(grays)

    geometry_block = _build_geometry_block(reg_full)

    return {
        "_meta": meta,
        "tartan": tartan,
        "grays": grays,
        "luma_scale": luma_scale,
        "geometry": geometry_block,
    }
```

Add a helper before `measure()`:

```python
def _build_geometry_block(reg_full):
    """Convert the in-memory geometry from register_with_geometry into the
    JSON shape documented in the Stage 2 design spec."""
    geom = reg_full["geometry"]
    if geom is None:
        return None
    initial = reg_full["initial"]
    final = reg_full["final"]

    # Serialize triangle dicts (drop NumPy tuples to plain lists).
    def _ser_pt(p):
        if p is None:
            return None
        return [float(p[0]), float(p[1])]

    tris_out = {}
    for tid, t in geom["fiducials"]["triangles"].items():
        if t is None:
            tris_out[tid] = None
            continue
        tris_out[tid] = {
            "back_corner_1": _ser_pt(t["back_corner_1"]),
            "back_corner_2": _ser_pt(t["back_corner_2"]),
            "back_midpoint": _ser_pt(t["back_midpoint"]),
            "apex_inferred": _ser_pt(t["apex_inferred"]),
            "apex_detected": _ser_pt(t["apex_detected"]),
            "confidence":    float(t["confidence"]),
            "failure_reason": None,
        }
    cross_out = None
    rc = geom["fiducials"]["cross"]
    if rc is not None:
        cross_out = {
            "center": [float(rc["x"]), float(rc["y"])],
            "h_arm_len_px": float(rc["h_arm_len_px"]),
            "v_arm_len_px": float(rc["v_arm_len_px"]),
            "confidence":   float(rc["confidence"]),
        }
    circle_out = None
    bc = geom["fiducials"]["circle"]
    if bc is not None:
        circle_out = {
            "center":      [float(bc["cx"]), float(bc["cy"])],
            "rx":          float(bc["rx"]),
            "ry":          float(bc["ry"]),
            "rotation_deg": float(bc["rotation_deg"]),
            "fit_rms":     float(bc["fit_rms"]),
            "confidence":  float(bc["confidence"]),
        }

    derived = geom["derived"]
    derived_out = {}
    for key in ("active_picture_box", "picture_extent_px",
                "picture_offset_from_ideal", "corner_skew_px",
                "cross_offset_from_ideal",
                "aperture_symmetry", "aspect_ratio_check",
                "diameter_vs_picture_height", "circle_fit_rms"):
        derived_out[key] = derived.get(key)
    derived_out["arrow_tip_coords"] = {
        k: _ser_pt(v) for k, v in (derived.get("arrow_tip_coords") or {}).items()
    } if derived.get("arrow_tip_coords") else None
    derived_out["clip_detected"] = derived.get("clip_detected")

    quality_flag, quality_reason = _compute_geometry_quality(geom, final)

    return {
        "fiducials": {"triangles": tris_out, "cross": cross_out, "circle": circle_out},
        "derived":   derived_out,
        "registration_refit": {
            "inlier_count_initial": int(initial.get("inliers", 0)),
            "inlier_count_final":   int(final.get("inliers", 0)),
            "final_residuals_px":   final["residuals_px"],
            "anchors_added":        reg_full["anchors_added"],
        },
        "quality_flag":   quality_flag,
        "quality_reason": quality_reason,
    }


def _compute_geometry_quality(geom, final):
    tris = geom["fiducials"]["triangles"]
    detected_count = sum(1 for t in tris.values() if t is not None)
    cross_ok = geom["fiducials"]["cross"] is not None
    circle_fit_rms = (geom["fiducials"]["circle"] or {}).get("fit_rms")
    mean_res = final["residuals_px"]["mean"]

    if detected_count < 2 and not cross_ok:
        return "failed", "cross missing and <2 triangles detected"
    if detected_count < 3 or not cross_ok:
        return "partial", f"only {detected_count}/4 triangles, cross={cross_ok}"
    # Both present in quantity; classify by thresholds.
    if (circle_fit_rms is not None and circle_fit_rms < 2.0
            and mean_res < 1.0):
        return "ok", None
    if ((circle_fit_rms is None or circle_fit_rms < 5.0)
            and mean_res < 2.0):
        return "warn", "residuals or circle_fit_rms exceed ok threshold"
    return "failed", "residuals or circle fit too poor"
```

- [ ] **Step 4: Run; verify pass**

```bash
python test_cases/test_tp_measure.py
python test_cases/test_tp_compare.py  # smoke — should still pass since geometry is additive
```

- [ ] **Step 5: Commit**

```bash
git add tp_measure.py test_cases/test_tp_measure.py
git commit -m "feat(tp_measure): use register_with_geometry; embed geometry block in JSON

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: tp_compare: render_geometry_section + CSS

**Files:**
- Modify: `tp_compare.py:362-380` (insert into `render_page`)
- Modify: `tp_compare.py:252-276` (add CSS for geometry panels)
- Test:   `test_cases/test_tp_compare.py`

Per-capture geometry panel between registration summary and tartan deltas. Picture box / clip detection / cross / circle / refit benefit subsections.

- [ ] **Step 1: Write the failing test**

Add to `test_cases/test_tp_compare.py`:

```python
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
    assert "active_picture_box" in html or "Picture box" in html
    # Cross panel
    assert "Aperture" in html or "aperture" in html
    # Circle panel
    assert "Aspect" in html or "aspect" in html
    # Clip status
    assert "no clip detected" in html or "apex_visible" in html
    # Refit benefit
    assert "11" in html  # initial inlier count
    assert "20" in html  # final inlier count


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
```

Append to `TESTS_NO_TMPDIR`.

- [ ] **Step 2: Run; verify failure**

- [ ] **Step 3: Implement render_geometry_section**

In `tp_compare.py`, add the function after `render_luma_scale_analysis`:

```python
def _delta_class_offset(value, green_lt, yellow_lt):
    """Color class for an absolute offset/skew value."""
    a = abs(value) if value is not None else 0
    if a < green_lt:
        return "delta-good"
    if a < yellow_lt:
        return "delta-warn"
    return "delta-bad"


def render_geometry_section(captures: List[Dict[str, Any]]) -> str:
    panels = []
    for c in captures:
        cap_name = _basename(c["_meta"]["capture"])
        g = c.get("geometry")
        if g is None:
            panels.append(
                f"<div class='geometry-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no geometry block (older JSON)</p></div>"
            )
            continue
        d = g["derived"]
        flag = g.get("quality_flag", "?")
        flag_class = {"ok": "ok", "warn": "warn", "partial": "warn",
                      "failed": "bad"}.get(flag, "")
        box = d.get("active_picture_box")
        if box is not None:
            offset = d.get("picture_offset_from_ideal", {})
            skew = d.get("corner_skew_px", {})
            box_html = (
                f"<table class='geo-table'>"
                f"<tr><th>top</th><th>left</th><th>bottom</th><th>right</th>"
                f"<th>w&times;h</th></tr>"
                f"<tr><td>{box['top']:.1f}</td><td>{box['left']:.1f}</td>"
                f"<td>{box['bottom']:.1f}</td><td>{box['right']:.1f}</td>"
                f"<td>{d['picture_extent_px']['width']:.1f}&times;"
                f"{d['picture_extent_px']['height']:.1f}</td></tr></table>"
                f"<div class='small'>"
                f"offset <span class='{_delta_class_offset(offset.get('dx', 0), 2, 5)}'>"
                f"dx={offset.get('dx', 0):+.1f}</span> "
                f"<span class='{_delta_class_offset(offset.get('dy', 0), 2, 5)}'>"
                f"dy={offset.get('dy', 0):+.1f}</span>"
                f" &nbsp; skew "
                f"<span class='{_delta_class_offset(skew.get('top_vs_bottom_width_diff', 0), 2, 5)}'>"
                f"w_diff={skew.get('top_vs_bottom_width_diff', 0):.1f}</span> "
                f"<span class='{_delta_class_offset(skew.get('left_vs_right_height_diff', 0), 2, 5)}'>"
                f"h_diff={skew.get('left_vs_right_height_diff', 0):.1f}</span>"
                f"</div>"
            )
        else:
            box_html = "<p class='muted'>picture box not derivable</p>"

        clip = d.get("clip_detected") or {}
        clip_rows = []
        for tid in ("TL", "TR", "BL", "BR"):
            entry = clip.get(tid, {})
            visible = entry.get("apex_visible", False)
            interp = entry.get("interpretation", "?")
            cls = "ok" if visible else "bad"
            clip_rows.append(
                f"<tr><td>{tid}</td>"
                f"<td class='{cls}'>{'visible' if visible else 'clipped'}</td>"
                f"<td>{_h.escape(interp)}</td></tr>"
            )
        clip_html = (
            f"<table class='geo-table'><tr><th>Corner</th>"
            f"<th>Apex</th><th>Interpretation</th></tr>"
            + "".join(clip_rows) + "</table>"
        )

        cross_off = d.get("cross_offset_from_ideal") or [None, None]
        aperture = d.get("aperture_symmetry")
        cross_html = (
            f"<div class='small'>"
            f"offset dx={cross_off[0]:+.2f}, dy={cross_off[1]:+.2f}"
            f" &nbsp; aperture_symmetry={aperture:.3f}"
            f"</div>"
        ) if cross_off[0] is not None and aperture is not None else "<p class='muted'>cross missing</p>"

        aspect = d.get("aspect_ratio_check")
        dvp = d.get("diameter_vs_picture_height")
        circle_fit_rms = d.get("circle_fit_rms")
        if aspect is not None:
            circle_html = (
                f"<div class='small'>"
                f"aspect_ratio_check={aspect:.4f} &nbsp; "
                f"diameter_vs_picture_height={dvp:.3f} &nbsp; "
                f"fit_rms={circle_fit_rms:.2f}px"
                f"</div>"
            )
        else:
            circle_html = "<p class='muted'>circle missing</p>"

        refit = g.get("registration_refit", {})
        refit_html = (
            f"<div class='small'>"
            f"inliers: {refit.get('inlier_count_initial', '?')} "
            f"&rarr; <b>{refit.get('inlier_count_final', '?')}</b> "
            f" &nbsp; mean_residual={refit.get('final_residuals_px', {}).get('mean', float('nan')):.2f}px"
            f"</div>"
        )

        panels.append(
            f"<div class='geometry-panel'>"
            f"<h3>{_h.escape(cap_name)} "
            f"<span class='{flag_class}'>[{flag}]</span></h3>"
            f"<h4>Picture box</h4>{box_html}"
            f"<h4>Clip detection</h4>{clip_html}"
            f"<h4>Registration cross</h4>{cross_html}"
            f"<h4>Black circle</h4>{circle_html}"
            f"<h4>Refit benefit</h4>{refit_html}"
            f"</div>"
        )

    return f"""
<section class="geometry">
  <h2>Geometry</h2>
  <p class="legend">
    Picture-in-raster geometry from the SW2 boundary triangles, picture-
    centered registration cross, and black-ring circle. Active picture box
    is bounded by the 4 triangle apexes (inferred from their back corners
    when the apex is clipped). Aspect / aperture / fit_rms surface
    decoder-side geometry artifacts; clip detection flags overscan or
    letterboxing.
  </p>
  {''.join(panels)}
</section>
"""
```

Add CSS additions to `_CSS` (append before the closing `"""`):

```python
.geometry-panel { margin: 12px 0; padding: 8px 12px; background: #1d2026; border: 1px solid #2a2e36; }
.geometry-panel h3 { margin: 4px 0 8px 0; font-size: 14px; }
.geometry-panel h4 { margin: 8px 0 4px 0; font-size: 12px; color: #c5d1e0; }
.geo-table { border-collapse: collapse; }
.geo-table th, .geo-table td { border: 1px solid #2a2e36; padding: 3px 6px; font-size: 12px; }
```

Update `render_page` to insert the geometry section between registration and tartan:

```python
def render_page(captures: List[Dict[str, Any]]) -> str:
    title = f"SW2 Comparison — {len(captures)} captures"
    sections = (
        render_registration_summary(captures)
        + render_geometry_section(captures)
        + render_tartan_deltas(captures)
        + render_gray_deltas(captures)
        + render_luma_scale_analysis(captures)
    )
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{_h.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{_CSS}</style>
</head><body>
<h1>{_h.escape(title)}</h1>
{sections}
</body></html>
"""
```

- [ ] **Step 4: Run; verify pass**

```bash
python test_cases/test_tp_compare.py
```

- [ ] **Step 5: Commit**

```bash
git add tp_compare.py test_cases/test_tp_compare.py
git commit -m "feat(tp_compare): render_geometry_section between registration and tartan

Per-capture panels: picture box + offset/skew, clip detection,
registration cross, circle, refit benefit. CSS additions for the
panel layout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 22: tp_calibrate stage2-fiducials preset

**Files:**
- Modify: `tp_calibrate.py:40-58`

Add a new preset that walks the operator through Stage 2 fiducial calibration targets: each triangle's two back corners + apex (12 points), plus the registration cross center (1 point), plus 4 sample points around the circle ring (1 per cardinal direction). Total: 17 targets.

- [ ] **Step 1: Add the preset entry**

In `tp_calibrate.py`, extend the `_PRESETS` dict (around line 41-58):

```python
_PRESETS: Dict[str, List[str]] = {
    "stage1-regions": [
        "YEL", "CYN", "BLU", "RED",
        "MAG", "GRN", "RED2", "CYN2",
        "G1", "G2", "G3", "G4",
    ],
    "stage1-landmarks": [
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    ],
    "stage1-all": [
        "YEL", "CYN", "BLU", "RED",
        "MAG", "GRN", "RED2", "CYN2",
        "G1", "G2", "G3", "G4",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    ],
    "stage2-fiducials": [
        # 4 triangles, each with back-corner-1, back-corner-2, apex.
        "TL_back_corner_1", "TL_back_corner_2", "TL_apex",
        "TR_back_corner_1", "TR_back_corner_2", "TR_apex",
        "BL_back_corner_1", "BL_back_corner_2", "BL_apex",
        "BR_back_corner_1", "BR_back_corner_2", "BR_apex",
        # Registration cross center.
        "RC_center",
        # Black circle ring sample points (12, 3, 6, 9 o'clock).
        "BC_north", "BC_east", "BC_south", "BC_west",
    ],
    "stage2-landmarks": [
        # Updated 12-anchor grid catalog.
        "L1", "L2", "L3", "L4", "L5", "L6",
        "L7", "L8", "L9", "L10", "L11", "L12",
    ],
}
```

- [ ] **Step 2: Manual verification**

```bash
source venv/bin/activate
python tp_calibrate.py --help 2>&1 | grep -A2 preset
```

Expected: help text includes `stage2-fiducials` and `stage2-landmarks` in the list of valid presets (if argparse choices are derived from the dict keys).

Then verify the preset list lengths:

```bash
python -c "import tp_calibrate; print(list(tp_calibrate._PRESETS.keys()))"
python -c "import tp_calibrate; print(len(tp_calibrate._PRESETS['stage2-fiducials']))"
# expect: 17
```

- [ ] **Step 3: Commit**

```bash
git add tp_calibrate.py
git commit -m "feat(tp_calibrate): add stage2-fiducials and stage2-landmarks presets

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 23: Real-capture smoke test + catalog refinement (manual)

**Files:**
- Modify: `tp_chart.py` (refine BOUNDARY_TRIANGLES, REGISTRATION_CROSS, BLACK_CIRCLE coords from calibration JSON)
- Output: `tp_smoke_outputs/stage2_smoke_report.html`

Run tp_calibrate on a real Snell HD capture using the `stage2-fiducials` preset, transcribe the clicked coordinates back into `tp_chart.py` to refine the chart-spec defaults, then regenerate the 3-way comparison HTML for dvdrip/snellhd/snellld and inspect.

- [ ] **Step 1: Locate a representative Snell HD capture**

```bash
ls /wintmp/analog_video/tp_compare/captures/*.mov 2>/dev/null | head -5
# OR wherever the latest Snell HD SDI captures live -- use the file the Stage 1
# calibration used (see tp_smoke_outputs/snellhd_calib.json).
```

- [ ] **Step 2: Run calibration with stage2-fiducials preset**

```bash
source venv/bin/activate
python tp_calibrate.py /wintmp/analog_video/tp_compare/captures/snellhd_<latest>.mov \
    --preset stage2-fiducials \
    --output tp_smoke_outputs/stage2_calib.json
```

Walk through the 17 prompts, clicking the precise pixel center of each feature. Save and exit ('q').

- [ ] **Step 3: Inspect the calibration JSON**

```bash
cat tp_smoke_outputs/stage2_calib.json
```

Expected: keys for each of the 17 labels with `(x, y)` integer pairs.

- [ ] **Step 4: Refine tp_chart.py constants if real captures diverge significantly from the defaults**

If a real chart's TL back corners aren't at ~(50,30) and (70,30), update `_make_triangle` or override individual triangles' positions. Likewise for the cross and circle.

This is intentionally a judgment call by the implementer. If the captured fiducial positions are within ±5 px of the synthesized-chart placeholders, the affine fit absorbs the difference and no edit is needed. Larger drifts (>5 px) warrant updating `tp_chart.py` to reflect the real chart layout.

Re-run the chart-catalog tests after any edit:

```bash
python test_cases/test_tp_chart.py
```

- [ ] **Step 5: Regenerate the 3-way comparison report**

```bash
# Re-run tp_measure on each capture to pick up the new geometry block.
for cap in /wintmp/analog_video/tp_compare/captures/{dvdrip,snellhd,snellld}_*.mov; do
    base=$(basename "$cap" .mov)
    python tp_measure.py "$cap" --frame 60 \
        --output "tp_smoke_outputs/${base}_stage2.json"
done

# Generate the comparison HTML.
python tp_compare.py \
    tp_smoke_outputs/dvdrip_*_stage2.json \
    tp_smoke_outputs/snellhd_*_stage2.json \
    tp_smoke_outputs/snellld_*_stage2.json \
    --output tp_smoke_outputs/stage2_smoke_report.html
```

- [ ] **Step 6: Visually inspect the report and document findings**

Open the HTML and verify:

- **Geometry section appears** between registration summary and tartan deltas.
- **dvdrip**: shows top + bottom edges clipped by ~3 px each (the DVD master drops 6 lines, the padding-from-480 step adds them back as grey rows, and the boundary triangle apex_inferred values should fall outside or near the edge of the captured frame's content rows).
- **snellhd / snellld**: no clipping detected. All 4 triangles' apex_visible == True.
- **Refit benefit**: inlier count should now be substantially higher than the Stage 1 4/8 — expect ~11/12 grids surviving + ~8 triangle back corners + 1 cross = ~20 anchors total.
- **Refit residuals**: mean_residual should be < 1 px on all three captures (vs the Stage 1 ~1.5-2 px).

If any of these expectations are violated, examine the JSON to diagnose: which detector is failing, what its confidence is, where the search window was looking. This is a real-capture validation — the fixture tests already confirmed the detector accuracy contracts hold; this step confirms they hold against the real chart's idiosyncrasies.

- [ ] **Step 7: Commit any catalog refinements + the smoke artifacts**

```bash
git add tp_chart.py tp_smoke_outputs/stage2_smoke_report.html tp_smoke_outputs/stage2_calib.json
git commit -m "$(cat <<'EOF'
chore: refine Stage 2 chart catalog coords from real-capture calibration

stage2_calib.json captures the operator-clicked positions on the actual
Snell HD capture; tp_chart.py constants updated where they exceeded
the ±5 px tolerance against the synthesized-chart placeholders. The
3-way smoke report demonstrates dvdrip top/bottom clip detection and
the inlier-count improvement on snellhd/snellld.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (after writing all tasks)

Before handing off to the executor:

- [x] All 23 tasks have explicit file paths.
- [x] Every step that changes code includes the actual code block.
- [x] No "TBD"/"TODO" placeholders left in the plan.
- [x] Test conventions match the existing plain-Python harness (no pytest).
- [x] Each detector has fixture-driven accuracy tests with the spec's threshold.
- [x] Type/signature consistency: `detect_fiducial(Y, fid)`, `register_with_geometry(Y) -> dict`, `detect_geometry(Y, M) -> dict`, `tp_fixtures.synthesize_with_ground_truth(...)` returns `(Y, U, V, gt)` everywhere they're invoked.
- [x] Geometry JSON shape (Task 20 helper) matches the spec exactly: `fiducials.{triangles,cross,circle}` + `derived.*` + `registration_refit` + `quality_flag/reason`.
- [x] HTML section ordering (Task 21) places Geometry between Registration and Tartan — matches the spec.
- [x] tp_compare gracefully handles older JSONs without a `geometry` block (renders "no geometry block (older JSON)" panel).
- [x] Real-capture smoke test (Task 23) is the only manual task; everything before it is fully automated and CI-verifiable.
