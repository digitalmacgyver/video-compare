# SW2 Stage 2 — Geometry / Picture-in-Raster Design

Status: design / brainstorming complete, awaiting plan
Date: 2026-05-11

## Purpose

Answer the question the SW2 Stage 1 brainstorming originally posed as (a):
**where is the picture in the raster, how big is it, is it square?** The
chart's deliberate registration features — boundary triangles, registration
cross, and black circle — were designed for exactly this and Stage 1
deliberately deferred them. Stage 2 adds detectors for each, derives
per-capture geometry measurements (active picture box, offsets, aspect
ratio, clip-from-edge detection), and feeds the new fiducials back into
Stage 1's registration step to tighten the affine fit that the existing
tartan/gray sampling already uses.

## Scope

In scope for this iteration:
- 4 boundary triangle detectors (top-left, top-right, bottom-left,
  bottom-right) that register on **back corners** (the points furthest
  from the picture edge each triangle indicates), not the apex.
- 1 registration-cross detector (sub-pixel template match) with
  horizontal/vertical aperture symmetry measurement.
- 1 black-circle detector (ellipse fit on dark-ring pixels) for aspect-
  ratio and diameter-vs-picture-height checks.
- A generic dispatcher (`detect_fiducial(Y, fid)`) routing by a `kind`
  field on each fiducial record. The existing Stage 1 `detect_landmark`
  becomes the implementation for `kind="grid_intersection"`.
- Grid-landmark catalog refresh: drop L8 (no clean cross at the
  position), relocate L1/L2/L3 from y=54 to y=108, add anchors at y=216
  to fill the y-axis gap. Target catalog size 10–12 with all anchors in
  the chart's safe interior (x ∈ [180, 540], y ∈ [108, 270]).
- A new `tp_fixtures.py` test-only module producing synthesized frames
  with known ground-truth fiducial positions plus optional degradations
  (shift, rotation, noise, blur, top/bottom/left/right clipping). Each
  Stage 2 detector is validated against fixture variants with explicit
  sub-pixel accuracy thresholds before being run on real captures.
- Synthesizer extension: all 4 boundary triangles, the registration
  cross, and the black circle become part of `tp_synthesize.synthesize()`.
- Sequential-with-feedback registration: `register()` unchanged; new
  `register_with_geometry()` runs registration, detects Stage 2
  fiducials in registered coords, then re-fits the affine with all
  fiducials as anchors. Stage 1 sampling benefits from the tighter fit.
- New `geometry` top-level key in the per-capture JSON, plus a new
  Geometry section in the comparison HTML.

Out of scope:
- Stage 3 frequency / burst region analysis (separate spec).
- Decoder-class classification from artifact signatures (Stage 3).
- Multi-frame averaging of geometry measurements.

## Non-goals

- Not a real-time geometry monitor; operates offline on captured files.
- Not a precise sub-pixel calibration tool for analog timing
  (registration cross gives us sub-pixel-of-pixel; analog timing would
  require a different signal class entirely).
- Not a guarantee-of-coverage gauge: we report what we observe, not
  what's outside the active picture region. Clip detection tells you a
  processor lost the apex of a triangle; it does not measure exactly
  how much picture content was lost beyond the apex.

## Reference materials

- Stage 1 design: `docs/superpowers/specs/2026-05-10-sw2-tp-compare-design.md`
- Stage 1 plan + execution: `docs/superpowers/plans/2026-05-10-sw2-tp-compare-stage1.md`
- TPG20/21 manual (PAL chart spec at page 4.25, NTSC deltas at page 4.30):
  `/wintmp/analog_video/tp_compare/ref/tpg20_21_page_58.pdf`
- "What is it?" overview (NTSC):
  `/wintmp/analog_video/tp_compare/ref/The_Snell_and_Wilcox_Test_Chart_2_What_is_it.pdf`
- Composite Decoder Whitepaper (decoder-class taxonomy):
  `/wintmp/analog_video/tp_compare/ref/compositedecoder_wp_tp2_sections_explained.pdf`
- Existing calibration JSON for Stage 1 region centers (tp_calibrate.py
  workflow): `tp_smoke_outputs/snellhd_calib.json` (regenerable; not
  committed but documents the calibration source for tp_chart.py).

## Diagnostic findings driving the design

Investigation against the three real captures (dvdrip, snellhd, snellld)
revealed three issues the design addresses:

1. **L8 in the current grid-landmark catalog (360, 378) is universally
   undetectable.** The chart has no clean grid-line crossing at that
   position. Drop it.

2. **The top-row landmarks (L1/L2/L3 at y=54) sit near the chart's busy
   upper-feature region (boundary triangles + tartan) and show 1–4 px
   detection noise.** Landmarks at y=162/216/270 in the chart interior
   give sub-pixel-to-1px residuals against a single-affine fit.
   Relocate the top row to y=108; add anchors at y=216 to fill the gap.

3. **Only 4 of 8 current landmarks survive RANSAC.** With detection
   noise > the per-fiducial geometric drift the affine is trying to
   fit, RANSAC correctly rejects half. Solution: more anchors (12 vs 8)
   in cleaner positions, plus Stage 2 fiducials feeding the re-fit step
   for additional anchor count.

The fixture-first development discipline (synthesizing test frames with
known ground truth and validating each detector against per-variant
accuracy thresholds) lets us separate detector noise from chart-layout
drift in future investigations, decoupling these two failure modes.

## Architecture

No new product modules. Changes happen in existing files, plus one new
test-only module:

```
tp_chart.py        # ADD: Stage 2 fiducial catalogs (boundary triangles,
                   #      registration cross, black circle). Each entry
                   #      has a `kind` field for the dispatcher.
                   # ADD: refreshed grid-landmark catalog (12 anchors in
                   #      safe interior).
                   # ADD: chart-spec offsets needed for back-corner →
                   #      apex inference per triangle.

tp_synthesize.py   # ADD: render all 4 boundary triangles, registration
                   #      cross, black circle. Correct the existing
                   #      _draw_boundary_triangle_upper_left orientation
                   #      to match the real chart (apex toward the
                   #      boundary it indicates).

tp_register.py     # REFACTOR: introduce detect_fiducial(Y, fid)
                   #           dispatcher. Existing detect_landmark
                   #           becomes the implementation for
                   #           kind="grid_intersection".
                   # ADD: _detect_boundary_triangle (back-corner-first).
                   # ADD: _detect_registration_cross (template match
                   #      with sub-pixel parabolic refinement).
                   # ADD: _detect_black_circle (cv2.fitEllipse on
                   #      annular dark-pixel sample).
                   # ADD: detect_geometry(Y, M_initial) orchestrator
                   #      (uses the affine to map each fiducial's ideal
                   #      position into a capture-coord search window).
                   # ADD: register_with_geometry(Y) — sequential-with-
                   #      feedback wrapper.

tp_measure.py      # MODIFY: call register_with_geometry() instead of
                   #         register(); embed the geometry block in
                   #         JSON at top-level as "geometry".

tp_compare.py      # ADD: render_geometry_section(captures) inserted
                   #      between registration summary and tartan deltas
                   #      in render_page.

tp_calibrate.py    # ADD: stage2-fiducials preset that walks the
                   #      operator through back-corner-1, back-corner-2,
                   #      and apex of each boundary triangle, plus the
                   #      registration cross center, plus a few ring
                   #      sample points for the circle.

tp_fixtures.py     # NEW (test-only): synthesize_with_ground_truth(...)
                   # producing (Y, U, V, ground_truth) frames with
                   # optional degradations and clip simulations.

test_cases/test_tp_chart.py      # MODIFY: new catalog IDs
test_cases/test_tp_synthesize.py # ADD: tests for Stage 2 rendering
test_cases/test_tp_register.py   # ADD: per-detector + dispatcher tests
                                 #      using tp_fixtures
test_cases/test_tp_fixtures.py   # NEW: fixture API tests
test_cases/test_tp_measure.py    # ADD: geometry block in JSON
test_cases/test_tp_compare.py    # ADD: geometry section in HTML
```

## Synthesizer + fixture infrastructure

### Synthesizer extensions

`tp_synthesize.synthesize(width, height)` learns three new render
functions, called in this order after the existing tartan/gray steps so
they overdraw the grid lines where appropriate:

```python
def _draw_boundary_triangles(Y):
    """Render all 4 boundary triangles per tp_chart._BOUNDARY_TRIANGLES.
    Each is a black filled triangle with apex pointing toward the
    boundary it marks (top-left and top-right point up; bottom-left and
    bottom-right point down). The base sits inside the picture; the
    apex sits at or just inside the boundary."""

def _draw_registration_cross(Y):
    """Render the registration-check feature: a small black box
    (~24x24 px) containing a centered white "+" with ~3 px arms.
    Position from tp_chart._REGISTRATION_CROSS."""

def _draw_black_circle(Y):
    """Render the black-ring circle, diameter = picture height ~ 486 px,
    centered at the picture center. 168 ns line width => ~3 px thick."""
```

The existing `_draw_boundary_triangle_upper_left` is removed and its
correct orientation is folded into `_draw_boundary_triangles`.

### Fixture API

New module `tp_fixtures.py` (test-only — imported by test files, not by
the production pipeline):

```python
def synthesize_with_ground_truth(
    width=720, height=486,
    shift=(0, 0),          # global translation in capture coords
    rotation_deg=0.0,      # global rotation about picture center
    noise_sigma=0.0,       # Gaussian noise on Y, in 10-bit codes
    blur_sigma=0.0,        # Gaussian blur of Y (chroma untouched)
    clip_top=0, clip_bottom=0,   # zero out top/bottom N rows
    clip_left=0, clip_right=0,   # zero out left/right N columns
) -> Tuple[Y, U, V, ground_truth]:
    """Render the ideal frame with controlled degradations and return
    a ground_truth dict mapping every fiducial id to its exact pixel
    position in the returned frame after the requested transformations.

    Ground truth shape:
        {
          "grid_intersections": {"L1": (180, 108), ...},  # 12 entries
          "triangles": {
            "TL": {
              "back_corner_1": (x, y),  # both corners of the back edge
              "back_corner_2": (x, y),
              "back_midpoint": (x, y),
              "apex":          (x, y),  # may be outside the frame if
                                         # clip_top removes it
              "apex_visible":  bool,
            },
            "TR": {...}, "BL": {...}, "BR": {...},
          },
          "cross": {
            "center": (x, y),
            "h_arm_endpoints": ((x1, y), (x2, y)),
            "v_arm_endpoints": ((x, y1), (x, y2)),
          },
          "circle": {
            "center": (cx, cy),
            "radius": r,  # rx == ry under identity; differs only when
                          # a non-square shift+rotation is applied
          },
        }

    Positions are computed analytically from the chart-defined ideal
    positions and the requested transformation. Detection is not
    involved in producing ground truth.
    """
```

### Fixture-driven detector accuracy thresholds

Each detector ships with explicit per-variant accuracy thresholds the
tests enforce. The thresholds are part of the design contract — a
detector that doesn't meet them fails CI.

| Detector | Identity | Noise σ=20 | Blur σ=1.5 | Combined |
|---|---|---|---|---|
| grid_intersection (current) | < 0.3 px | < 1.0 px | < 1.5 px | < 2.0 px |
| boundary_triangle back corners | < 0.5 px | < 1.0 px | < 1.5 px | < 2.5 px |
| boundary_triangle apex | < 1.0 px | < 2.0 px | < 3.0 px | < 4.0 px |
| registration_cross center | < 0.2 px | < 0.6 px | < 1.0 px | < 1.5 px |
| black_circle center | < 0.5 px | < 1.0 px | < 1.5 px | < 2.0 px |
| black_circle radius | within 0.5 px | within 1.5 px | within 2.0 px | within 3.0 px |

A separate fixture variant `clip_top=5` (zero out top 5 rows) verifies
the boundary triangle detector still returns valid back corners when
the apex is gone — back-corner thresholds stay the same; `apex_detected`
becomes `None`; `apex_inferred` is computed from the back corners and
chart-spec offsets.

## Detector specifications

All detectors are routed through a generic dispatcher:

```python
def detect_fiducial(Y, fid):
    """Dispatch by fid['kind'] to the right implementation. Returns a
    detector-specific dict, or None on detection failure."""
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

### `_detect_grid_intersection` (relocated Stage 1 detector)

The existing `detect_landmark` becomes this. No behavioral change.
The fixture suite formalizes the accuracy contract.

### `_detect_boundary_triangle`

**Returns**: `{back_corner_1, back_corner_2, back_midpoint,
apex_inferred, apex_detected, confidence}` or None.

**Algorithm**:
1. Crop the search window centered on the **ideal back-midpoint** (not
   apex — the back is what we expect to find present in any capture).
2. Threshold dark pixels: Y10 < 0.3 × `GREY_BACKGROUND_Y10` (≈ 150).
3. Reject if dark_count < 50 (window contains no plausible triangle).
4. Based on `fid['orientation']` (one of `apex_up`, `apex_down`,
   `apex_left`, `apex_right`), identify the "back" edge of the dark
   cluster:
   - `apex_up`:    back = bottom-most row containing dark pixels.
   - `apex_down`:  back = top-most row.
   - `apex_left`:  back = right-most column.
   - `apex_right`: back = left-most column.
5. Within the back edge, find the leftmost and rightmost (or top-most
   and bottom-most) dark pixels — these are `back_corner_1` and
   `back_corner_2`. Apply sub-pixel refinement by weighting nearby
   dark pixels.
6. `back_midpoint` = midpoint of the two back corners.
7. `apex_inferred` = `back_midpoint` + chart-spec offset vector from
   `fid['ideal_apex'] - fid['ideal_back_midpoint']`.
8. Try to detect `apex_detected` from data: find the extreme point of
   the dark cluster in the orientation direction. If it lies within
   `chart_spec_distance ± 2 px` of `back_midpoint`, return it; else
   return None (apex clipped or unreliable).
9. `confidence` = dark_count / window_area, clipped to [0, 1].

The two back corners feed into the affine re-fit as registration
anchors. The picture-boundary derivation uses `apex_inferred`.

### `_detect_registration_cross`

**Returns**: `{x, y, h_arm_len_px, v_arm_len_px, confidence}` or None.

**Algorithm**:
1. At module load, synthesize a 17×17 px template matching the chart's
   "200 ns white cross on black box" — a black square with a 3-px-wide
   white "+" centered.
2. Crop the search window from Y centered on `fid['ideal_x'], fid['ideal_y']`.
3. `cv2.matchTemplate(region, template, cv2.TM_SQDIFF_NORMED)`.
4. Find the minimum of the SQDIFF surface; do parabolic interpolation
   on the 3×3 neighborhood for sub-pixel offset.
5. Measure h_arm_len_px by counting bright pixels (Y10 > 700) along
   the horizontal centerline at the detected position. Same for v_arm.
6. `confidence` = `1 − min(SQDIFF_NORMED)`, clipped to [0, 1].
7. Reject if confidence < 0.4 (no plausible cross found).

### `_detect_black_circle`

**Returns**: `{cx, cy, rx, ry, rotation_deg, fit_rms, confidence}` or None.

**Algorithm**:
1. Build an annular mask: pixels within ±`fid['search_band_px']` of
   `fid['expected_radius_px']` from `(fid['ideal_cx'], fid['ideal_cy'])`.
2. Threshold dark pixels within the annulus.
3. Reject if dark_pixel_count < 100 (circle too obscured or missing).
4. Extract (x, y) coords of dark pixels as a point cloud.
5. `cv2.fitEllipse(points)` returns ((cx, cy), (axis_a, axis_b),
   rotation_deg). Use `min` and `max` of the two axes for `rx, ry`.
6. Compute `fit_rms` as the mean perpendicular distance from each
   point to the fitted ellipse.
7. `confidence` = 1 − fit_rms / 10.0 (clamped to [0, 1]).
8. Reject if fit_rms > 5.0 px (fit is too poor to trust).

## Geometry derivation

After detection, `detect_geometry(Y, M_initial)` runs all Stage 2
detectors against the captured frame `Y` (using `M_initial` to map
each fiducial's ideal position into a capture-coord search-window
center), then computes derived measurements:

### From boundary triangles (back-corner-driven):

```python
# All positions in this section are in capture coords (the same coords
# the apex_inferred values are reported in).
#
# Each triangle's apex_inferred is the boundary point that triangle
# indicates. The active picture extent is bounded by them:
active_picture_box = {
    "top":    mean(TL.apex_inferred.y, TR.apex_inferred.y),
    "bottom": mean(BL.apex_inferred.y, BR.apex_inferred.y),
    "left":   mean(TL.apex_inferred.x, BL.apex_inferred.x),
    "right":  mean(TR.apex_inferred.x, BR.apex_inferred.x),
}
picture_extent_px = {
    "width":  active_picture_box["right"]  - active_picture_box["left"],
    "height": active_picture_box["bottom"] - active_picture_box["top"],
}

# Map the chart's ideal active-picture corners (a new constant
# IDEAL_PICTURE_BOX in tp_chart.py, listing the chart-defined active
# picture corners in ideal coords) through the final affine into
# capture coords for a like-with-like comparison:
ideal_corners_in_capture = apply_affine(M_final, IDEAL_PICTURE_BOX_corners)
picture_offset_from_ideal = {
    "dx": active_picture_box["left"] - ideal_corners_in_capture.left,
    "dy": active_picture_box["top"]  - ideal_corners_in_capture.top,
}
top_width    = TR.apex_inferred.x - TL.apex_inferred.x
bottom_width = BR.apex_inferred.x - BL.apex_inferred.x
left_height  = BL.apex_inferred.y - TL.apex_inferred.y
right_height = BR.apex_inferred.y - TR.apex_inferred.y
corner_skew_px = {
    "top_vs_bottom_width_diff":  abs(top_width  - bottom_width),
    "left_vs_right_height_diff": abs(left_height - right_height),
}
arrow_tip_coords = {
    "TL": TL.apex_inferred, "TR": TR.apex_inferred,
    "BL": BL.apex_inferred, "BR": BR.apex_inferred,
}
```

### Clip detection (per-triangle):

For each triangle, compare `apex_detected` to `apex_inferred` to
determine whether the boundary edge clipped the apex, and if so, by
how many pixels:

```python
# apex_visible: was apex_detected returned from the captured frame?
apex_visible = (triangle.apex_detected is not None)

# clip_px: distance from the inferred apex to the nearest captured-
# frame edge in the direction the triangle's apex points. Zero when
# apex_visible is True. Positive when the apex sits outside the frame
# or so close to the edge that the apex feature is unrecoverable.
#
# Per-orientation formula (capture coords, y grows downward, x grows
# rightward; H = frame height, W = frame width):
#   apex_up    (TL, TR):  clip_px = max(0,  -apex_inferred.y)
#   apex_down  (BL, BR):  clip_px = max(0,  apex_inferred.y - (H - 1))
#   apex_left  (none in SW2): clip_px = max(0,  -apex_inferred.x)
#   apex_right (none in SW2): clip_px = max(0,  apex_inferred.x - (W - 1))

interpretation = f"top edge clipped ~{clip_px:.0f} px"   # for apex_up
                  # bottom / left / right for the other orientations;
                  # "no clip detected" when clip_px == 0.
```

A processor with overscan applied to the top edge produces
`TL.apex_visible == False` and `TR.apex_visible == False` with
matching `clip_px` values; the value quantifies the overscan amount
on that edge. When all four corners are clipped by similar amounts
the picture has been letterboxed/pillarboxed symmetrically.

### Registration cross derivation:

```python
cross_offset_from_ideal = cross.center - fid_RC.ideal_center
aperture_symmetry = min(h_arm, v_arm) / max(h_arm, v_arm)
```

### Circle derivation:

```python
aspect_ratio_check = min(rx, ry) / max(rx, ry)
diameter_vs_picture_height = 2 * max(rx, ry) / picture_extent_px.height
```

## Sequential-with-feedback registration

New function `register_with_geometry(Y) -> RegistrationResult`:

```python
def register_with_geometry(Y):
    # Step 1: run Stage 1 registration unchanged.
    initial_reg = register(Y)  # uses GRID_LANDMARKS only

    if initial_reg["affine_matrix"] is None:
        # No affine, no Stage 2 either.
        return {"initial": initial_reg, "geometry": None, "final": initial_reg}

    # Step 2: detect Stage 2 fiducials in capture coords. Their ideal
    # positions are in ideal coords; we use initial_reg's affine to map
    # ideal -> capture for the search-window centers.
    M_initial = initial_reg["affine_matrix"]
    geometry_fiducials = detect_geometry(Y, M_initial)

    # Step 3: collect ALL successfully-detected fiducials (grids +
    # geometry fiducials' back corners + cross center + circle center)
    # as anchors. Re-fit the affine.
    all_anchors_capture = collect_capture_positions(initial_reg, geometry_fiducials)
    all_anchors_ideal   = collect_ideal_positions(initial_reg, geometry_fiducials)
    final_reg = fit_affine(all_anchors_capture, all_anchors_ideal)

    return {
        "initial":  initial_reg,    # what Stage 1 alone produced
        "geometry": geometry_fiducials,
        "final":    final_reg,      # what tartan/gray sampling uses
        "anchors_added": [...],     # ids of fiducials added to the fit
    }
```

`tp_measure.measure()` calls `register_with_geometry()` and uses
`result["final"]["affine_matrix"]` for region sampling. Stage 1 tartan
and gray measurements pick up the tighter fit transparently.

## JSON shape

A new top-level `geometry` key joins `_meta`, `tartan`, `grays`,
`luma_scale`:

```json
"geometry": {
  "fiducials": {
    "triangles": {
      "TL": {
        "back_corner_1": [16, 96],
        "back_corner_2": [16, 109],
        "back_midpoint": [16, 102.5],
        "apex_inferred": [3, 102.5],
        "apex_detected": null,
        "confidence":    0.42,
        "failure_reason": null
      },
      "TR": {...}, "BL": {...}, "BR": {...}
    },
    "cross": {
      "center": [240.5, 94.1],
      "h_arm_len_px": 6.0,
      "v_arm_len_px": 6.2,
      "confidence": 0.97
    },
    "circle": {
      "center": [360.2, 243.1],
      "rx": 242.8,
      "ry": 243.4,
      "rotation_deg": 0.3,
      "fit_rms": 0.42,
      "confidence": 0.96
    }
  },
  "derived": {
    "active_picture_box": {"top": 6, "left": 9, "bottom": 480, "right": 711},
    "picture_extent_px": {"width": 702, "height": 474},
    "picture_offset_from_ideal": {"dx": 3, "dy": 3},
    "corner_skew_px": {"top_vs_bottom_width_diff": 0.5,
                       "left_vs_right_height_diff": 0.3},
    "arrow_tip_coords": {
      "TL": [3, 102.5], "TR": [711, 102.5],
      "BL": [3, 384.5], "BR": [711, 384.5]
    },
    "clip_detected": {
      "TL": {"apex_visible": false, "clip_px": 3,
             "interpretation": "top edge clipped ~3 px"},
      "TR": {"apex_visible": false, "clip_px": 3,
             "interpretation": "top edge clipped ~3 px"},
      "BL": {"apex_visible": false, "clip_px": 3,
             "interpretation": "bottom edge clipped ~3 px"},
      "BR": {"apex_visible": false, "clip_px": 3,
             "interpretation": "bottom edge clipped ~3 px"}
    },
    "cross_offset_from_ideal": [0.5, 0.1],
    "aperture_symmetry": 0.968,
    "aspect_ratio_check": 0.998,
    "diameter_vs_picture_height": 1.001,
    "circle_fit_rms": 0.42
  },
  "registration_refit": {
    "inlier_count_initial": 6,
    "inlier_count_final":   11,
    "final_residuals_px": {"mean": 0.18, "max": 0.42},
    "anchors_added": ["TL.bc1", "TL.bc2", "TR.bc1", "TR.bc2",
                      "BL.bc1", "BL.bc2", "BR.bc1", "BR.bc2", "RC"]
  },
  "quality_flag": "ok",
  "quality_reason": null
}
```

`quality_flag` is one of `ok` / `warn` / `partial` / `failed`:
- `ok`: ≥3 of 4 triangles detected with both back corners, cross
  detected, circle fit_rms < 2 px, refit residuals.mean < 1 px.
- `warn`: data complete but some thresholds exceeded (e.g., fit_rms
  between 2 and 5, or refit residuals.mean between 1 and 2 px).
- `partial`: <3 triangles or cross missing — some `derived` fields
  cannot be computed (set to null) but the rest of the analysis is
  still valid.
- `failed`: insufficient fiducials to derive geometry; cross missing
  AND <2 triangles.

`luma_scale`, `tartan`, `grays` continue to use the final (Stage-2-
refitted) affine implicitly via `register_with_geometry()`.

## HTML report extension

`render_geometry_section(captures)` inserted in `render_page` between
the registration summary and tartan deltas sections.

The section contains per-capture panels:

**Picture box panel**: compact table — `top / left / bottom / right` +
`width × height`. Adjacent: `picture_offset_from_ideal` and the two
`corner_skew_px` values. Cell color-coded green/yellow/red on offset
and skew magnitudes (thresholds: green < 2 px, yellow < 5 px, red ≥ 5 px).

**Clip detection panel**: per-triangle `apex_visible` and (when
clipped) the `clip_px` value with the human-readable interpretation.
A clipped capture shows e.g., "top edge clipped ~3 px" in red.

**Registration cross panel**: `cross_offset_from_ideal` + horizontal
vs vertical arm length + `aperture_symmetry` ratio. Asymmetry < 0.95
displayed in yellow (directionally-biased sharpening filter).

**Circle panel**: `(rx, ry)` + `aspect_ratio_check` +
`diameter_vs_picture_height`. Values within ±0.5% of 1.0 = green;
within ±2% = yellow; beyond = red.

**Re-fit benefit row**: `inlier_count_initial → inlier_count_final` and
the `final_residuals_px.mean` value. A small win indicator if the
final inlier count exceeds the initial.

A small SVG schematic in the section header shows the synthesized
ideal frame outline with the 4 triangles, cross, and circle in their
nominal positions — operator-friendly legend.

## Testing

Tests follow the existing plain-Python convention in `test_cases/`.

### `test_tp_fixtures.py` (new)
- `test_fixture_ground_truth_identity_matches_synthesizer` — without
  any degradation, ground truth positions equal the analytically-
  derived chart positions.
- `test_fixture_ground_truth_tracks_shift` — `shift=(5, 7)` updates
  all ground-truth positions by (5, 7).
- `test_fixture_ground_truth_tracks_rotation` — `rotation_deg=1.5`
  rotates positions about picture center; ground truth records the
  rotated coords.
- `test_fixture_clip_top_marks_apex_invisible` — `clip_top=5` zeroes
  out the first 5 rows; `TL.apex_visible` becomes False; back corners
  stay visible.

### `test_tp_synthesize.py` additions
- `test_synthesize_renders_all_four_boundary_triangles` — each cell
  has the expected dark count and orientation.
- `test_synthesize_renders_registration_cross_with_white_arms` —
  cross center is bright (Y > 700); surrounding box is dark.
- `test_synthesize_renders_black_circle_at_expected_radius` —
  dark-ring count at expected radius ± 2 px is high; at radius+10 px
  is near-zero.

### `test_tp_register.py` additions

For each detector kind, fixture-driven test matrix:

```python
# Identity variant (must beat the tightest threshold)
test_detect_boundary_triangle_back_corners_identity()
test_detect_boundary_triangle_apex_identity()
test_detect_registration_cross_center_identity()
test_detect_black_circle_center_and_radius_identity()

# Plus noise / blur / shift / rotation / clipped / combined variants,
# each asserting their per-row accuracy threshold from the table.
test_detect_*_noise_sigma_20()
test_detect_*_blur_sigma_1_5()
test_detect_*_shift_subpixel()
test_detect_*_combined()
test_detect_boundary_triangle_back_corners_with_clipped_apex()

# Dispatcher routing
test_detect_fiducial_dispatches_correctly()
test_detect_fiducial_unknown_kind_raises()

# End-to-end orchestrator
test_detect_geometry_returns_full_block_on_clean_fixture()
test_detect_geometry_partial_on_clipped_fixture()
test_register_with_geometry_improves_inlier_count()
test_register_with_geometry_preserves_correct_affine()
```

### `test_tp_measure.py` additions
- `test_measure_end_to_end_includes_geometry_block_on_synthesized_input`
  — JSON has `geometry` top-level key with documented schema.
- `test_measure_uses_refitted_affine_for_sampling` — verifies that
  measuring on a synthesized-then-shifted fixture produces tartan
  deltas no worse than measuring on the unshifted fixture (the refit
  absorbs the shift).

### `test_tp_compare.py` additions
- `test_render_geometry_section_shows_picture_box_and_clip_status` —
  HTML contains the box dimensions, clip-detected interpretation
  strings, cross and circle panels.

### `test_tp_chart.py` modification
- `test_grid_landmarks_count_and_distribution` updated to expect 12
  anchors with distribution across y ∈ {108, 162, 216, 270}.
- New `test_boundary_triangle_catalog_present_with_correct_orientations`
  asserting all 4 triangle records have the right fields and
  orientations (TL/TR `apex_up`; BL/BR `apex_down`).
- `test_registration_cross_catalog_present`.
- `test_black_circle_catalog_present`.

### Real-capture smoke test (manual)
Regenerate the 3-way comparison HTML on dvdrip/snellhd/snellld and
manually inspect the new geometry section. Expected outcomes:
- dvdrip (720×480 padded to 720×486): top + bottom edges clipped by
  ~3 px each, as the DVD master drops 6 lines. Geometry section shows
  this explicitly.
- snellhd, snellld (native 720×486): no clipping detected.
- Registration refit benefit: inlier count rises from current 4/8 to
  ~11/(12+8) anchors; refit residuals tighter.

## Open issues / decisions deferred

- **Exact ideal coordinates of the 4 boundary triangles, the
  registration cross, and the black circle** require a calibration
  pass against the SDI capture via the new `stage2-fiducials` preset
  in tp_calibrate.py. The catalog placeholders in `tp_chart.py` will
  be filled in during the implementation phase, after a calibration
  run produces a JSON the implementer can transcribe.
- **Triangle orientations** (which way each triangle's apex points)
  also depend on the real chart's rendering. Working assumption:
  TL/TR point up (toward top edge); BL/BR point down. To be confirmed
  during calibration.
- **Registration cross template shape**. The chart spec describes a
  black box with a 200 ns white cross. Exact pixel-domain dimensions
  of the cross at 13.5 MHz sampling: 200 ns ≈ 2.7 samples, so a 3 px
  arm width is a reasonable initial template; refine if matchTemplate
  performance is poor.
- **Black circle vs ellipse**. `cv2.fitEllipse` returns an ellipse
  with separate axes; we report both and derive `aspect_ratio_check`
  from their ratio. If `cv2.HoughCircles` proves more robust against
  partial occlusion on real captures, swap during implementation; the
  detector's external contract (return dict shape) is preserved.
- **Per-detector confidence thresholds for the quality_flag** are
  initial guesses that the smoke test will tune.

## Why not the alternatives we considered

- **Pure sequential registration (Stage 1 untouched, Stage 2 only
  measures)**: doesn't address the 4/8 inlier issue that's the
  primary diagnostic finding driving this iteration. The catalog tune
  alone is partial; the Stage 2 fiducials are an obvious second pool
  of anchors that should feed back.
- **Unified single-affine fit (Stage 2 fiducials BEFORE registration)**:
  tightly couples Stage 1 and Stage 2; harder to reason about
  failures; Stage 1 stops working standalone. Sequential-with-
  feedback keeps Stage 1 a usable standalone capability.
- **Register on triangle centroids or apex instead of back corners**:
  the apex is the most clip-vulnerable point on the triangle, so
  registering on it loses robustness exactly where the chart's
  registration marks were designed to be most informative. Centroid
  is OK in principle but conflates "where is the back" and "where is
  the apex" into a single midpoint and loses the per-edge skew info
  the two-back-corner approach surfaces.
- **Single detector function per kind with no dispatcher**: works for
  3-4 detectors but doesn't compose to Stage 3 where new burst
  detectors will share crop/threshold infrastructure. The dispatcher
  layer is the right abstraction now even though it's slightly more
  ceremony.
