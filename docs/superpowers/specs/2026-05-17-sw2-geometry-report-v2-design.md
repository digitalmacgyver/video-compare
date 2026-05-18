# SW2 Geometry Report v2 — Layperson-Friendly Design

Status: design / brainstorming complete, awaiting plan
Date: 2026-05-17

## Purpose

The Stage 2 geometry pipeline (`tp_register.detect_geometry` and
`tp_compare.render_geometry_section`) currently emits engineery numbers:
`aspect_ratio_check=0.9082`, `picture_offset_from_ideal dx=178.8`,
`corner_skew_px {top_vs_bottom_width_diff: 1.0, left_vs_right_height_diff: 0.0}`,
etc. These are diagnostics, not a report a layperson can read.

This iteration rewrites the **report layer** so each capture's
geometry summary reads like:

- "Top edge arrowhead spacing: 358.0 px (ideal 357, +1.0)."
- "Picture center shifted +1.2 px right, -0.3 px up."
- "Horizontal scale 99.8%, vertical scale 100.2%."
- "Keystone: top is 2.5 px wider than bottom."
- "Displayed circularity 1.00 (round)."

The change is intentionally **report-layer only**. Detectors, registration,
and other report sections (tartan, gray, frequency, artifacts,
decoder-class) are untouched.

This iteration also adds a **directory wrapper** so the combined report
can be produced in one step on a directory of captures (the
`/wintmp/analog_video/tp_compare/` directory presently holds 6 captures).

## Scope

In scope:

- New `geometry.derived.summary` block in tp_measure JSON, populated by
  `tp_register._derive_geometry`. Contains arrow-spacing-vs-ideal,
  picture displacement, horizontal/vertical scale, keystone, and a
  PAR-aware circle block (horizontal & vertical diameters, displayed
  circularity, rotation).
- Fix to `_detect_black_circle` so it returns `rx_horizontal_px` and
  `ry_vertical_px` (direction-preserving) computed from the pass-2
  midline bounding box. The existing `rx`/`ry` fields stay for
  back-compat (defined as `min`/`max`).
- Rewrite of `tp_compare.render_geometry_section` so each per-capture
  panel reads as prose, using only `geometry.derived.summary` plus
  `geometry.derived.clip_detected`. The old engineery fields are no
  longer surfaced in the report (but stay in the JSON).
- New `tp_report.py` wrapper script: `python tp_report.py <directory>
  --output <html>` enumerates capture files, runs `tp_measure` per
  file (skipping ones with an existing `_stage2.json` unless
  `--force`), then calls `tp_compare.render_page` on the resulting
  JSONs.
- Synthetic test cases for the new summary fields (known shift, known
  scale, round circle in raster).

Out of scope:

- Other report sections (tartan, gray, luma scale, frequency,
  artifacts, decoder-class, sampling diagnostics) — these will be
  refined in subsequent iterations as the user reviews them.
- Re-tuning detection thresholds. Same detectors, same numbers; only
  the derived/reported numbers change.
- Multi-frame averaging.
- A dynamic chart-spec-from-PAR aware ideal target (e.g., a different
  ideal frame size). The chart is canonically 720x486 and that's
  baked into `tp_chart`.

## Non-goals

- Not a re-architecture of how registration is done.
- Not a change to per-fiducial detectors beyond preserving direction
  in the ellipse-axis output.

## Reference materials

- Stage 2 design: `docs/superpowers/specs/2026-05-11-sw2-stage2-geometry-design.md`
- Chart layout: `docs/sw2_chart_layout.md`
- TPG20/21 manual:
  `/wintmp/analog_video/tp_compare/ref/tpg20_21_page_58.pdf`
- "What is it?" overview:
  `/wintmp/analog_video/tp_compare/ref/The_Snell_and_Wilcox_Test_Chart_2_What_is_it.pdf`

## Reference values (chart spec)

All values below come from `tp_chart.BOUNDARY_TRIANGLES`,
`tp_chart.BLACK_CIRCLE`, `tp_chart.IDEAL_PICTURE_BOX`, and
`tp_chart.NTSC_PAR_X_OVER_Y`. **Nothing in the new derive/render code
hard-codes these — they are pulled from the catalogs at runtime.**

| Quantity | Source | Value |
|---|---|---|
| TL apex (ideal) | `BOUNDARY_TRIANGLES[id=TL].ideal_apex` | (181, 1) |
| TR apex (ideal) | `BOUNDARY_TRIANGLES[id=TR].ideal_apex` | (538, 1) |
| BL apex (ideal) | `BOUNDARY_TRIANGLES[id=BL].ideal_apex` | (181, 484) |
| BR apex (ideal) | `BOUNDARY_TRIANGLES[id=BR].ideal_apex` | (538, 484) |
| Top/bottom ideal spacing | `TR.x - TL.x` | 357 px |
| Left/right ideal spacing | `BL.y - TL.y` | 483 px |
| Ideal raster center | `(IDEAL_PICTURE_BOX.right/2, IDEAL_PICTURE_BOX.bottom/2)` | (359.5, 242.5) |
| Ideal circle radius | `BLACK_CIRCLE.expected_radius_px` | 243 |
| PAR (x/y) for NTSC SD | `NTSC_PAR_X_OVER_Y` | 11/10 |

## New `geometry.derived.summary` schema

```python
summary = {
    "arrow_spacings_px": {
        "top":    {"actual": float, "ideal": float, "delta": float},
        "bottom": {"actual": float, "ideal": float, "delta": float},
        "left":   {"actual": float, "ideal": float, "delta": float},
        "right":  {"actual": float, "ideal": float, "delta": float},
    },
    "picture_center_offset_px": {"dx": float, "dy": float},
    "picture_scale_pct": {"horizontal": float, "vertical": float},
    "keystone_px": {
        "horizontal_top_minus_bottom": float,
        "vertical_left_minus_right": float,
    },
    "circle": {
        "horizontal_diameter_px": float | None,
        "vertical_diameter_px":   float | None,
        "expected_h_over_v_for_round": float,   # = NTSC_PAR_X_OVER_Y
        "actual_h_over_v":            float | None,
        "displayed_circularity":      float | None,  # 1.0 = round in display
        "rotation_deg":               float | None,
    },
}
```

Computation:

- **Arrow spacings**:
  - `top.actual    = TR.x - TL.x`
  - `bottom.actual = BR.x - BL.x`
  - `left.actual   = BL.y - TL.y`
  - `right.actual  = BR.y - TR.y`
  - `ideal_top = ideal_bottom = TR_spec.x - TL_spec.x`
  - `ideal_left = ideal_right = BL_spec.y - TL_spec.y`
  - `delta = actual - ideal`
- **Picture center offset**: mean of four detected apex coords vs ideal
  raster center.
- **Picture scale (%)**:
  - `horizontal = mean(top.actual, bottom.actual) / ideal_top * 100`
  - `vertical   = mean(left.actual, right.actual) / ideal_left * 100`
- **Keystone**:
  - `horizontal_top_minus_bottom = top.actual - bottom.actual`
  - `vertical_left_minus_right   = left.actual - right.actual`
- **Circle**:
  - `horizontal_diameter_px = 2 * rx_horizontal_px`
  - `vertical_diameter_px   = 2 * ry_vertical_px`
  - `actual_h_over_v        = rx_horizontal_px / ry_vertical_px`
  - `displayed_circularity  = actual_h_over_v / NTSC_PAR_X_OVER_Y`

The block is set to `None` for any sub-field that depended on a missing
fiducial (e.g. circle block is mostly `None` when `fiducials.circle` is
None; arrow_spacings is `None` if any of the four triangles was not
detected).

## Direction-preserving ellipse axes

`tp_register._detect_black_circle` currently returns:

```python
rx = min(axis_a, axis_b) / 2.0   # loses direction
ry = max(axis_a, axis_b) / 2.0
```

This is replaced with bounding-box of the pass-2 midline point cloud:

```python
rx_horizontal_px = (mxs.max() - mxs.min()) / 2.0
ry_vertical_px   = (mys.max() - mys.min()) / 2.0
```

These are added alongside the existing `rx`/`ry` (which keep the
`min`/`max` semantics for back-compat — `aspect_ratio_check` continues
to be derivable from them).

The bounding-box derivation is correct for an axis-aligned ellipse with
small tilt (real-chart rotation has been observed ≤ 1° in current
captures), and a clear failure mode for severe tilt (the bounding box
expands as tilt approaches 45°). Severe tilt is already a warn signal
elsewhere; the rotation_deg field stays in the JSON for diagnostics.

## HTML report layout per capture

`render_geometry_section` renders one panel per capture. The new layout:

```
<heading>: <capture name>  [ok / warn / failed]

Arrowhead spacing (vs chart spec):
   Top edge      358.0 px  (ideal 357, +1.0)
   Bottom edge   355.5 px  (ideal 357, -1.5)
   Left edge     484.0 px  (ideal 483, +1.0)
   Right edge    483.5 px  (ideal 483, +0.5)

Picture displacement:
   Center shifted +1.2 px right, -0.3 px up.
   Horizontal scale 99.8%, vertical scale 100.2%.
   Keystone: top is 2.5 px wider than bottom;
             left is 0.5 px taller than right.

Clip detection:
   TL apex visible, TR visible, BL visible, BR visible.

Circle (PAR-aware, NTSC 10:11):
   Horizontal diameter 532.0 px, vertical diameter 484.0 px.
   Actual H/V ratio 1.099; expected 1.100 for a round circle.
   Displayed circularity 1.00
     (>1 = stretched horizontally, <1 = stretched vertically).
```

Color-coding kept via the existing `_delta_class_offset` thresholds.
Defaults to apply on the new fields:

| Field | green | yellow | red |
|---|---|---|---|
| `arrow_spacings_px.*.delta` (abs) | < 2 | < 5 | ≥ 5 |
| `picture_center_offset_px.*` (abs) | < 2 | < 5 | ≥ 5 |
| `picture_scale_pct.*` deviation from 100 | < 1 pp | < 3 pp | ≥ 3 pp |
| `keystone_px.*` (abs) | < 2 | < 5 | ≥ 5 |
| `displayed_circularity` deviation from 1.0 | < 0.02 | < 0.05 | ≥ 0.05 |

Existing sub-sections kept as-is in the panel:

- Registration cross (offset + aperture symmetry) — already
  layperson-friendly enough; no changes this iteration.
- Refit benefit (inliers + residuals) — diagnostic, kept.
- Fiducial crops — useful visual context, kept.

## `tp_report.py` wrapper

New CLI script. Single-pass directory-driven workflow.

```
python tp_report.py <directory>
       --output <html>
       [--pattern '*.avi *.mkv *.mov *.mp4']
       [--outdir <jsondir>]          # default = directory
       [--force]                      # re-measure even if JSON exists
       [--name <prefix>]              # JSON name prefix
       [--frame-index N]              # passthrough to tp_measure
```

Behavior:

1. Enumerate `directory` for files matching pattern (default `.avi`,
   `.mkv`, `.mov`, `.mp4`).
2. For each capture, target JSON path is
   `<outdir>/<stem>_stage2.json`. If it exists and `--force` is not
   set, skip the measurement step (re-uses the JSON).
3. Otherwise call `tp_measure.measure(input=cap_path, frame_index=…)`
   and write the JSON. (We may need to expose a Python entry point in
   `tp_measure.py` if it currently only exposes `_main`.)
4. After all measurements, call `tp_compare.render_page` on the list
   of JSONs and write to `--output`.

Failure mode: if `tp_measure` raises on a single capture, log a warning
to stderr and continue with the rest.

Subdirectories of the input directory are **not** recursed by default
(the test corpus has captures both at the top level and in subdirs;
the operator points the tool at one directory at a time for now).

## Architecture

```
tp_chart.py        # UNCHANGED (catalogs are the source of truth for
                   #            ideal apex coords, ideal raster, PAR).

tp_register.py     # MODIFY: _detect_black_circle adds rx_horizontal_px
                   #         and ry_vertical_px to its return dict.
                   # MODIFY: _derive_geometry populates new
                   #         "summary" sub-block.

tp_measure.py      # UNCHANGED (already calls register_with_geometry
                   #            and embeds geometry into JSON; the new
                   #            fields ride along automatically).
                   # Optionally: expose `measure()` Python entry point
                   #            (currently `_main` only) — only needed
                   #            if it doesn't already exist; check.

tp_compare.py      # MODIFY: render_geometry_section rewrite to
                   #         consume geometry.derived.summary and emit
                   #         the new layperson prose. Old fields are no
                   #         longer surfaced but still exist in the
                   #         JSON.

tp_report.py       # NEW: directory wrapper, calls tp_measure +
                   #      tp_compare.

test_cases/
  test_geometry_summary.py   # NEW: synthetic-shift, synthetic-scale,
                             # round-circle tests against the new
                             # summary block.
```

## Data flow

```
directory of N captures
   │
   ▼
tp_report.py ──── per file ────► tp_measure.measure(...)
   │                                  │
   │                                  ▼
   │                          <stem>_stage2.json
   │                          (geometry.derived.summary)
   ▼
collect N JSONs ──────────────► tp_compare.render_page ──► report.html
```

## Backwards compatibility

- Old `_stage2.json` files without the `geometry.derived.summary` block
  are rendered with a "summary missing — re-run tp_measure" note. They
  are not auto-upgraded at render time.
- The old `geometry.derived` raw fields (`active_picture_box`,
  `picture_extent_px`, `corner_skew_px`, `arrow_tip_coords`,
  `clip_detected`, `cross_offset_from_ideal`, `aperture_symmetry`,
  `aspect_ratio_check`, `diameter_vs_picture_height`,
  `circle_fit_rms`) stay in the JSON unchanged.
- The detector's `rx`/`ry` fields stay (defined as `min`/`max`); only
  add `rx_horizontal_px`/`ry_vertical_px`.

## Testing

New `test_cases/test_geometry_summary.py`:

- **Synthesized round capture** (using `tp_synthesize.synthesize()`
  with no shifts/scales): assert
  - All four arrow spacings within 1 px of ideal.
  - `picture_center_offset_px` within 1 px of zero.
  - `picture_scale_pct.*` within 0.5 of 100.
  - `keystone_px.*` within 1 px.
  - `displayed_circularity` within 0.02 of 1.0.
  - The synthesizer is updated in this iteration to render the black
    ring as PAR-elliptical (horizontal semi-axis = vertical semi-axis
    × NTSC_PAR_X_OVER_Y), matching real captures. With that change,
    `displayed_circularity ≈ 1.0` on the round-in-display synth case.
- **Shifted capture** (synthesizer + manual `shift_dx`, `shift_dy`):
  assert `picture_center_offset_px` matches within 1 px.
- **Scaled capture** (synthesizer + manual `scale_x`, `scale_y`):
  assert `picture_scale_pct` matches within 0.5 pp.

Smoke test:

- Run `python tp_report.py /wintmp/analog_video/tp_compare/ --output
  /tmp/tp_v2.html` and confirm 6 panels render with sensible numbers.

## Synthesizer alignment

The current synthesizer renders the black ring as round-in-raster (no
PAR baked in). Real captures show PAR-elliptical rings. To keep the
"displayed_circularity ≈ 1.0 on a clean signal" expectation true for
both synth and real, this iteration also updates `tp_synthesize` to
render the ring with horizontal semi-axis = vertical semi-axis ×
`NTSC_PAR_X_OVER_Y`.

This is the smallest change that makes synth match real. The existing
detector annulus is already PAR-shaped, so detection still finds it.
