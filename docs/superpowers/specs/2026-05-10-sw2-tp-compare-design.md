# SW2 Test Pattern Comparison Tool — Design

Status: design / brainstorming complete, awaiting plan
Date: 2026-05-10 (revised 2026-05-10 with whitepaper findings: NTSC triangle polarity resolved; Stage 3 metric inventory expanded)

## Reference materials

- TPG20/21 manual (PAL chart spec at page 4.25; NTSC differences at page 4.30): `/wintmp/analog_video/tp_compare/ref/tpg20_21_page_58.pdf`
- "What is it?" overview (NTSC): `/wintmp/analog_video/tp_compare/ref/The_Snell_and_Wilcox_Test_Chart_2_What_is_it.pdf`
- Updated NTSC text version: `/wintmp/analog_video/tp_compare/ref/tp2_description_updated_ntsc.txt`
- **Composite Decoder Whitepaper** (decoder-class taxonomy and per-region artifact signatures): `/wintmp/analog_video/tp_compare/ref/compositedecoder_wp_tp2_sections_explained.pdf`. Provides a "Source Picture" reference of the NTSC chart and circles the regions where simple/notch/line-comb/field-comb decoders show their characteristic artifacts. Drives Stage 3 metric design (see below).
- Engineering guides on encoding/decoding, digital video, and standards conversion (general background): `engineer_guide_*.pdf` in the same directory.
- Existing codex starter: `/home/viblio/coding_projects/sw2_analysis/codex/analyze_tartan.py`

## Purpose

Compare how different video pipelines (composite-video processors, decoders,
digitizers) reproduce the **Snell & Wilcox Test Chart #2 (NTSC version)**. This
is a reference-based, region-targeted analysis distinct from the existing
no-reference statistical pipeline (`quality_report.py`, `cross_clip_report.py`),
and lives alongside it without modifying it.

The fundamental approach is to **synthesize a canonical ideal SW2 frame in
software**, then measure how each capture deviates from that ideal in
specific, named regions. Where ideal values exist (75% color bars, gray steps,
geometry positions), we compare absolute deltas. Where the question is one of
fidelity (frequency burst regions), we measure how much modulation a decoder
preserves vs. the ideal grating — which gives a quantitative answer to
"is this rendering correct" without arguing about whether the right output is
houndstooth-black-and-white or gray smear: closer to the ideal grating is
better.

## Scope

The work is staged. Only Stage 1 is fully specified here — Stages 2 and 3 are
sketched so the architecture is known to support them, but their detailed
specs come later.

- **Stage 1** (this spec): top-left tartan + 4-step gray scale, end-to-end
  through synthesis → registration → measurement → comparison HTML.
- **Stage 2** (sketch): geometry / picture-in-raster — boundary triangles,
  registration cross, black circle.
- **Stage 3** (sketch): frequency / burst regions — sine-grating fidelity,
  plus a per-region artifact-metric inventory (cross-color, cross-luma,
  hanging dots, chroma bandwidth) and an optional decoder-class guess,
  driven by the composite-decoder whitepaper's taxonomy. Per-field
  analysis is included where dot crawl matters.

## Non-goals

- Not a no-reference quality tool. The existing `quality_report.py` covers that.
- Not a real-time / live tool. Operates offline on captured video files.
- Not a pass/fail compliance certifier. Produces measurements with absolute
  deltas; downstream interpretation is the operator's.
- Not a multi-frame statistical tool (initially). Stage 1 works on a single
  decoded frame per capture. Multi-frame averaging may be added later if
  noise on real captures requires it.
- Does not perform fancy de-interlacing. Interlaced sources are weaved
  (lines interleaved as-stored) — no `yadif`, no `bwdif`, no field-blending.

## Architecture

Three new sibling scripts at the `video_compare/` repo root, plus one shared
constants/helpers module, mirroring the layout of the existing tools:

```
video_compare/
├─ common.py                # existing — read_frame, decode_command, probe_video, etc.
├─ quality_report.py        # existing — untouched
├─ quality_metrics.py       # existing — untouched
├─ cross_clip_report.py     # existing — untouched
├─ tp_chart.py              # NEW: SW2 chart constants — region table, ideal YUV
│                            #      values, registration landmark catalog
├─ tp_synthesize.py         # NEW: build ideal yuv422p10le reference frame for
│                            #      a target raster (CLI-debuggable; library role)
├─ tp_measure.py            # NEW: extract a frame from a capture, register
│                            #      against ideal, sample patches, write JSON
└─ tp_compare.py            # NEW: combine N tp_measure JSONs into HTML
```

`common.py` is reused for `probe_video`, `decode_command`, frame-decoding
helpers, and `parse_skip_args`. `tp_chart.py` carries SW2 domain knowledge so
it isn't scattered across the three pipeline scripts.

### Why a separate `tp_chart.py`

The synthesizer, measurer, and comparer all need the same things: ideal-frame
region coordinates, ideal YUV values, the named landmark catalog. Keeping
those in one module makes the chart spec versioned, testable in isolation,
and easy to extend across stages without reshaping the pipeline scripts.

### Workflow (mirrors `quality_metrics → cross_clip_report`)

```bash
source venv/bin/activate

# Per-capture measurement
python tp_measure.py /path/to/capture1.mov --frame 60 --output cap1.json
python tp_measure.py /path/to/capture2.avi --frame 60 --output cap2.json

# Comparison report
python tp_compare.py cap1.json cap2.json --output report.html

# Debugging the synthesizer
python tp_synthesize.py --raster 720x486 --output ideal.png   # PNG preview
python tp_synthesize.py --raster 720x486 --output ideal.yuv   # raw 10-bit
```

### Resolution handling (720×480 vs 720×486)

The synthesizer emits at the SDI active raster: 720×486. Captures vary:
- SDI v210 sources (e.g., the existing `tpgSw2_composite_snell{hd,ld}_*.avi`):
  720×486, no padding required.
- DVD-rip ProRes sources (e.g., `sw2_dvdrip_sample.mov`): 720×480. The DVD
  pipeline drops 6 active lines (typically 3 top + 3 bottom). `tp_measure.py`
  pads the capture to 720×486 with the chart's grey-background level
  (Y10 ≈ 502, U=V=512) before registration. Landmarks within the padded
  margin are skipped.
- Other rasters: padded with grey to 720×486 if smaller, or rejected with a
  clear error if the raster doesn't match an expected NTSC active size.

We do not maintain two synthesized ideal-frame variants. Padding the capture
side keeps a single source of truth.

## Reference frame: NTSC SW2 in YUV422p10le, BT.601 limited range

The ideal frame represents what a perfect-decoder would produce after
analog→digital conversion of the canonical SW2 NTSC composite signal. After
decode, the format is 10-bit YUV422 limited range with the standard BT.601
quantization (black=64, white=940 in Y; chroma centered at 512 with full
range 64–960).

### Reference values (from the TPG20/21 manual + chart spec)

The TPG's internal scale uses 10-bit, 588 levels black-to-white, with
`grey_background = 548 / 588 ≈ 50% IRE`. After decode to BT.601 limited range:

| Region | Ideal Y10 (NTSC, limited-range) | Source |
|---|---|---|
| Black | 64 | BT.601 |
| White | 940 | BT.601 |
| Grey background | ≈ 502 (50% IRE) | TPG spec |
| Gray step 20% | 239.2 | `64 + 0.20 × 876` |
| Gray step 40% | 414.4 | `64 + 0.40 × 876` |
| Gray step 60% | 589.6 | `64 + 0.60 × 876` |
| Gray step 80% | 764.8 | `64 + 0.80 × 876` |

For the 75% tartan colors, the ideal Y10/U10/V10 is computed via Rec.601
limited-range matrix from R'G'B' = 0.75 of saturated primary/secondary. This
is the same math the existing codex starter uses; reusing it keeps the spec
verifiable.

The lower-row tartan patches are intentionally lower chroma — codex names
them `magenta_low`, `green_low`, `red_low`, `cyan_low`. Their ideal values
are taken directly from the TPG generator definition (not derived from
75%-of-something) and stored as constants in `tp_chart.py`. We do **not**
back-derive these from a "reduce sat to 25%" recipe — the actual generator
values are the reference.

### NTSC vs. PAL chart differences (translated)

The TPG manual section we have is for PAL. For NTSC SW2:
- 2T pulse width is 250 ns (PAL: 200 ns).
- Subcarrier is 3.58 MHz (PAL: 4.43 MHz). The chart still includes the PAL-SC
  burst region as a comparison artifact.
- 7.5 IRE setup pedestal is added at the analog stage. It is removed during
  any spec-conformant decode to BT.601, so the digitized YUV we measure is
  pedestal-free. Captures where this is mis-handled show up as a uniform
  gray-level offset.

### Synthesizer scope for Stage 1

`tp_synthesize.py` produces only what Stage 1 needs:

- Grey background filling the active picture (Y10 ≈ 502, U=V=512).
- Black rectangular grid lines on the grey, drawn with 168 ns line width
  (≈ 2.3 samples at 13.5 MHz, anti-aliased to subpixel edges).
- Top-left tartan: 4×2 block of color rectangles, sharp edges, uniform
  interior, ideal YUV from `tp_chart.py`.
- 4-step gray strip directly below the tartan, ideal Y10 per the table above.
- Black boundary triangle in the upper-left "below-tartan" cell — included
  because it sits next to the Stage 1 region and is harmless filler. The
  other boundary triangles arrive in Stage 2.

Everything else on the ideal frame is grey-background-with-grid filler. The
synthesizer is structured so adding Stage 2/3 regions is additive — each
region is an independent renderer registered with `tp_chart.py`.

### Region table in `tp_chart.py`

Each region is a record with:

```python
{
  "id": "YEL",                          # short, stable
  "name": "yellow_75_tartan",            # descriptive
  "kind": "tartan_rect",                 # tartan_rect | gray_step | grid_intersection | ...
  "ideal_box": (x, y, w, h),             # in 720×486 ideal coords
  "expected": {"y10": ..., "u10": ..., "v10": ...},
  "sample": {"kind": "center_window", "size_frac": 0.2}  # patch occupies 20% of box width/height
}
```

Patch size is expressed as a *fraction of the box*, not a fixed pixel count —
so it scales correctly when the registration transform implies a different
effective box size on the captured raster. (Codex used a fixed 6×6 window at
720×486; we generalize to "X% of the box" with the codex value as the default
calibration target.)

## Registration (Stage 1)

Goal: a 2D affine transform mapping ideal coordinates → captured-frame
coordinates, robust to the kinds of distortion typical of NTSC processors
(picture H/V offset, very small scale changes, occasional skew from
non-orthogonal sampling).

### Landmark choice

Per operator guidance: **avoid landmarks in high-frequency-prone regions** —
tartan box corners, burst region edges, the chart border, anywhere ringing
or comb-filter artifacts will corrupt the position estimate.

Stage 1 uses **black-grid-line intersections on the grey background** as
landmarks. The grid is sin²-shaped 168 ns lines drawn by the TPG itself —
specifically engineered for registration-quality measurement. A grid-line
intersection on flat grey background is the safest signal we have:
no chroma involved, surrounded by uniform low-frequency content, sub-pixel
detectable.

`tp_chart.py` contains a curated landmark list of 6–8 grid intersections,
distributed across the picture but biased toward the upper third (where
Stage 1's region of interest lives). At least two anchors are placed lower
on the chart so the affine fit isn't degenerate when the upper anchors all
agree. Each landmark record is:

```python
{"id": "GX_3_2", "ideal_x": ..., "ideal_y": ..., "search_window_px": 24}
```

The `search_window_px` is sized so registration tolerates ±10 px of initial
mis-positioning before launch. Landmarks too close to known-troublesome
regions (chart border, tartan edges, burst boxes, the zone plate) are
explicitly excluded.

### Detector

For each landmark, within its search window in the captured frame:

1. Threshold to find dark pixels (Y10 below `0.3 × grey_background`,
   i.e. ≈ 150).
2. Verify the dark cluster looks like a "+" cross: at least one
   approximately-horizontal run AND one approximately-vertical run of dark
   pixels intersect within the window. A single dark blob (no cross
   structure) is rejected.
3. Compute the dark-cluster centroid weighted by `(grey_background - Y10)`
   — gives subpixel accuracy.
4. Return `(detected_x, detected_y, confidence)`.

### Transform fit

Given `N` detected → ideal landmark pairs (with `N >= 4` for an affine fit):

- Run RANSAC with a 1.5 px inlier threshold to reject bad detections.
- Fit a 2D affine transform on inliers (translation, scale x/y, shear,
  rotation; 6 DoF).
- Report fit residuals: mean and max landmark-to-mapped-ideal distance, in
  pixels, on the inlier set.
- If fewer than 4 inliers, registration fails and `tp_measure.py` exits with
  an explanatory error and a debug dump (a PNG with detected landmarks
  overlaid on the captured frame).

Residuals above a threshold (placeholder: 2 px mean / 4 px max — to be
calibrated on real data) are flagged in the JSON `_meta.registration` block
and surfaced in the comparison report.

## Measurement (`tp_measure.py`)

Pseudocode:

```
probe capture                 # ffprobe: w, h, pix_fmt, field_order, codec
extract single frame at index # default 60; configurable via --frame
                              # ffmpeg with -pix_fmt yuv422p10le, NO deinterlace filter
if field_order indicates progressive:
    emit warning to stderr; flag in JSON _meta
if (w, h) != (720, 486):
    pad to 720×486 with grey-background YUV (Y10≈502, U=V=512)
    record padding offsets in _meta

run registration → affine M
for each Stage-1 region in tp_chart.py:
    map region center via M to capture coords
    sample center patch (size = box × sample.size_frac)
    compute mean Y10, U10, V10
    compute deltas vs region.expected
    compute sat_pct_vs_ideal (chroma magnitude ratio)

write per-capture JSON
```

### Per-capture JSON shape

```json
{
  "_meta": {
    "capture": "/path/to/capture.mov",
    "frame_index": 60,
    "raster_in": [720, 480],
    "raster_processed": [720, 486],
    "padding_offsets": {"top": 3, "bottom": 3, "left": 0, "right": 0},
    "field_order": "tb",
    "progressive_warning": false,
    "decode": {"deinterlace": "none-weave-only", "pix_fmt": "yuv422p10le"},
    "registration": {
      "affine": [[a, b, tx], [c, d, ty]],
      "residuals_px": {"mean": 0.4, "max": 1.1, "inliers": 7, "total": 8},
      "quality_flag": "ok"
    },
    "ideal_frame_md5": "...",
    "tp_chart_version": 1,
    "tool_version": "tp_measure 0.1"
  },
  "tartan": [
    {
      "id": "YEL",
      "name": "yellow_75_tartan",
      "ideal_yuv10": [639.1, 172.6, 567.7],
      "measured_yuv10": [...],
      "delta_yuv10": [...],
      "sat_pct_vs_ideal": 76.4,
      "patch_size_px": [3, 3],
      "patch_center_capture_xy": [...]
    }
    /* ... 7 more tartan patches ... */
  ],
  "grays": [
    {
      "id": "G1",
      "ideal_y10": 239.2,
      "measured_y10": 233.8,
      "delta_y10": -5.4,
      "u10": 512.0,
      "v10": 512.0
    }
    /* ... G2, G3, G4 ... */
  ]
}
```

## Comparison HTML (`tp_compare.py`)

Stage 1 report has three sections, dark-themed to match the existing tooling:

1. **Registration summary** (per capture): mean/max residual, decoded
   affine parameters (offset, scale x/y, shear, rotation), inlier count.
   Cells are red-flagged when residual exceeds the threshold or when the
   progressive-warning flag is set.

2. **Tartan deltas table**: rows are captures, columns are the 8 patches.
   Each cell shows ΔY/ΔU/ΔV color-coded by magnitude, plus a small swatch
   pair (measured RGB | ideal RGB). Hovering shows raw measured + ideal
   YUV10 values.

3. **Gray deltas table**: rows are captures, columns are G1–G4. Each cell
   shows ΔY10 (color-coded). A small linearity plot shows measured Y10 vs
   ideal Y10 across the 4 steps, one line per capture — non-linear gamma
   or compression curves jump out instantly.

No overall composite ranking yet. These are absolute, directional deltas;
ranking semantics (e.g., "which capture is closest to ideal across all
regions?") will be designed when we see real data from multiple processors.

## Testing

A new test file `test_cases/test_tp_chart.py` (and helpers as needed)
validates:

- Synthesizer at 720×486 produces a frame with Y10 ≈ 502 in the grey
  background, Y10 ≈ 64 along the grid lines, and the expected tartan/gray
  YUV in the relevant box centers — all checked by sampling, no eyeballing.
- Registration end-to-end on the synthesized ideal: feed `tp_measure.py`
  the synthesized frame and confirm it returns an identity transform with
  near-zero residuals and zero deltas.
- Registration robustness: synthesize the ideal, shift it by `(dx, dy)`
  and rotate slightly, run registration, confirm the recovered affine
  inverts the applied transform within a small tolerance.
- Padding: feed a 720×480 cropped version of the synthesized ideal and
  confirm padding + registration recovers the correct geometry.

The existing `test_cases/test_metrics.py` is untouched.

## Stage 2 — Geometry / picture-in-raster (sketch)

Goal: answer "where is the picture, how big is it, is it square?"

Extends the registration / fiducial infrastructure of Stage 1 with new
landmark kinds:

- 4 boundary triangles (top-left, top-right, bottom-left, bottom-right):
  centroid of dark cluster within bounded search window. The four centroids
  define the picture's bounding rectangle.
- "Registration Check" box (small black square containing 200 ns white
  cross): sub-pixel template match on the cross intersection.
- Black circle (diameter = picture height): fit a circle to the dark-ring
  centerline pixels; compare diameter to bounding-rectangle height to
  detect aspect distortion / non-square pixels.

Outputs added to `_meta.geometry`:
- `active_picture_box` in capture pixel coords (from triangle centroids)
- `picture_extent_px` (width, height)
- `picture_offset_in_raster` (delta from ideal position)
- `aspect_ratio_check` (circle aspect, 1.0 = square)
- `arrow_tip_coords` (named, per-tip, for the operator's specific reading)

The Stage 1 affine transform continues serving sampling; Stage 2 adds
*interpretation* of fiducial positions as geometry measurements.

## Stage 3 — Frequency / burst regions (sketch)

Goal: answer "for each frequency region, how faithfully is the captured
content reconstructed compared to an ideal sine grating?"

`tp_synthesize.py` learns to render the frequency regions per the chart
spec (each is an independent renderer registered with `tp_chart.py`):

- 3.58 MHz vertical bursts; 4.43 MHz vertical bursts (PAL-SC region on
  the NTSC chart); 4.286 MHz SECAM bell.
- 100 / 200 / 300 TVL oblique bursts (vertical response).
- 300 / 400 TVL vertical and diagonal gratings (H/D frequency response).
- 1.5–5.5 MHz frequency wedge; radial wedge up to 200 TVL; zone plate
  up to 5.5 MHz.

Each is rendered as a true sine-modulated grating at the stated spatial
frequency. The ideal frame becomes the literal answer to "what should a
perfect decoder produce."

### Measurement model per burst ROI

The composite-decoder whitepaper enumerates a hierarchy of decoder
strategies — simple low-/high-pass, notch, line comb, field/temporal
comb, frame comb, adaptive — each producing a characteristic artifact
signature on TP2's burst regions. Stage 3 measures these signatures
explicitly rather than collapsing them all into a single MTF number.

After registration, per-ROI metrics:

- **Modulation depth**: `(p95 − p5) / mean` of the Y plane within the ROI
  (or the chroma plane for chroma-modulated bursts) — proxy for MTF
  retained at that frequency.
- **Cross-correlation** against the registered ideal grating —
  orientation-aware fidelity score (catches a decoder that low-passes vs.
  one that produces phase-shifted content).
- **Cross-color leakage**: chroma energy (|U − 512| + |V − 512|, or
  RMS chroma magnitude) in regions that should be Y-only — e.g., the
  3.58 MHz luma burst, the 100/200/300 TVL bursts, the radial wedge.
  Detects notch-filter leakage, simple-decoder bandwidth limits, and
  inadequate 2D/3D adaptive comb performance.
- **Cross-luma residual**: high-frequency Y energy (after a high-pass
  filter) inside regions that are pure-chroma in the source — e.g., the
  100% red box, the magenta chroma-staircase. Detects simple/notch
  decoders that fail to subtract chroma cleanly from luma.
- **Hanging-dot intensity**: localized dark/bright vertical-line
  artifacts in narrow strips above and below sharp vertical chroma
  transitions (red box edges, magenta-staircase edges, the 1.5 MHz Y/C
  burst region). Specifically a line-comb signature; absent in
  field/frame-comb decoders.
- **Chroma bandwidth**: amplitude retained in the 1.0 / 0.5 / 1.5 MHz
  blue-yellow and green-magenta Y/C-timing bursts. A direct proxy for
  the decoder's chroma low-pass cutoff.

Each region in `tp_chart.py` carries metadata declaring which of these
metrics applies to it (and what "ideal" looks like there), so the
measurement loop can dispatch to the correct metric per region.

### Decoder-class classification

Combinations of these per-region metrics map to decoder families with
clear physical interpretations (per the whitepaper):

- High cross-color **and** high cross-luma **and** narrow chroma
  bandwidth → simple low-/high-pass or notch decoder.
- Low cross-color **but** strong hanging-dot signature → line-comb
  decoder (good static H/V separation; struggles at sharp vertical
  chroma transitions).
- Low cross-color **and** low hanging dots **but** residual artifacts
  on the upper-mid frequency oblique bursts → field/frame-comb decoder
  (good static performance; loses on motion, but our static charts
  expose the static behavior only).
- All artifacts low across the board → adaptive decoder.

Stage 3 may emit a per-capture decoder-class guess as a derived datum
in `_meta`. This is interpretive (heuristic, not authoritative), but
the underlying per-region metrics are unambiguous and remain the
primary report content.

### Field-aware analysis

Bursts at or near subcarrier-related frequencies show inter-field
differences (dot crawl is exactly the inter-field high-frequency
difference). Stage 3 measures these ROIs per-field separately and reports
both, with an inter-field-disagreement metric. The Stage 1 weave decode
remains the default; Stage 3 splits weaved frames back into fields for
the affected ROIs.

### Resolves "what is correct" for high-frequency content

A notch-filter decoder produces gray smear in the 400 TVL diagonal box →
low modulation depth, low correlation → quantitatively worse, with a
clear physical interpretation. A 2D adaptive comb decoder produces the
houndstooth → high modulation, high correlation → quantitatively better.
We don't argue "should it be black/white or gray" — we measure *fidelity
to the ideal grating*, and the houndstooth wins because it's literally
closer to the original signal.

### Comparison HTML extension

- MTF / fidelity table per burst region.
- Per-capture frequency response curve (x-axis frequency / TVL, y-axis
  modulation retained), one line per capture.
- Cross-color, cross-luma, hanging-dot, and chroma-bandwidth tables —
  each scoped to the regions where that metric is meaningful, with the
  ideal=0 (or ideal-band) baseline drawn alongside.
- Decoder-class guess per capture (with a confidence note and the
  per-metric pattern that drove it), to give readers a quick mental
  hook before they dive into the numbers.
- Side-by-side region thumbnails so the visual story is preserved
  alongside the numbers.

## Open issues / decisions deferred

- **Registration residual thresholds**: `2 px mean / 4 px max` are
  placeholders. Calibrate against real captures — likely tighter on SDI
  sources, looser on DVD-rip sources where mpeg2 noise dilutes the grid
  lines.
- **Patch size as fraction of box**: default `size_frac=0.2` matches the
  codex 6×6 patch on a ~30 px-wide box (codex tartan/gray centers are
  spaced 30 px apart at 720×486). Verify on real captures that this
  stays inside the visually-uniform interior of each tartan box (away
  from the high-frequency edges).
- **Ranking semantics**: Stage 1 deltas are absolute, directional, and
  per-region. A composite "which capture is best" is intentionally
  deferred until we have a multi-capture data corpus to design ranking
  against.
- **Frame-selection policy**: default frame index 60 follows codex
  precedent. If real captures show the test pattern is unstable or
  mis-framed at index 60 in some sources, we may switch to "median across
  frames N..M" or an explicit per-capture override.
- ~~**NTSC version of the boundary triangles**~~ — RESOLVED: per the
  composite-decoder whitepaper "Source Picture" (NTSC), the boundary
  markers are **black filled triangles** on the grey background. The
  Stage 1 synthesizer's `_draw_boundary_triangle_upper_left` already
  draws black, which is correct. Stage 2's detector should look for
  dark-cluster centroids inside known cells (not white-on-black template
  matching).
- **Grey background level**: derived from spec at 50% IRE → Y10 ≈ 502.
  Verify against the codex captures; if real captures consistently
  diverge by a known offset, treat as a measurement (a "grey-level
  offset" datum) rather than adjusting the synthesizer.

## Why not the alternatives we considered

- **Pure analytical / spec-based reference**: would require accepting that
  some regions (frequency bursts) have no single "correct" rendering and
  can only be measured, not graded. The synthesized-ideal approach gives
  a literal correct answer everywhere.
- **Known-good capture as ground truth**: bakes in the chosen capture's
  flaws and makes "is this color actually right?" unanswerable in
  absolute terms.
- **Mixed analytical + known-good**: workable, but gives every measurement
  two interpretations depending on its reference type — synthesized ideal
  unifies the model.
- **Synthesizer-first** (build the whole chart before any measurement):
  no comparable output until late in the project, and burst-region
  synthesis details will reshape after we see real captures anyway. The
  vertical-slice path here delivers usable output at the end of Stage 1.
- **Cross-correlation registration only** (no fiducials): only gives
  global translation/scale, not the rich geometry data Stage 2 needs.
- **Hand-typed per-capture calibration**: doesn't actually answer the
  geometry question (registration values are typed, not measured) and
  breaks for any new capture.
- **Per-capture markdown reports** (the codex shape): doesn't scale to
  multiple captures and doesn't give cross-capture comparison.
