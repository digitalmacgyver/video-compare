# Stage 3 — Frequency Response, Decoder Artifacts, Decoder-Class Classification

**Status:** Design, approved 2026-05-12.

**Builds on:** Stage 1 (tartan/gray sampling) and Stage 2 (registration + geometry +
affine matrix), already in main.

## Goal

For each ProRes-encoded capture of the Snell & Wilcox Test Chart #2 (SW2 NTSC):

1. Quantify **frequency response** at known burst/wedge regions on the chart (3.58 MHz,
   4.43 MHz, 4.286 MHz, diagonal 300/400 tvl, MHz-wedge 3/4/5, TVL-100 scale).
2. Detect **decoder artifacts** at known chroma-transition regions: dot crawl,
   hanging dots, cross-color, cross-luma, and zone-plate chroma leak.
3. Infer a discrete **decoder class** (`notch` / `line_comb` / `temporal_comb_or_adaptive`
   / `undetermined`) from the combined signature, with confidence and supporting
   evidence.

These outputs are the primary value of the whole tp_compare pipeline — they expose
the analog-decoder behavior of each capture device under test.

## Why now

- Stages 1 and 2 give us reliable per-frame sampling at chart-anchored coordinates
  via the affine matrix. Stage 3 reuses that machinery and adds new sample regions
  without touching registration/geometry code.
- The composite-decoder whitepaper (`ref/compositedecoder_wp_tp2_sections_explained.pdf`)
  enumerates the SW2 chart features that diagnose simple vs notch vs line-comb vs
  temporal-comb vs adaptive decoders. We have the chart, the geometry pipeline, and
  the reference theory; we just need the measurement layer.

## Reference material

- `/wintmp/analog_video/tp_compare/ref/tp2_description_updated_ntsc.txt` — chart feature
  list with frequencies.
- `/wintmp/analog_video/tp_compare/ref/compositedecoder_wp_tp2_sections_explained.pdf` —
  decoder taxonomy and the per-chart-feature signatures for each decoder class
  (screenshots of simple/notch/line-comb/field-comb outputs with the diagnostic
  regions circled).
- `docs/sw2_chart_layout.md` and `docs/sw2_chart_layout.json` — cell-by-cell map of
  the chart, used to place the new sample boxes.

## Architecture

Three new modules join the existing `tp_*` family:

```
tp_chart.py        + BURST_REGIONS catalog
                   + ARTIFACT_REGIONS catalog (incl. zone-plate chroma-leak region)
tp_freq.py         (new) frequency-response measurement per burst region
tp_artifacts.py    (new) dot crawl / hanging dots / cross-color / cross-luma /
                         zone-plate chroma leak
tp_classify.py     (new) decoder-class heuristic, consumes tp_freq + tp_artifacts
tp_measure.py      + invokes Stage 3 after Stage 2; embeds output in JSON
tp_compare.py      + renders frequency_response, artifacts, decoder_class sections
cross_clip_report  + overlay frequency curves; artifact table; decoder-class column
tp_synthesize.py   + draws burst patterns and saturated chroma blocks so synthesized
                     frames exercise Stage 3
tp_fixtures.py     + degradation helpers (inject cross-color, inject hanging dots,
                     band-limit luma) for unit tests
tp_calibrate.py    + stage3-bursts and stage3-artifacts presets
```

Each new module has one clear responsibility and depends only on `tp_chart`, numpy,
and (for FFTs) scipy. `tp_classify` depends on the dict outputs of `tp_freq` and
`tp_artifacts` — not on their internals — so all three can be tested independently.

## Catalog extensions

### `tp_chart.BURST_REGIONS`

A list of dicts, each with:

```python
{
    "id":            "BURST_3p58",                  # unique short id
    "kind":          "burst_vertical",              # see "Kinds" below
    "frequency_MHz": 3.58,                          # expected dominant frequency
    "ideal_box":     (x, y, w, h),                  # chart-coords sample box
    "sample":        {"kind": "center_window", "size_frac": 0.6},
    "label":         "3.58 MHz vertical (NTSC SC)", # for HTML reports
}
```

Kinds (every burst is luma-only on the chart; chroma is not modulated):
- `burst_vertical` — vertical stripes (frequency along x). Modulation extracted from
  a horizontal line cross-section through the box center.
- `burst_diagonal` — diagonal stripes. The region dict carries `stripe_angle_deg`
  (clockwise from horizontal); modulation is extracted from a line at
  `stripe_angle_deg + 90°` through the box center, sampled with bilinear
  interpolation.
- `burst_horizontal` — horizontal stripes (almost-horizontal; frequency along y).
  Used for the "Frequency/Vertical Response" 100/200/300 tvl bursts if added.
- `wedge_segment` — a single-frequency slice of the col-10 narrowing luma wedge; the
  region's `frequency_MHz` is the segment's nominal center frequency (3, 4, or 5 MHz).

Initial region set (8 entries):
- `BURST_3p58` — cell (2,2), 3.58 MHz vertical, NTSC SC frequency.
- `BURST_4p43` — cell (2,11), 4.43 MHz vertical, PAL SC. Used as a cross-color anchor
  on NTSC captures (an NTSC decoder shouldn't pass this through cleanly).
- `BURST_4p286_SECAM` — cell (8,2), 4.286 MHz vertical.
- `BURST_300TVL_DIAG` — cell (2,5), diagonal 300 tvl.
- `BURST_400TVL_DIAG` — cell (2,8), diagonal 400 tvl.
- `WEDGE_3MHz` — cell (4,10), segment of the MHz-wedge labeled "3".
- `WEDGE_4MHz` — cell (5,10), segment labeled "4".
- `WEDGE_5MHz` — cell (6,10), segment labeled "5".

Operator-tuned coordinates land via `tp_calibrate.py --preset stage3-bursts`.

### `tp_chart.ARTIFACT_REGIONS`

Each entry:

```python
{
    "id":            "HD_RED_TOP",
    "artifact_kind": "hanging_dots",                # see "Artifact kinds" below
    "ideal_box":     (x, y, w, h),
    "sample":        {"kind": "center_window", "size_frac": 1.0},
    "neighbor_baseline_y10": 502,                   # optional: expected baseline
    "label":         "Hanging dots above red 100% block",
}
```

Artifact kinds and what each measures:

- `hanging_dots` — vertical-color-transition cross-luma. A thin (5 px tall) horizontal
  strip immediately above/below a saturated chroma block. The baseline is the grey
  background (`Y10 ≈ 502`). Cross-luma from a line-comb decoder appears as a row
  of bright/dark dots at SC frequency.
  - `HD_RED_TOP`, `HD_RED_BOTTOM` — flanking the red 100% block in cells (9,10)–(9,12).
  - `HD_MAGENTA_TOP`, `HD_MAGENTA_BOTTOM` — flanking the magenta steps in (9,1)–(9,3).
- `dot_crawl` — cross-color spreading away from a sharp chroma transition. Sample
  a 5–8 px wide vertical strip in the grey region immediately to the side of a
  tartan vertical edge. Baseline chroma should be `(U, V) ≈ (512, 512)`.
  - `DC_TARTAN_EDGE_LEFT`, `DC_TARTAN_EDGE_RIGHT` — flanking the tartan in (1,1).
- `cross_color` — chroma appearing inside black-and-white burst patterns where there
  should be none. Indicates the decoder is mistaking high-frequency luma for chroma.
  - `XC_BURST_300TVL`, `XC_BURST_400TVL` — inside the 300/400 tvl burst regions.
  - `XC_WEDGE_4MHz`, `XC_WEDGE_5MHz` — inside the 4/5 MHz wedge segments.
- `cross_luma` — luma modulation at SC frequency inside large uniform chroma areas.
  Indicates the decoder is mistaking the color subcarrier for luma detail.
  - `XL_RED_INTERIOR` — interior of the red 100% block.
  - `XL_MAGENTA_INTERIOR` — interior of the brightest magenta step.
- `zone_plate_chroma_leak` — chroma appearing inside the zone-plate reserved
  container. The zone plate is a black/white-only pattern (concentric arcs); any
  chroma there is decoder-introduced cross-color. This is a strong differentiator
  between notch/simple decoders (high leak) and comb decoders (low leak).
  - `ZP_CHROMA_LEAK` — one region covering cells (3,4)–(6,9) i.e. a 360×216 px box
    centered on chart center. The zone-plate pattern itself only fills a circle of
    radius 73 px (0.15 × picture height) inside that box; the rest is grey. We
    integrate chroma over the whole box — grey-background pixels have near-zero
    chroma and only dilute the mean rather than bias it.

## Measurement algorithms

### `tp_freq.measure(Y, U, V, affine) -> dict`

Returns:
```python
{
    "regions": {
        "BURST_3p58": {
            "frequency_MHz_expected": 3.58,
            "frequency_MHz_detected": 3.55,
            "modulation_pct":         91.4,        # 0..100, measured / ideal
            "modulation_db":          -0.78,
            "snr_db":                 28.3,
            "sample_box_capture":     (x, y, w, h),  # post-affine
        },
        ...
    },
    "summary": {
        "luma_response_curve":   [(freq_MHz, modulation_db), ...],
        "minus_3db_freq_MHz":    4.1,              # interpolated; null if cannot find
        "minus_6db_freq_MHz":    4.7,
    },
}
```

Per-region algorithm:
1. Project `ideal_box` through `affine` to get the capture-coords sample window.
2. Extract a single 1D Y-plane line cross-section in the burst's modulation direction
   (horizontal for `burst_vertical`/`wedge_segment`; rotated for `burst_diagonal`).
3. Subtract the line's mean to get an AC signal.
4. Run a 1D real FFT. Convert sample-axis frequency to MHz given the NTSC raster
   (`13.5 MHz` sample rate horizontally; per-line vertical bursts use NTSC field timing).
5. Find the peak FFT bin nearest the expected frequency; record its amplitude and the
   detected peak frequency.
6. `modulation_pct = 100 * peak_amplitude / (Y_max_ideal - Y_min_ideal)` where the
   ideal max/min are 100% white/black (`876` codes in 10-bit limited range).
7. `modulation_db = 20*log10(modulation_pct/100)`.
8. SNR = peak amplitude vs. RMS of non-peak FFT bins.

The summary `−3 dB` / `−6 dB` frequencies are interpolated from the per-region curve
by sorting regions by `frequency_MHz` and finding the highest frequency at which the
modulation is still above the threshold.

### `tp_artifacts.measure(Y, U, V, affine) -> dict`

Returns:
```python
{
    "regions": {
        "HD_RED_TOP": {
            "artifact_kind":           "hanging_dots",
            "metric_y10_pp":           38.4,   # peak-to-peak Y modulation, 10-bit codes
            "metric_y10_pp_normalized": 0.044, # relative to chart contrast
            "sample_box_capture":      (x, y, w, h),
        },
        "ZP_CHROMA_LEAK": {
            "artifact_kind":           "zone_plate_chroma_leak",
            "chroma_rms":              23.8,   # sqrt(mean((U-512)² + (V-512)²))
            "chroma_present":          True,   # chroma_rms > T_zp_threshold (15)
            "sample_box_capture":      (x, y, w, h),
        },
        ...
    },
    "summary": {
        "max_hanging_dots_y10_pp":  38.4,
        "max_dot_crawl_chroma_rms": 12.1,
        "max_cross_color_chroma_rms": 31.5,
        "max_cross_luma_y10_pp":    18.2,
        "zone_plate_chroma_present": True,
    },
}
```

Per-kind algorithm:
- `hanging_dots`: extract the 5-row Y strip, subtract row mean, compute peak-to-peak
  Y modulation along x. Normalize by (`WHITE_Y10 − BLACK_Y10`) = 876.
- `dot_crawl`: extract the chroma-side strip; compute RMS of `sqrt((U−512)² + (V−512)²)`.
  This region should be uniformly grey-chroma; non-zero RMS = cross-color leakage.
- `cross_color`: same `sqrt(ΔU² + ΔV²)` RMS over the entire burst box. The burst is
  black/white only — any chroma there is decoder cross-color.
- `cross_luma`: extract the Y plane over a flat-color block, subtract mean, bandpass
  around SC (3.58 MHz horizontally ≈ 1.8 sample period; bandpass 3.0–4.0 MHz),
  compute peak-to-peak amplitude.
- `zone_plate_chroma_leak`: compute `chroma_rms` over the whole region; set
  `chroma_present = chroma_rms > T_zp_threshold`.

All thresholds live in `tp_artifacts.THRESHOLDS` for easy tuning.

### `tp_classify.classify(freq, artifacts) -> dict`

Returns:
```python
{
    "decoder_class": "notch",                  # or line_comb / temporal_comb_or_adaptive / undetermined
    "confidence":    0.78,                     # 0..1
    "evidence": {
        "rule_fired":           "notch_rule_1",
        "contributing_metrics": {
            "max_cross_color_chroma_rms": 31.5,    # high
            "zone_plate_chroma_present":  True,    # high cross-color
            "WEDGE_5MHz_modulation_pct":  18.0,    # low chroma bandwidth
        },
    },
    "thresholds_used": {...},                  # snapshot of THRESHOLDS at classify time
}
```

Rule logic (matching the whitepaper's per-decoder screenshots):

1. **notch / simple low-pass** (whitepaper "Output from simple NTSC notch"):
   - `summary.max_cross_color_chroma_rms > T_xc_high` (e.g. 20)
   - AND `summary.zone_plate_chroma_present == True`
   - AND `WEDGE_5MHz.modulation_pct < T_chroma_bw_low` (e.g. 30%)
   - → `decoder_class = "notch"`.

2. **line comb** (whitepaper "Hanging dots ... vertical chrominance transition"):
   - `summary.max_cross_color_chroma_rms < T_xc_low` (e.g. 12)
   - AND `summary.max_hanging_dots_y10_pp > T_hd_high` (e.g. 25 codes)
   - → `decoder_class = "line_comb"`.

3. **temporal comb or adaptive** (whitepaper "Field comb decoded"):
   - `summary.max_cross_color_chroma_rms < T_xc_low`
   - AND `summary.max_hanging_dots_y10_pp < T_hd_low` (e.g. 12 codes)
   - AND `summary.zone_plate_chroma_present == False`
   - AND `summary.max_cross_luma_y10_pp < T_xl_low` (e.g. 15 codes)
   - → `decoder_class = "temporal_comb_or_adaptive"`.
     Subclassification (temporal vs adaptive) is deferred to Stage 3.5 (requires
     zone-plate motion analysis across frames).

4. **otherwise**: `decoder_class = "undetermined"` with `confidence < 0.5` and
   evidence listing the metrics that fell in mixed ranges.

Confidence formula: for each contributing metric `m` with rule threshold `T` and a
"hard-yes" value `T_hard` (where the rule definitely fires), compute
`m_conf = clip((m − T) / (T_hard − T), 0, 1)` (sign-flipped for "less than"
thresholds). The rule's confidence is the geometric mean of its `m_conf` values.
`T_hard` for each threshold lives next to it in `tp_classify.THRESHOLDS`. The
classifier emits `decoder_class = "undetermined"` whenever the highest-scoring
rule's confidence is below 0.5.

## Data flow

```
tp_measure.measure(capture_path, frame_index)
  ├── extract_frame → pad_to_486                        (existing)
  ├── tp_register.register_with_geometry → affine M     (existing)
  ├── sample_region (Stage 1: tartan, gray, ramp fit)   (existing)
  ├── tp_freq.measure(Y, U, V, M)                       (new)
  ├── tp_artifacts.measure(Y, U, V, M)                  (new)
  └── tp_classify.classify(freq_results, artifact_results) (new)
  → JSON with new top-level keys:
       frequency_response, artifacts, decoder_class
```

Stage 3 runs only when `_meta.registration.quality_flag` is `"ok"` or `"warn"`.
On `failed`, the three new keys are `null` and a top-level note explains why.

## HTML rendering

`tp_compare.py` gains three rendering functions, mirroring the existing
`render_tartan_deltas` / `render_gray_deltas` / `render_geometry_section` pattern:

- `render_frequency_response(captures)` — a per-clip line chart (freq vs dB)
  using Chart.js; below the chart, a small table of per-burst modulation and the
  −3 dB / −6 dB frequencies.
- `render_artifacts(captures)` — a table with rows per artifact-region id and
  columns per clip, color-coded by severity (green/yellow/red thresholds matching
  the classifier's `T_*` values).
- `render_decoder_class(captures)` — a badge per clip showing the discrete class,
  confidence, and the rule_fired/contributing_metrics evidence in a tooltip or
  expandable panel.

`cross_clip_report.py`:
- Overlay all clips' frequency curves on a single chart (one line per clip).
- Add an "Artifacts" table where each clip is a row and each artifact-summary
  metric is a column; color-coded by ranking within the set.
- Add a "Decoder class" column to the existing per-clip summary header table.

## Testing strategy

**Synthesizer additions (`tp_synthesize.py`)**
Draw the new chart features so the synthesized ideal frame exercises Stage 3:
- Vertical bursts at 3.58, 4.43, 4.286 MHz at their cell locations.
- Diagonal bursts at 300/400 tvl.
- MHz-wedge segments at 3/4/5 MHz in col 10.
- Saturated chroma blocks: red 100% (9,10–9,12), magenta steps (9,1–9,3).
- The synthesized chart deliberately omits the moving zone plate (since the real
  pattern is non-deterministic per frame). `ZP_CHROMA_LEAK` over the empty grey
  region of the synthesized chart should yield `chroma_rms ≈ 0`,
  `chroma_present == False`.

**Synthetic-baseline tests** (`test_cases/test_tp_freq.py`,
`test_cases/test_tp_artifacts.py`, `test_cases/test_tp_classify.py`)
- Synthesized ideal frame → every burst modulation ≥ 95%, every artifact metric
  near zero, decoder_class = `temporal_comb_or_adaptive` (the cleanest class).

**Degradation tests** (`tp_fixtures.py` degradation helpers)
- `inject_cross_color(Y, U, V, regions)` — paint chroma noise into the named burst
  regions. Verify `tp_artifacts` reports elevated `cross_color_chroma_rms` and
  `tp_classify` flips to `"notch"` when cross-color is widespread.
- `inject_hanging_dots(Y, regions)` — overlay a 3.58 MHz luma ripple in the strips
  above/below the red block. Verify `tp_artifacts` reports elevated
  `hanging_dots_y10_pp` and `tp_classify` flips to `"line_comb"`.
- `band_limit_luma(Y, cutoff_MHz)` — lowpass the Y plane. Verify the `WEDGE_*`
  modulation drops in the expected order.

**Classifier-rule unit tests**
Synthetic `freq` and `artifact` dicts hand-constructed to exercise each rule branch;
verify the right `decoder_class`, confidence, and `rule_fired` strings come out.

**Real-capture smoke tests**
Run on the existing snellhd / snellld captures across 5 frames; verify
`decoder_class` is consistent frame-to-frame (the decoder doesn't change mid-clip).
This test runs in a slow/optional bucket — not part of the default CI loop —
because it requires the `/wintmp/analog_video/tp_compare/` corpus.

## Calibration

`tp_calibrate.py` gains two presets:
- `stage3-bursts` — operator clicks the centers of the 8 burst regions.
- `stage3-artifacts` — operator clicks the centers of the artifact regions.

The output JSON updates the corresponding catalogs in `tp_chart.py` (or a side-car
override file, matching the pattern Stage 2 already uses).

## Implementation phasing

Phasing for the implementation plan — each phase is independently shippable and
testable:

1. **Phase 1 — Frequency response (smallest, highest immediate value).**
   - tp_chart `BURST_REGIONS` catalog.
   - tp_synthesize burst drawing.
   - tp_freq module + unit tests.
   - JSON wiring in tp_measure.
   - tp_compare frequency-response section + cross_clip overlay.

2. **Phase 2 — Artifact metrics.**
   - tp_chart `ARTIFACT_REGIONS` catalog (incl. `ZP_CHROMA_LEAK`).
   - tp_synthesize saturated-chroma-block additions.
   - tp_artifacts module + unit tests.
   - tp_fixtures degradation helpers.
   - JSON wiring in tp_measure.
   - tp_compare artifacts section + cross_clip table.

3. **Phase 3 — Decoder-class classifier.**
   - tp_classify module + rule unit tests.
   - Threshold calibration on real captures.
   - JSON wiring.
   - tp_compare decoder-class badge + cross_clip column.

4. **Phase 4 — Calibration presets.**
   - tp_calibrate `stage3-bursts` and `stage3-artifacts` presets.
   - Operator pass on real captures to lock in coordinates.

The plan that follows this spec will lay out each phase as a sequence of bite-sized
tasks per the writing-plans skill.

## Out of scope for Stage 3 (explicit, to prevent scope creep)

- Temporal/zone-plate motion analysis across frames (needed to distinguish
  `temporal_comb` from `adaptive`; requires cross-frame field comparison).
- PLUGE / interlace check / Y/C timing 1.0/0.5/1.5 MHz / radial wedge / pulse-and-bar
  measurements. These are mostly independent of the freq+artifact+classify trio and
  belong in a Stage 3.5 spec.
- Field-vs-frame subclassification within `temporal_comb_or_adaptive`.
- DVD-variant boundary-arrow detection (already noted in Stage 2; orthogonal).
- Chroma differential phase / differential gain (would need analog test signal,
  not visible from the SW2 chart alone in a usable form).
