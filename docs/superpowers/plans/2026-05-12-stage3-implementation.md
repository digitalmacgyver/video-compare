# Stage 3 Implementation Plan

> **For agentic workers:** Plan is executed inline in the same session that wrote it (user requested no approval gate). TDD per task: failing test → minimal impl → green → commit.

**Goal:** Add Stage 3 to the tp_compare pipeline: frequency-response measurement, decoder-artifact detection (incl. zone-plate chroma leak), and a rule-based decoder-class classifier. Output flows into per-clip JSON and the HTML reports.

**Architecture:** Three new modules (`tp_freq`, `tp_artifacts`, `tp_classify`) consuming chart-anchored sample regions from extended `tp_chart` catalogs, sampled via the existing Stage 2 affine matrix. `tp_measure` orchestrates; `tp_compare` + `cross_clip_report` render. Synthesizer and fixtures extended so synthetic frames exercise the pipeline end-to-end before real-capture validation.

**Tech Stack:** Python 3, numpy, scipy (FFT + ndimage rotate), opencv (existing for I/O), Chart.js 4.x (HTML).

**Spec:** `docs/superpowers/specs/2026-05-12-stage3-frequency-decoder-artifacts-design.md`.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `tp_chart.py` | modify | Add `BURST_REGIONS`, `ARTIFACT_REGIONS` catalogs |
| `tp_synthesize.py` | modify | Draw burst patterns, saturated chroma blocks |
| `tp_freq.py` | create | Per-burst FFT-based modulation measurement |
| `tp_artifacts.py` | create | Hanging-dots / dot-crawl / cross-color / cross-luma / zone-plate-chroma-leak metrics |
| `tp_classify.py` | create | Rule-based decoder classifier with confidence |
| `tp_fixtures.py` | modify | `inject_cross_color`, `inject_hanging_dots`, `band_limit_luma` |
| `tp_measure.py` | modify | Invoke Stage 3 after Stage 2; embed JSON keys |
| `tp_compare.py` | modify | Render frequency, artifacts, decoder-class sections |
| `cross_clip_report.py` | modify | Overlay freq curves; artifact table; decoder-class column |
| `test_cases/test_tp_chart.py` | modify | Catalog-shape tests for new regions |
| `test_cases/test_tp_synthesize.py` | modify | Burst-drawing + saturated-block tests |
| `test_cases/test_tp_freq.py` | create | Unit tests for tp_freq |
| `test_cases/test_tp_artifacts.py` | create | Unit tests for tp_artifacts |
| `test_cases/test_tp_classify.py` | create | Unit tests for tp_classify |
| `test_cases/test_tp_measure.py` | modify | End-to-end key presence + classifier output |

## Coordinate decisions (initial values; operator calibration can refine later)

**BURST_REGIONS** (chart cells are 60×54 px; intersections at (60c, 54r)):
- `BURST_3p58` — cell (2,2), 3.58 MHz vertical. Box `(70, 61, 40, 40)`.
- `BURST_4p43` — cell (2,11), 4.43 MHz vertical. Box `(610, 61, 40, 40)`.
- `BURST_4p286_SECAM` — cell (8,2), 4.286 MHz vertical. Box `(70, 385, 40, 40)`.
- `BURST_300TVL_DIAG` — cell (2,5), 300 tvl diagonal. Box `(250, 61, 40, 40)`, `stripe_angle_deg=45`.
- `BURST_400TVL_DIAG` — cell (2,8), 400 tvl diagonal. Box `(430, 61, 40, 40)`, `stripe_angle_deg=45`.
- `WEDGE_3MHz` — cell (4,10), 3.0 MHz. Box `(550, 169, 40, 40)`.
- `WEDGE_4MHz` — cell (5,10), 4.0 MHz. Box `(550, 223, 40, 40)`.
- `WEDGE_5MHz` — cell (6,10), 5.0 MHz. Box `(550, 277, 40, 40)`.

**ARTIFACT_REGIONS**:
- `HD_RED_TOP` — strip above red block in (9,10)-(9,12). Box `(550, 427, 160, 5)`. `artifact_kind="hanging_dots"`.
- `HD_MAGENTA_TOP` — strip above magenta steps in (9,1)-(9,3). Box `(30, 427, 130, 5)`.
- `DC_TARTAN_BELOW` — grey region below tartan in cell (2,1). Box `(10, 64, 44, 40)`. `artifact_kind="dot_crawl"`.
- `XC_BURST_300TVL` — chroma in the 300 tvl burst (luma-only on chart). Box `(250, 61, 40, 40)`. `artifact_kind="cross_color"`.
- `XC_BURST_400TVL` — same, 400 tvl. Box `(430, 61, 40, 40)`.
- `XC_WEDGE_4MHz` — chroma in 4 MHz wedge. Box `(550, 223, 40, 40)`.
- `XC_WEDGE_5MHz` — chroma in 5 MHz wedge. Box `(550, 277, 40, 40)`.
- `XL_RED_INTERIOR` — luma SC modulation inside red block. Box `(600, 445, 60, 30)`. `artifact_kind="cross_luma"`.
- `XL_MAGENTA_INTERIOR` — luma SC modulation inside 100% magenta step. Box `(130, 445, 40, 30)`.
- `ZP_CHROMA_LEAK` — full zone-plate container (3,4)-(6,9). Box `(180, 108, 360, 216)`. `artifact_kind="zone_plate_chroma_leak"`.

## Constants

- `tp_chart.NTSC_SAMPLE_RATE_MHZ = 13.5` (already implicit in chart geometry).
- `tp_artifacts.THRESHOLDS = {"T_zp_threshold": 15.0, "T_hd_high": 25.0, "T_hd_low": 12.0, "T_xc_high": 20.0, "T_xc_low": 12.0, "T_xl_low": 15.0, "T_chroma_bw_low_pct": 30.0}`.
- `tp_classify` reads `tp_artifacts.THRESHOLDS` and adds hard-yes values for confidence.

---

## Phase 1 — Frequency Response

### Task 1: BURST_REGIONS catalog

**Files:**
- Modify: `tp_chart.py` — append `BURST_REGIONS` constant
- Modify: `test_cases/test_tp_chart.py` — add catalog-shape tests

- [ ] **Step 1: Write failing tests** (append to `test_cases/test_tp_chart.py` before `TESTS = [`):

```python
def test_burst_regions_catalog():
    ids = [r["id"] for r in tp_chart.BURST_REGIONS]
    assert len(ids) == 8
    expected = {"BURST_3p58", "BURST_4p43", "BURST_4p286_SECAM",
                "BURST_300TVL_DIAG", "BURST_400TVL_DIAG",
                "WEDGE_3MHz", "WEDGE_4MHz", "WEDGE_5MHz"}
    assert set(ids) == expected
    for r in tp_chart.BURST_REGIONS:
        assert "frequency_MHz" in r
        assert r["frequency_MHz"] > 0
        assert r["kind"] in ("burst_vertical", "burst_diagonal", "wedge_segment")
        x, y, w, h = r["ideal_box"]
        assert 0 <= x and x + w <= 720
        assert 0 <= y and y + h <= 486
        if r["kind"] == "burst_diagonal":
            assert "stripe_angle_deg" in r
```

And add `test_burst_regions_catalog` to the `TESTS` list.

- [ ] **Step 2: Run, expect FAIL**: `python test_cases/test_tp_chart.py`

- [ ] **Step 3: Implement** — append to `tp_chart.py` after the existing catalogs:

```python
# =====================================================================
# BURST / WEDGE REGIONS (Stage 3 — frequency response)
# =====================================================================

NTSC_SAMPLE_RATE_MHZ = 13.5  # horizontal sample rate for NTSC SDI 720-wide

_BURST_RAW = [
    # id,                 kind,             freq_MHz, ideal_box,         extras
    ("BURST_3p58",        "burst_vertical", 3.58,     (70, 61, 40, 40),  {}),
    ("BURST_4p43",        "burst_vertical", 4.43,     (610, 61, 40, 40), {}),
    ("BURST_4p286_SECAM", "burst_vertical", 4.286,    (70, 385, 40, 40), {}),
    ("BURST_300TVL_DIAG", "burst_diagonal", 3.95,     (250, 61, 40, 40), {"stripe_angle_deg": 45}),
    ("BURST_400TVL_DIAG", "burst_diagonal", 5.27,     (430, 61, 40, 40), {"stripe_angle_deg": 45}),
    ("WEDGE_3MHz",        "wedge_segment",  3.0,      (550, 169, 40, 40), {}),
    ("WEDGE_4MHz",        "wedge_segment",  4.0,      (550, 223, 40, 40), {}),
    ("WEDGE_5MHz",        "wedge_segment",  5.0,      (550, 277, 40, 40), {}),
]

# 300 tvl ≈ 3.95 MHz, 400 tvl ≈ 5.27 MHz at NTSC 4.18 MHz/100 tvl horizontal.

BURST_REGIONS = [
    {
        "id": rid,
        "kind": kind,
        "frequency_MHz": freq,
        "ideal_box": box,
        "sample": {"kind": "center_window", "size_frac": 0.6},
        **extras,
    }
    for rid, kind, freq, box, extras in _BURST_RAW
]
```

- [ ] **Step 4: Run tests**: `python test_cases/test_tp_chart.py` — expect all pass.

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(tp_chart): add BURST_REGIONS catalog (Stage 3)"
```

### Task 2: Synthesizer burst-drawing helpers

**Files:**
- Modify: `tp_synthesize.py` — add `_draw_bursts`, call from `synthesize`
- Modify: `test_cases/test_tp_synthesize.py` — add modulation check

- [ ] **Step 1: Write failing test** (append before `TESTS = [`):

```python
def test_synthesized_bursts_have_expected_modulation():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    for r in tp_chart.BURST_REGIONS:
        x, y, w, h = r["ideal_box"]
        win = Y[y:y+h, x:x+w].astype(np.float32)
        amp = float(win.max() - win.min())
        # Synthesizer renders 100% contrast bursts (black<->white).
        assert amp > 600, f"{r['id']}: amp={amp:.0f}, want >600"
```

Add to `TESTS` list. Import `tp_chart` and `numpy as np` if not already.

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement burst drawing in `tp_synthesize.py`**:

```python
def _draw_bursts(Y: np.ndarray) -> None:
    """Render BURST_REGIONS as black/white stripes at the chart frequency.

    Horizontal sample rate is 13.5 MHz; period_px = 13.5 / freq_MHz.
    For burst_vertical and wedge_segment, stripes are vertical (x varies).
    For burst_diagonal, stripes are rotated by stripe_angle_deg.
    """
    for r in tp_chart.BURST_REGIONS:
        x, y, w, h = r["ideal_box"]
        period = tp_chart.NTSC_SAMPLE_RATE_MHZ / r["frequency_MHz"]
        kind = r["kind"]
        if kind in ("burst_vertical", "wedge_segment"):
            xs = np.arange(w, dtype=np.float32)
            phase = 2.0 * np.pi * xs / period
            row = np.where(np.sin(phase) >= 0, tp_chart.WHITE_Y10, tp_chart.BLACK_Y10).astype(np.uint16)
            Y[y:y+h, x:x+w] = row[None, :]
        elif kind == "burst_diagonal":
            theta = np.deg2rad(r["stripe_angle_deg"])
            xs = np.arange(w, dtype=np.float32)
            ys = np.arange(h, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            # Project (xx, yy) onto the stripe-normal direction.
            proj = xx * np.cos(theta) + yy * np.sin(theta)
            phase = 2.0 * np.pi * proj / period
            tile = np.where(np.sin(phase) >= 0, tp_chart.WHITE_Y10, tp_chart.BLACK_Y10).astype(np.uint16)
            Y[y:y+h, x:x+w] = tile
```

In `synthesize`, call `_draw_bursts(Y)` after `_draw_gray_strip`:

```python
    _draw_gray_strip(Y, U, V)
    _draw_bursts(Y)                      # NEW
    _draw_boundary_triangles(Y)
```

- [ ] **Step 4: Run tests**: `python test_cases/test_tp_synthesize.py` and `python test_cases/test_tp_chart.py`. Both should pass.

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(tp_synthesize): draw burst patterns for Stage 3"
```

### Task 3: tp_freq module

**Files:**
- Create: `tp_freq.py`
- Create: `test_cases/test_tp_freq.py`

- [ ] **Step 1: Write tests** in `test_cases/test_tp_freq.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_freq: per-burst frequency-response measurement."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart
import tp_synthesize
import tp_freq


def _identity_affine():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def test_freq_measure_returns_all_regions():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_freq.measure(Y, U, V, _identity_affine())
    assert set(out["regions"].keys()) == {r["id"] for r in tp_chart.BURST_REGIONS}


def test_freq_measure_synthesized_high_modulation():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_freq.measure(Y, U, V, _identity_affine())
    for rid, rdata in out["regions"].items():
        assert rdata["modulation_pct"] > 50.0, (
            f"{rid}: modulation_pct={rdata['modulation_pct']:.1f}"
        )


def test_freq_measure_band_limited_drops_high_freq():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Box-blur Y to attenuate high frequencies.
    import cv2
    Yf = cv2.boxFilter(Y.astype(np.float32), -1, (5, 1))
    Yblur = Yf.astype(np.uint16)
    base = tp_freq.measure(Y, U, V, _identity_affine())
    blur = tp_freq.measure(Yblur, U, V, _identity_affine())
    assert (blur["regions"]["WEDGE_5MHz"]["modulation_pct"]
            < base["regions"]["WEDGE_5MHz"]["modulation_pct"] - 10), (
        "expected band-limit to drop WEDGE_5MHz modulation"
    )


def test_freq_measure_summary_has_minus3db():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_freq.measure(Y, U, V, _identity_affine())
    assert "luma_response_curve" in out["summary"]
    assert "minus_3db_freq_MHz" in out["summary"]
    assert "minus_6db_freq_MHz" in out["summary"]


TESTS = [
    test_freq_measure_returns_all_regions,
    test_freq_measure_synthesized_high_modulation,
    test_freq_measure_band_limited_drops_high_freq,
    test_freq_measure_summary_has_minus3db,
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

- [ ] **Step 2: Run, expect FAIL** (`tp_freq` import error).

- [ ] **Step 3: Implement `tp_freq.py`**:

```python
"""Per-burst frequency-response measurement (Stage 3)."""
from __future__ import annotations
import math
import numpy as np
import tp_chart


def _apply_affine_pt(M, x, y):
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _sample_box_capture(M, ideal_box):
    """Project an ideal (x,y,w,h) box through M to capture coords. Returns
    integer (x, y, w, h) in the capture plane."""
    x, y, w, h = ideal_box
    cx, cy = _apply_affine_pt(M, x + w / 2.0, y + h / 2.0)
    # Assume near-identity scale (Stage 2 affine is ≈ identity for SDI).
    cx = int(round(cx))
    cy = int(round(cy))
    return (cx - w // 2, cy - h // 2, w, h)


def _crop(Y, box, h_lim, w_lim):
    x, y, w, h = box
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(w_lim, x + w); y1 = min(h_lim, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return Y[y0:y1, x0:x1].astype(np.float32)


def _line_modulation(line, freq_MHz, sample_rate_MHz):
    """1D FFT modulation amplitude at the expected frequency.

    Returns (peak_amplitude_codes, detected_freq_MHz, snr_db). Amplitude is
    peak-to-peak (twice the FFT magnitude for a sinusoid)."""
    line = line - line.mean()
    n = len(line)
    if n < 8:
        return 0.0, 0.0, 0.0
    spec = np.fft.rfft(line)
    mags = np.abs(spec) * (2.0 / n)  # scale: single-sided spectrum
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_MHz)
    # Find FFT bin within ±20% of the expected frequency; otherwise nearest.
    tol = max(0.5, 0.2 * freq_MHz)
    candidates = np.where(np.abs(freqs - freq_MHz) <= tol)[0]
    if len(candidates) == 0:
        idx = int(np.argmin(np.abs(freqs - freq_MHz)))
    else:
        idx = int(candidates[np.argmax(mags[candidates])])
    peak_amp_pp = float(mags[idx] * 2.0)
    detected = float(freqs[idx])
    # SNR: peak vs RMS of non-peak bins (excluding DC).
    non_peak = np.concatenate([mags[1:idx], mags[idx + 1:]])
    if len(non_peak) > 0 and non_peak.mean() > 0:
        noise_rms = float(np.sqrt((non_peak ** 2).mean()))
        snr_db = 20.0 * math.log10(max(peak_amp_pp, 1e-9) / max(noise_rms, 1e-9))
    else:
        snr_db = 0.0
    return peak_amp_pp, detected, snr_db


def _extract_line(win, kind, stripe_angle_deg=None):
    """Extract a 1D Y line cross-section in the modulation direction."""
    h, w = win.shape
    if kind in ("burst_vertical", "wedge_segment"):
        return win[h // 2, :]
    if kind == "burst_horizontal":
        return win[:, w // 2]
    if kind == "burst_diagonal":
        # Rotate so stripes become vertical, then sample the middle row.
        import scipy.ndimage as ndi
        rotated = ndi.rotate(win, -stripe_angle_deg, reshape=False, order=1)
        return rotated[rotated.shape[0] // 2, :]
    raise ValueError(f"unknown burst kind: {kind}")


def measure(Y, U, V, affine):
    h_lim, w_lim = Y.shape
    regions_out = {}
    contrast_full = float(tp_chart.WHITE_Y10 - tp_chart.BLACK_Y10)  # 876
    for r in tp_chart.BURST_REGIONS:
        box = _sample_box_capture(affine, r["ideal_box"])
        win = _crop(Y, box, h_lim, w_lim)
        if win is None or win.size == 0:
            regions_out[r["id"]] = {
                "frequency_MHz_expected": r["frequency_MHz"],
                "frequency_MHz_detected": None,
                "modulation_pct": 0.0,
                "modulation_db": float("-inf"),
                "snr_db": 0.0,
                "sample_box_capture": list(box),
            }
            continue
        line = _extract_line(win, r["kind"], r.get("stripe_angle_deg"))
        peak_pp, detected, snr_db = _line_modulation(
            line, r["frequency_MHz"], tp_chart.NTSC_SAMPLE_RATE_MHZ
        )
        mod_pct = 100.0 * peak_pp / contrast_full
        mod_db = 20.0 * math.log10(max(mod_pct, 1e-6) / 100.0)
        regions_out[r["id"]] = {
            "frequency_MHz_expected": r["frequency_MHz"],
            "frequency_MHz_detected": detected,
            "modulation_pct": float(mod_pct),
            "modulation_db": float(mod_db),
            "snr_db": float(snr_db),
            "sample_box_capture": list(box),
        }
    summary = _summarize(regions_out)
    return {"regions": regions_out, "summary": summary}


def _summarize(regions_out):
    curve = sorted(
        ((d["frequency_MHz_expected"], d["modulation_db"])
         for d in regions_out.values()),
        key=lambda p: p[0],
    )

    def _crossing(threshold_db):
        # Highest frequency where modulation_db is still >= threshold_db,
        # interpolating linearly between the last "above" and first "below".
        last_above = None
        for f, db in curve:
            if db >= threshold_db:
                last_above = (f, db)
            elif last_above is not None:
                f0, db0 = last_above
                if db0 == db:
                    return f
                ratio = (db0 - threshold_db) / (db0 - db)
                return float(f0 + ratio * (f - f0))
        return None

    return {
        "luma_response_curve": [list(p) for p in curve],
        "minus_3db_freq_MHz": _crossing(-3.0),
        "minus_6db_freq_MHz": _crossing(-6.0),
    }
```

- [ ] **Step 4: Run tests**: `python test_cases/test_tp_freq.py` — expect all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add tp_freq.py test_cases/test_tp_freq.py
git commit -m "feat(tp_freq): per-burst FFT modulation measurement"
```

### Task 4: tp_measure wiring + tp_compare frequency-response section

**Files:**
- Modify: `tp_measure.py` — call `tp_freq.measure`, add `frequency_response` key
- Modify: `tp_compare.py` — add `render_frequency_response`
- Modify: `cross_clip_report.py` — add freq overlay chart
- Modify: `test_cases/test_tp_measure.py` — assert new JSON key present

- [ ] **Step 1: Test** in `test_cases/test_tp_measure.py` (add inside the `test_measure_end_to_end_includes_geometry_block` body or as separate test):

```python
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
    assert "regions" in fr
    assert "summary" in fr
    assert "BURST_3p58" in fr["regions"]
```

Add to `TESTS_TMPDIR`.

- [ ] **Step 2: Wire `tp_measure.py`** — after the `_build_geometry_block` call (around line 380) add:

```python
    import tp_freq
    Y_for_stage3, _, _ = Y_p, U_p, V_p  # reuse padded planes
    try:
        if reg["affine_matrix"] is not None and reg["quality_flag"] != "failed":
            freq_block = tp_freq.measure(Y_p, U_p, V_p, reg["affine_matrix"])
        else:
            freq_block = None
    except Exception as e:
        freq_block = {"error": str(e)}
    result["frequency_response"] = freq_block
```

Note: locate the dict that becomes the JSON (let me re-check tp_measure.py to find the exact insertion point during implementation; the diff above is the conceptual addition).

- [ ] **Step 3: Add `render_frequency_response` to `tp_compare.py`** — render a Chart.js line chart per clip plus a per-region modulation table. Inserted into `render_page` in the appendix section.

```python
def render_frequency_response(captures):
    if not any(c.get("frequency_response") for c in captures):
        return ""
    rows = []
    rows.append('<section class="freq-response"><h2>Frequency response</h2>')
    rows.append('<table class="freq-table"><thead><tr><th>Clip</th><th>−3 dB (MHz)</th><th>−6 dB (MHz)</th></tr></thead><tbody>')
    for c in captures:
        fr = c.get("frequency_response") or {}
        s = fr.get("summary", {}) or {}
        m3 = s.get("minus_3db_freq_MHz")
        m6 = s.get("minus_6db_freq_MHz")
        rows.append(
            f'<tr><td>{_basename(c["_meta"]["capture_path"])}</td>'
            f'<td>{m3:.2f}</td><td>{m6:.2f}</td></tr>'
            if m3 is not None and m6 is not None
            else f'<tr><td>{_basename(c["_meta"]["capture_path"])}</td><td>—</td><td>—</td></tr>'
        )
    rows.append('</tbody></table>')
    # Chart.js overlay
    datasets = []
    for c in captures:
        fr = c.get("frequency_response") or {}
        curve = (fr.get("summary") or {}).get("luma_response_curve", [])
        if not curve:
            continue
        datasets.append({
            "label": _basename(c["_meta"]["capture_path"]),
            "data": [{"x": f, "y": d} for f, d in curve],
        })
    rows.append('<canvas id="freqChart" height="160"></canvas>')
    rows.append('<script>')
    rows.append(f'const FREQ_DATA = {datasets!r};')
    rows.append("""
    new Chart(document.getElementById('freqChart').getContext('2d'), {
        type: 'line',
        data: {datasets: FREQ_DATA.map(d => ({...d, fill: false, tension: 0.2, parsing: false}))},
        options: {
            scales: {x: {type: 'linear', title: {display: true, text: 'Frequency (MHz)'}},
                     y: {title: {display: true, text: 'Modulation (dB)'}}},
            plugins: {legend: {position: 'top'}}
        }
    });""")
    rows.append('</script></section>')
    return "\n".join(rows)
```

In `render_page`, insert the call in the appendix region after `render_geometry_section`:

```python
        render_geometry_section(captures),
        render_frequency_response(captures),
```

- [ ] **Step 4: Mirror in `cross_clip_report.py`** — copy the same `render_frequency_response` logic into a function there (or import from tp_compare if HTML scaffolding allows; otherwise duplicate). Embed into the cross-clip page.

- [ ] **Step 5: Tests + commit**: `python test_cases/test_tp_measure.py` (all pass), `python test_cases/test_tp_synthesize.py`, `python test_cases/test_tp_chart.py`.

```bash
git add tp_measure.py tp_compare.py cross_clip_report.py test_cases/test_tp_measure.py
git commit -m "feat(tp_measure,tp_compare): wire Stage 3 frequency-response into JSON + HTML"
```

---

## Phase 2 — Artifacts

### Task 5: ARTIFACT_REGIONS catalog

**Files:** `tp_chart.py`, `test_cases/test_tp_chart.py`

- [ ] **Step 1: Test** (append):

```python
def test_artifact_regions_catalog():
    ids = [r["id"] for r in tp_chart.ARTIFACT_REGIONS]
    expected = {"HD_RED_TOP", "HD_MAGENTA_TOP", "DC_TARTAN_BELOW",
                "XC_BURST_300TVL", "XC_BURST_400TVL", "XC_WEDGE_4MHz",
                "XC_WEDGE_5MHz", "XL_RED_INTERIOR", "XL_MAGENTA_INTERIOR",
                "ZP_CHROMA_LEAK"}
    assert set(ids) == expected
    valid_kinds = {"hanging_dots", "dot_crawl", "cross_color",
                   "cross_luma", "zone_plate_chroma_leak"}
    for r in tp_chart.ARTIFACT_REGIONS:
        assert r["artifact_kind"] in valid_kinds
        x, y, w, h = r["ideal_box"]
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= 720
        assert 0 <= y and y + h <= 486
```

Add to `TESTS`.

- [ ] **Step 2: FAIL.**

- [ ] **Step 3: Implement** — append to `tp_chart.py`:

```python
# =====================================================================
# ARTIFACT REGIONS (Stage 3 — decoder-artifact detection)
# =====================================================================

_ARTIFACT_RAW = [
    ("HD_RED_TOP",        "hanging_dots",          (550, 427, 160, 5)),
    ("HD_MAGENTA_TOP",    "hanging_dots",          (30,  427, 130, 5)),
    ("DC_TARTAN_BELOW",   "dot_crawl",             (10,  64,  44,  40)),
    ("XC_BURST_300TVL",   "cross_color",           (250, 61,  40,  40)),
    ("XC_BURST_400TVL",   "cross_color",           (430, 61,  40,  40)),
    ("XC_WEDGE_4MHz",     "cross_color",           (550, 223, 40,  40)),
    ("XC_WEDGE_5MHz",     "cross_color",           (550, 277, 40,  40)),
    ("XL_RED_INTERIOR",   "cross_luma",            (600, 445, 60,  30)),
    ("XL_MAGENTA_INTERIOR", "cross_luma",          (130, 445, 40,  30)),
    ("ZP_CHROMA_LEAK",    "zone_plate_chroma_leak", (180, 108, 360, 216)),
]

ARTIFACT_REGIONS = [
    {
        "id": rid,
        "artifact_kind": kind,
        "ideal_box": box,
        "sample": {"kind": "center_window", "size_frac": 1.0},
    }
    for rid, kind, box in _ARTIFACT_RAW
]
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add tp_chart.py test_cases/test_tp_chart.py
git commit -m "feat(tp_chart): add ARTIFACT_REGIONS catalog (Stage 3)"
```

### Task 6: Synthesizer — saturated chroma blocks for artifact sampling

**Files:** `tp_synthesize.py`, `test_cases/test_tp_synthesize.py`

- [ ] **Step 1: Test** (append):

```python
def test_synthesized_red_block_is_saturated():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Red block roughly at (540, 432, 180, 54). Sample interior.
    win_y = Y[445:475, 600:660].astype(np.float32).mean()
    # 100% red in BT.601 limited range: Y10 ~ 64 + 0.299*876 ≈ 326.
    assert 250 < win_y < 400, f"red Y mean={win_y:.0f}, want ~326"
    # Chroma at chart center for 100% red: V high, U low.
    win_v = V[445:475, 300:330].astype(np.float32).mean()  # V is chroma-sampled at half-x
    assert win_v > 700, f"red V mean={win_v:.0f}, want >700"


def test_synthesized_magenta_steps():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Three magenta levels at increasing brightness in cells (9,1)-(9,3).
    # Sample y centers in 445..475 (row 9), x in (10..50), (70..110), (130..170).
    ys = [Y[450:470, x0:x0+30].astype(np.float32).mean() for x0 in (15, 75, 135)]
    # Should be monotonic increasing (33/66/100% magenta).
    assert ys[0] < ys[1] < ys[2], f"magenta steps not monotonic: {ys}"
```

- [ ] **Step 2: FAIL.**

- [ ] **Step 3: Implement** in `tp_synthesize.py`:

```python
def _draw_saturated_chroma_blocks(Y, U, V):
    """Render the row-9 saturated chroma blocks for artifact sampling.

    - Cells (9,10)-(9,12): 100% red block.
    - Cells (9,1)-(9,3): three magenta steps at 33/66/100%.
    """
    # Red 100% block — bounds (540, 432, 180, 54).
    y10, u10, v10 = tp_chart.rgb_norm_to_yuv10(1.0, 0.0, 0.0)
    _fill_box_yuv422(Y, U, V, (540, 432, 180, 54), y10, u10, v10)
    # Magenta steps: cells (9,1), (9,2), (9,3), each 60x54.
    for i, frac in enumerate([1/3, 2/3, 1.0]):
        y10, u10, v10 = tp_chart.rgb_norm_to_yuv10(frac, 0.0, frac)
        _fill_box_yuv422(Y, U, V, (i * 60, 432, 60, 54), y10, u10, v10)
```

In `synthesize`, call after `_draw_bursts`:

```python
    _draw_bursts(Y)
    _draw_saturated_chroma_blocks(Y, U, V)   # NEW
    _draw_boundary_triangles(Y)
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add tp_synthesize.py test_cases/test_tp_synthesize.py
git commit -m "feat(tp_synthesize): add row-9 saturated chroma blocks (Stage 3)"
```

### Task 7: tp_artifacts module

**Files:** create `tp_artifacts.py`, `test_cases/test_tp_artifacts.py`

- [ ] **Step 1: Test** in `test_cases/test_tp_artifacts.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_artifacts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tp_chart, tp_synthesize, tp_artifacts


def _identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def test_artifacts_measure_returns_all_regions():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    assert set(out["regions"].keys()) == {r["id"] for r in tp_chart.ARTIFACT_REGIONS}


def test_zone_plate_chroma_leak_synthesized_is_low():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    zp = out["regions"]["ZP_CHROMA_LEAK"]
    assert zp["chroma_rms"] < 5.0, f"zp.chroma_rms={zp['chroma_rms']:.2f}"
    assert zp["chroma_present"] is False


def test_zone_plate_chroma_leak_detects_injected_chroma():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    # Inject chroma into the zone-plate region.
    U[120:170, 100:200] = 600   # U2 chroma plane: half-width coords
    V[120:170, 100:200] = 600
    out = tp_artifacts.measure(Y, U, V, _identity())
    zp = out["regions"]["ZP_CHROMA_LEAK"]
    assert zp["chroma_rms"] > 15.0
    assert zp["chroma_present"] is True


def test_cross_color_metric_zero_on_synthesized_bursts():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    for rid in ("XC_BURST_300TVL", "XC_BURST_400TVL", "XC_WEDGE_4MHz", "XC_WEDGE_5MHz"):
        r = out["regions"][rid]
        assert r["chroma_rms"] < 3.0, f"{rid} chroma_rms={r['chroma_rms']:.2f}"


def test_summary_keys_present():
    Y, U, V = tp_synthesize.synthesize(720, 486)
    out = tp_artifacts.measure(Y, U, V, _identity())
    s = out["summary"]
    for key in ("max_hanging_dots_y10_pp", "max_dot_crawl_chroma_rms",
                "max_cross_color_chroma_rms", "max_cross_luma_y10_pp",
                "zone_plate_chroma_present"):
        assert key in s, f"missing summary key: {key}"


TESTS = [
    test_artifacts_measure_returns_all_regions,
    test_zone_plate_chroma_leak_synthesized_is_low,
    test_zone_plate_chroma_leak_detects_injected_chroma,
    test_cross_color_metric_zero_on_synthesized_bursts,
    test_summary_keys_present,
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

- [ ] **Step 2: FAIL.**

- [ ] **Step 3: Implement `tp_artifacts.py`**:

```python
"""Decoder-artifact metrics (Stage 3): hanging dots, dot crawl, cross-color,
cross-luma, zone-plate chroma leak."""
from __future__ import annotations
import math
import numpy as np
import tp_chart


THRESHOLDS = {
    "T_zp_threshold":       15.0,   # zone-plate chroma_rms threshold
    "T_hd_high":            25.0,   # hanging dots peak-to-peak Y10
    "T_hd_low":             12.0,
    "T_xc_high":            20.0,   # cross-color chroma_rms threshold
    "T_xc_low":             12.0,
    "T_xl_low":             15.0,   # cross-luma peak-to-peak Y10
    "T_chroma_bw_low_pct":  30.0,   # modulation pct at WEDGE_4/5MHz
}


def _apply_affine_pt(M, x, y):
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
            float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _project_box(M, box):
    x, y, w, h = box
    cx, cy = _apply_affine_pt(M, x + w / 2.0, y + h / 2.0)
    return (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), w, h)


def _crop_yuv(Y, U, V, box):
    """Crop Y at integer coords and U/V at half-x. Returns (y_win, u_win, v_win)."""
    x, y, w, h = box
    h_lim, w_lim = Y.shape
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(w_lim, x + w); y1 = min(h_lim, y + h)
    if x1 <= x0 or y1 <= y0:
        return None, None, None
    yw = Y[y0:y1, x0:x1].astype(np.float32)
    # U/V are 4:2:2 — half horizontal resolution. Slice with integer-floor.
    ux0 = x0 // 2; ux1 = (x1 + 1) // 2
    uw = U[y0:y1, ux0:ux1].astype(np.float32)
    vw = V[y0:y1, ux0:ux1].astype(np.float32)
    return yw, uw, vw


def _chroma_rms(u_win, v_win):
    du = u_win - tp_chart.CHROMA_CENTER
    dv = v_win - tp_chart.CHROMA_CENTER
    return float(np.sqrt((du * du + dv * dv).mean()))


def _hanging_dots(y_win):
    if y_win is None or y_win.size == 0:
        return 0.0
    # Subtract row means, then compute peak-to-peak along x across rows.
    row_demean = y_win - y_win.mean(axis=1, keepdims=True)
    pp = float(row_demean.max() - row_demean.min())
    return pp


def _cross_luma_sc_amp(y_win, sample_rate_MHz):
    """Bandpass Y around 3.0..4.0 MHz and report peak-to-peak amplitude."""
    if y_win is None or y_win.size == 0:
        return 0.0
    line = y_win[y_win.shape[0] // 2, :] - y_win.mean()
    n = len(line)
    if n < 8:
        return 0.0
    spec = np.fft.rfft(line)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_MHz)
    mask = (freqs >= 3.0) & (freqs <= 4.0)
    if not mask.any():
        return 0.0
    band = np.zeros_like(spec)
    band[mask] = spec[mask]
    filt = np.fft.irfft(band, n)
    return float(filt.max() - filt.min())


def measure(Y, U, V, affine):
    regions_out = {}
    sample_rate = tp_chart.NTSC_SAMPLE_RATE_MHZ
    for r in tp_chart.ARTIFACT_REGIONS:
        kind = r["artifact_kind"]
        box = _project_box(affine, r["ideal_box"])
        yw, uw, vw = _crop_yuv(Y, U, V, box)
        entry = {"artifact_kind": kind, "sample_box_capture": list(box)}
        if yw is None:
            entry["error"] = "out_of_frame"
        elif kind == "hanging_dots":
            entry["metric_y10_pp"] = _hanging_dots(yw)
            entry["metric_y10_pp_normalized"] = entry["metric_y10_pp"] / 876.0
        elif kind == "dot_crawl":
            entry["chroma_rms"] = _chroma_rms(uw, vw)
        elif kind == "cross_color":
            entry["chroma_rms"] = _chroma_rms(uw, vw)
        elif kind == "cross_luma":
            entry["metric_y10_pp"] = _cross_luma_sc_amp(yw, sample_rate)
        elif kind == "zone_plate_chroma_leak":
            crms = _chroma_rms(uw, vw)
            entry["chroma_rms"] = crms
            entry["chroma_present"] = crms > THRESHOLDS["T_zp_threshold"]
        regions_out[r["id"]] = entry
    summary = _summarize(regions_out)
    return {"regions": regions_out, "summary": summary, "thresholds": dict(THRESHOLDS)}


def _summarize(regions_out):
    def _max_metric(kind, key):
        vals = [r.get(key, 0.0) for r in regions_out.values()
                if r.get("artifact_kind") == kind and key in r]
        return float(max(vals)) if vals else 0.0

    return {
        "max_hanging_dots_y10_pp":    _max_metric("hanging_dots", "metric_y10_pp"),
        "max_dot_crawl_chroma_rms":   _max_metric("dot_crawl", "chroma_rms"),
        "max_cross_color_chroma_rms": _max_metric("cross_color", "chroma_rms"),
        "max_cross_luma_y10_pp":      _max_metric("cross_luma", "metric_y10_pp"),
        "zone_plate_chroma_present":  bool(
            regions_out.get("ZP_CHROMA_LEAK", {}).get("chroma_present", False)
        ),
        "zone_plate_chroma_rms":      regions_out.get("ZP_CHROMA_LEAK", {}).get("chroma_rms", 0.0),
    }
```

- [ ] **Step 4: Run** `python test_cases/test_tp_artifacts.py` — expect all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add tp_artifacts.py test_cases/test_tp_artifacts.py
git commit -m "feat(tp_artifacts): hanging dots / dot crawl / cross-color / cross-luma / zone-plate leak"
```

### Task 8: tp_fixtures degradation helpers

**Files:** `tp_fixtures.py`, `test_cases/test_tp_fixtures.py`

- [ ] **Step 1: Test** (append to `test_tp_fixtures.py`):

```python
def test_inject_hanging_dots_raises_hd_metric():
    import tp_artifacts
    Y, U, V = tp_synthesize.synthesize(720, 486)
    base = tp_artifacts.measure(
        Y, U, V, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    )
    Y2 = Y.copy()
    tp_fixtures.inject_hanging_dots(Y2, region_id="HD_RED_TOP", amplitude_y10=80)
    bad = tp_artifacts.measure(
        Y2, U, V, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    )
    assert bad["regions"]["HD_RED_TOP"]["metric_y10_pp"] > (
        base["regions"]["HD_RED_TOP"]["metric_y10_pp"] + 40
    )


def test_inject_cross_color_raises_xc_metric():
    import tp_artifacts
    Y, U, V = tp_synthesize.synthesize(720, 486)
    U2 = U.copy(); V2 = V.copy()
    tp_fixtures.inject_cross_color(U2, V2, region_id="XC_WEDGE_5MHz", chroma_amp=60)
    bad = tp_artifacts.measure(
        Y, U2, V2, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    )
    assert bad["regions"]["XC_WEDGE_5MHz"]["chroma_rms"] > 30
```

- [ ] **Step 2: FAIL.**

- [ ] **Step 3: Implement** — append to `tp_fixtures.py`:

```python
import tp_chart


def _region_by_id(regions, region_id):
    for r in regions:
        if r["id"] == region_id:
            return r
    raise KeyError(region_id)


def inject_hanging_dots(Y, region_id, amplitude_y10=80):
    """Overlay a 3.58 MHz Y modulation into an ARTIFACT_REGIONS box."""
    r = _region_by_id(tp_chart.ARTIFACT_REGIONS, region_id)
    x, y, w, h = r["ideal_box"]
    period = tp_chart.NTSC_SAMPLE_RATE_MHZ / 3.58
    xs = np.arange(w, dtype=np.float32)
    pattern = (amplitude_y10 * np.sin(2 * np.pi * xs / period)).astype(np.int32)
    region = Y[y:y+h, x:x+w].astype(np.int32) + pattern[None, :]
    Y[y:y+h, x:x+w] = np.clip(region, 0, 1023).astype(np.uint16)


def inject_cross_color(U, V, region_id, chroma_amp=60):
    """Add chroma offset into the U/V planes for a region (4:2:2 half-x)."""
    r = _region_by_id(tp_chart.ARTIFACT_REGIONS, region_id)
    x, y, w, h = r["ideal_box"]
    ux0 = x // 2; ux1 = (x + w) // 2
    U[y:y+h, ux0:ux1] = np.clip(
        U[y:y+h, ux0:ux1].astype(np.int32) + chroma_amp, 0, 1023
    ).astype(np.uint16)
    V[y:y+h, ux0:ux1] = np.clip(
        V[y:y+h, ux0:ux1].astype(np.int32) + chroma_amp, 0, 1023
    ).astype(np.uint16)
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add tp_fixtures.py test_cases/test_tp_fixtures.py
git commit -m "feat(tp_fixtures): inject_hanging_dots, inject_cross_color helpers"
```

### Task 9: Wire artifacts into measure + compare

**Files:** `tp_measure.py`, `tp_compare.py`, `cross_clip_report.py`, `test_cases/test_tp_measure.py`

- [ ] **Step 1: Test** in `test_tp_measure.py`:

```python
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
    assert "regions" in art
    assert "summary" in art
    assert "ZP_CHROMA_LEAK" in art["regions"]
    assert art["summary"]["zone_plate_chroma_present"] is False
```

Add to `TESTS_TMPDIR`.

- [ ] **Step 2: Wire `tp_measure.py`** — after the frequency_response insertion:

```python
    import tp_artifacts
    try:
        if reg["affine_matrix"] is not None and reg["quality_flag"] != "failed":
            artifacts_block = tp_artifacts.measure(Y_p, U_p, V_p, reg["affine_matrix"])
        else:
            artifacts_block = None
    except Exception as e:
        artifacts_block = {"error": str(e)}
    result["artifacts"] = artifacts_block
```

- [ ] **Step 3: Render** in `tp_compare.py`:

```python
def render_artifacts(captures):
    if not any(c.get("artifacts") for c in captures):
        return ""
    rows = ['<section class="artifacts"><h2>Decoder artifacts</h2>']
    rows.append('<table class="artifact-table"><thead><tr><th>Clip</th>'
                '<th>Hanging dots (pp)</th><th>Dot crawl (rms)</th>'
                '<th>Cross-color (rms)</th><th>Cross-luma (pp)</th>'
                '<th>Zone-plate chroma</th></tr></thead><tbody>')
    for c in captures:
        a = c.get("artifacts") or {}
        s = (a.get("summary") or {})
        zp_present = s.get("zone_plate_chroma_present")
        zp_rms = s.get("zone_plate_chroma_rms", 0)
        rows.append(
            f'<tr><td>{_basename(c["_meta"]["capture_path"])}</td>'
            f'<td>{s.get("max_hanging_dots_y10_pp", 0):.1f}</td>'
            f'<td>{s.get("max_dot_crawl_chroma_rms", 0):.2f}</td>'
            f'<td>{s.get("max_cross_color_chroma_rms", 0):.2f}</td>'
            f'<td>{s.get("max_cross_luma_y10_pp", 0):.1f}</td>'
            f'<td>{"YES" if zp_present else "no"} ({zp_rms:.1f})</td></tr>'
        )
    rows.append('</tbody></table></section>')
    return "\n".join(rows)
```

Call in `render_page` after `render_frequency_response`.

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add tp_measure.py tp_compare.py cross_clip_report.py test_cases/test_tp_measure.py
git commit -m "feat(tp_measure,tp_compare): wire Stage 3 artifacts into JSON + HTML"
```

---

## Phase 3 — Classifier

### Task 10: tp_classify module

**Files:** create `tp_classify.py`, `test_cases/test_tp_classify.py`

- [ ] **Step 1: Test** in `test_cases/test_tp_classify.py`:

```python
#!/usr/bin/env python3
"""Tests for tp_classify."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tp_classify


def _clean_freq():
    return {
        "regions": {
            "WEDGE_3MHz": {"frequency_MHz_expected": 3.0, "modulation_pct": 92.0, "modulation_db": -0.7},
            "WEDGE_4MHz": {"frequency_MHz_expected": 4.0, "modulation_pct": 88.0, "modulation_db": -1.1},
            "WEDGE_5MHz": {"frequency_MHz_expected": 5.0, "modulation_pct": 65.0, "modulation_db": -3.7},
        },
        "summary": {"luma_response_curve": [], "minus_3db_freq_MHz": 4.5, "minus_6db_freq_MHz": 5.2},
    }


def _clean_art():
    return {
        "regions": {},
        "summary": {
            "max_hanging_dots_y10_pp":    4.0,
            "max_dot_crawl_chroma_rms":   1.5,
            "max_cross_color_chroma_rms": 3.0,
            "max_cross_luma_y10_pp":      6.0,
            "zone_plate_chroma_present":  False,
            "zone_plate_chroma_rms":      3.1,
        },
    }


def test_clean_signal_classifies_as_temporal_or_adaptive():
    out = tp_classify.classify(_clean_freq(), _clean_art())
    assert out["decoder_class"] == "temporal_comb_or_adaptive"
    assert out["confidence"] >= 0.5


def test_high_xc_and_zp_classifies_as_notch():
    freq = _clean_freq()
    freq["regions"]["WEDGE_5MHz"]["modulation_pct"] = 18.0  # band-limited
    art = _clean_art()
    art["summary"]["max_cross_color_chroma_rms"] = 35.0
    art["summary"]["zone_plate_chroma_present"] = True
    art["summary"]["zone_plate_chroma_rms"] = 32.0
    out = tp_classify.classify(freq, art)
    assert out["decoder_class"] == "notch"


def test_high_hd_classifies_as_line_comb():
    art = _clean_art()
    art["summary"]["max_hanging_dots_y10_pp"] = 38.0
    out = tp_classify.classify(_clean_freq(), art)
    assert out["decoder_class"] == "line_comb"


def test_ambiguous_returns_undetermined():
    freq = _clean_freq()
    art = _clean_art()
    art["summary"]["max_cross_color_chroma_rms"] = 16.0  # in the middle
    art["summary"]["max_hanging_dots_y10_pp"] = 18.0     # in the middle
    out = tp_classify.classify(freq, art)
    assert out["decoder_class"] == "undetermined"


TESTS = [
    test_clean_signal_classifies_as_temporal_or_adaptive,
    test_high_xc_and_zp_classifies_as_notch,
    test_high_hd_classifies_as_line_comb,
    test_ambiguous_returns_undetermined,
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

- [ ] **Step 2: FAIL.**

- [ ] **Step 3: Implement `tp_classify.py`**:

```python
"""Decoder-class classifier (Stage 3): rule-based with confidence."""
from __future__ import annotations
import math
import tp_artifacts


def _ge_conf(value, threshold, hard_yes):
    """Confidence for "value >= threshold" rule. 0 at threshold, 1 at hard_yes."""
    if hard_yes <= threshold:
        return 1.0 if value >= threshold else 0.0
    if value <= threshold:
        return 0.0
    if value >= hard_yes:
        return 1.0
    return (value - threshold) / (hard_yes - threshold)


def _le_conf(value, threshold, hard_yes):
    """Confidence for "value <= threshold" rule. 0 at threshold, 1 at hard_yes."""
    if hard_yes >= threshold:
        return 1.0 if value <= threshold else 0.0
    if value >= threshold:
        return 0.0
    if value <= hard_yes:
        return 1.0
    return (threshold - value) / (threshold - hard_yes)


def _geom_mean(values):
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def classify(freq, artifacts):
    T = tp_artifacts.THRESHOLDS
    s = artifacts.get("summary", {}) or {}
    xc = s.get("max_cross_color_chroma_rms", 0.0)
    hd = s.get("max_hanging_dots_y10_pp", 0.0)
    xl = s.get("max_cross_luma_y10_pp", 0.0)
    zp_present = s.get("zone_plate_chroma_present", False)
    zp_rms = s.get("zone_plate_chroma_rms", 0.0)
    w5 = (freq.get("regions", {}) or {}).get("WEDGE_5MHz", {})
    w5_mod = w5.get("modulation_pct", 100.0)

    # Notch rule
    c_xc_high = _ge_conf(xc, T["T_xc_high"], T["T_xc_high"] * 2.0)
    c_w5_low = _le_conf(w5_mod, T["T_chroma_bw_low_pct"], T["T_chroma_bw_low_pct"] * 0.5)
    c_zp_present = 1.0 if zp_present else 0.0
    notch_conf = _geom_mean([c_xc_high, c_w5_low, c_zp_present])

    # Line-comb rule
    c_xc_low = _le_conf(xc, T["T_xc_low"], T["T_xc_low"] * 0.5)
    c_hd_high = _ge_conf(hd, T["T_hd_high"], T["T_hd_high"] * 2.0)
    line_comb_conf = _geom_mean([c_xc_low, c_hd_high])

    # Temporal/adaptive rule
    c_hd_low = _le_conf(hd, T["T_hd_low"], T["T_hd_low"] * 0.5)
    c_xl_low = _le_conf(xl, T["T_xl_low"], T["T_xl_low"] * 0.5)
    c_zp_absent = 1.0 if not zp_present else 0.0
    tca_conf = _geom_mean([c_xc_low, c_hd_low, c_xl_low, c_zp_absent])

    candidates = [
        ("notch", notch_conf, {
            "max_cross_color_chroma_rms": xc,
            "WEDGE_5MHz_modulation_pct": w5_mod,
            "zone_plate_chroma_present": zp_present,
            "zone_plate_chroma_rms": zp_rms,
        }),
        ("line_comb", line_comb_conf, {
            "max_cross_color_chroma_rms": xc,
            "max_hanging_dots_y10_pp": hd,
        }),
        ("temporal_comb_or_adaptive", tca_conf, {
            "max_cross_color_chroma_rms": xc,
            "max_hanging_dots_y10_pp": hd,
            "max_cross_luma_y10_pp": xl,
            "zone_plate_chroma_present": zp_present,
        }),
    ]
    candidates.sort(key=lambda c: c[1], reverse=True)
    best_class, best_conf, best_metrics = candidates[0]
    if best_conf < 0.5:
        return {
            "decoder_class":  "undetermined",
            "confidence":     float(best_conf),
            "evidence": {
                "rule_fired":           None,
                "candidate_confidences": {c[0]: float(c[1]) for c in candidates},
                "contributing_metrics": best_metrics,
            },
            "thresholds_used": dict(T),
        }
    return {
        "decoder_class":   best_class,
        "confidence":      float(best_conf),
        "evidence": {
            "rule_fired":           f"{best_class}_rule",
            "candidate_confidences": {c[0]: float(c[1]) for c in candidates},
            "contributing_metrics": best_metrics,
        },
        "thresholds_used": dict(T),
    }
```

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add tp_classify.py test_cases/test_tp_classify.py
git commit -m "feat(tp_classify): rule-based decoder-class classifier"
```

### Task 11: Wire classifier into measure + compare + cross_clip

**Files:** `tp_measure.py`, `tp_compare.py`, `cross_clip_report.py`, `test_cases/test_tp_measure.py`

- [ ] **Step 1: Test**:

```python
def test_measure_end_to_end_includes_decoder_class(tmp_dir):
    capture_path = os.path.join(tmp_dir, "ideal.mov")
    json_path = os.path.join(tmp_dir, "out_cls.json")
    _write_synthesized_prores(capture_path, 720, 486, frames=5)
    subprocess.run(
        ["python", "tp_measure.py", capture_path,
         "--frame", "0", "--output", json_path],
        check=True, cwd=PROJECT_ROOT,
    )
    with open(json_path) as f:
        data = json.load(f)
    assert "decoder_class" in data
    dc = data["decoder_class"]
    assert dc["decoder_class"] in (
        "notch", "line_comb", "temporal_comb_or_adaptive", "undetermined"
    )
```

- [ ] **Step 2: Wire** in `tp_measure.py`:

```python
    import tp_classify
    try:
        if freq_block and artifacts_block and "regions" in (freq_block or {}) and "regions" in (artifacts_block or {}):
            classify_block = tp_classify.classify(freq_block, artifacts_block)
        else:
            classify_block = None
    except Exception as e:
        classify_block = {"error": str(e)}
    result["decoder_class"] = classify_block
```

- [ ] **Step 3: Render** in `tp_compare.py`:

```python
def render_decoder_class(captures):
    if not any(c.get("decoder_class") for c in captures):
        return ""
    rows = ['<section class="decoder-class"><h2>Decoder class</h2>']
    rows.append('<table class="decoder-table"><thead><tr><th>Clip</th>'
                '<th>Class</th><th>Confidence</th><th>Evidence</th></tr></thead><tbody>')
    for c in captures:
        dc = c.get("decoder_class") or {}
        cls = dc.get("decoder_class", "—")
        conf = dc.get("confidence", 0)
        ev = dc.get("evidence", {})
        cands = ev.get("candidate_confidences", {}) or {}
        cands_str = ", ".join(f"{k}: {v:.2f}" for k, v in cands.items())
        rows.append(
            f'<tr><td>{_basename(c["_meta"]["capture_path"])}</td>'
            f'<td><strong>{cls}</strong></td><td>{conf:.2f}</td>'
            f'<td>{cands_str}</td></tr>'
        )
    rows.append('</tbody></table></section>')
    return "\n".join(rows)
```

Call in `render_page` after `render_artifacts`.

In `cross_clip_report.py`, add a `Decoder` column to the existing summary table.

- [ ] **Step 4: Tests pass.**

- [ ] **Step 5: Commit**

```bash
git add tp_measure.py tp_compare.py cross_clip_report.py test_cases/test_tp_measure.py
git commit -m "feat(tp_measure,tp_compare): wire Stage 3 decoder-class into JSON + HTML"
```

---

## Phase 4 — Calibration presets (optional, deferred for operator pass)

### Task 12: stage3-bursts and stage3-artifacts presets

**Files:** `tp_calibrate.py`

Append two presets sourced from the new catalogs:

```python
"stage3-bursts": [r["id"] for r in tp_chart.BURST_REGIONS],
"stage3-artifacts": [r["id"] for r in tp_chart.ARTIFACT_REGIONS],
```

Test by running `python tp_calibrate.py --list-presets` and confirming the new names appear.

Commit:
```bash
git add tp_calibrate.py
git commit -m "feat(tp_calibrate): stage3-bursts and stage3-artifacts presets"
```

---

## End-to-end validation

Run the full test suite:

```bash
for t in test_cases/test_tp_*.py test_cases/test_metrics.py; do
  echo "=== $t ==="; python "$t" 2>&1 | tail -3;
done
```

Generate a synthesized-frame report and inspect the HTML manually:

```bash
# Synthesize, encode, measure, compare
python tp_synthesize.py --raster 720x486 --output /tmp/ideal.png  # quick visual
# (existing pipeline) — confirm new sections appear in tp_compare HTML output.
```

Real-capture smoke test on a known clip:

```bash
python tp_measure.py /wintmp/analog_video/tp_compare/tpgSw2_composite_snellhdSnellGroup_pedIn_sdi.avi \
  --frame 60 --output /tmp/snellhd_stage3.json
python -c "import json; d=json.load(open('/tmp/snellhd_stage3.json')); \
  print('class:', d['decoder_class']['decoder_class'], \
  'conf:', d['decoder_class']['confidence']); \
  print('zp_present:', d['artifacts']['summary']['zone_plate_chroma_present'])"
```

If real-capture results look plausible (decoder_class consistent across frames; zp_present reasonable for the source), the implementation is ready for review.
