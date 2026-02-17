# Video Compare

No-reference quality metrics for comparing analog video capture pipelines. Analyzes ProRes 422 HQ clips on 11 brightness-agnostic quality metrics, producing interactive HTML reports with Chart.js visualizations, per-metric comparison frames with A/B slider, zoomable lightbox, and sortable heatmaps. No upstream brightness normalization required.

## Example Report

[View a sample report](https://digitalmacgyver.github.io/video-compare/example_report/) comparing 10 VCR/TBC configurations capturing the same source material.

## Workflow

```bash
# Activate the virtual environment
source venv/bin/activate

# Run quality analysis directly on clips (no normalization needed)
python quality_report.py /path/to/clips/

# With sample screenshots and timing alignment
python quality_report.py /path/to/clips/ --screenshots 5 --skip jvctbc1:1

# For non-.mov files, use --pattern
python quality_report.py /path/to/clips/ --pattern "*_sls.mp4" --name sls_report

# (Optional) Compare across multiple clip sets
python cross_clip_report.py set1.json set2.json set3.json --output comparison.html
```

### Quality Report

Runs all 11 no-reference metrics on each clip, computes a rank-based overall composite score, and generates reports.

```
python quality_report.py <src_dir> [options]

Options:
  --output-dir DIR    Where to write reports (default: same as src_dir)
  --name PREFIX       Report filename prefix (default: "quality_report")
  --pattern GLOB      File glob pattern (default: "*.mov")
  --text              Also generate a plain-text report
  --screenshots N     Embed N evenly-spaced sample frames per clip in HTML (default: 0)
  --skip PATTERN:N    Skip N initial frames for clips matching PATTERN (repeatable)
```

The `--skip` option handles timing misalignment between capture devices. PATTERN is matched as a substring of the clip filename.

**Always produces:**
- `<name>.json` — raw metric data (used by cross_clip_report.py)
- `<name>.html` — interactive HTML report with Chart.js visualizations, per-metric comparison frames with zoomable lightbox and A/B comparison slider

**Optionally produces:**
- `<name>.txt` — plain-text report with rankings (with `--text`)

### Cross-Clip Comparison (Optional)

Generates a comparison HTML report from two or more quality report JSONs, showing how the same devices perform across different source material.

```
python cross_clip_report.py <json1> <json2> [json3...] --output PATH
```

### Normalize (Optional)

Brings clips to a common brightness and saturation baseline. Not required for metrics (all metrics are brightness-agnostic), but useful for visual side-by-side comparison.

```
python normalize.py <src_dir> [--reference CLIP] [--output-dir DIR]
```

### Verify Normalization (Optional)

```bash
python verify_signalstats.py
```

Runs ffprobe signalstats on normalized clips and prints a convergence table.

## Metrics

All metrics are brightness-agnostic — they normalize by mean Y or use inherently scale-invariant computations, so no upstream brightness normalization is needed.

| Metric | Method | Direction |
|--------|--------|-----------|
| Sharpness | Laplacian variance / mean Y² | Higher = better |
| Edge Strength | Sobel gradient mean / mean Y | Higher = better |
| Blocking | 8x8 DCT boundary ratio | Closer to 1.0 = better |
| Detail | Local coefficient of variation median (16x16) | Higher = better |
| Texture Quality | Structure/noise ratio | Higher = better |
| Ringing | Near-edge Laplacian energy / mean Y | Lower = better |
| Temporal Stability | Frame-to-frame diff / mean Y | Lower = better |
| Colorfulness | Hasler-Susstrunk metric (chroma planes) | Higher = better |
| Naturalness | MSCN kurtosis | Higher = better |
| Crushed Blacks | Shadow headroom ratio (%) | Lower = better |
| Blown Whites | Highlight headroom ratio (%) | Lower = better |

The overall composite uses rank-based scoring: each clip is ranked 1 (best) to N (worst) per metric, and the overall score is the average rank across all 11 metrics. Equal weight, no single metric can dominate.

## Approach

All metrics are designed to be brightness-agnostic, eliminating the need for upstream brightness normalization. Metrics that compute variance or gradient magnitudes are divided by mean Y (or mean Y²) to remove dependence on capture brightness/gain settings. Crushed blacks and blown whites use headroom ratios — the fraction of shadow/highlight pixels clipped to near-black/near-white — which are inherently brightness-independent.

Eleven complementary metrics were selected to cover sharpness, detail preservation, artifact detection, temporal behavior, color, signal headroom, and naturalness. Metrics with high inter-correlation or near-zero discrimination were excluded during development to avoid redundant scoring. Ringing is the one exception — despite correlating with sharpness, it captures a distinct analog artifact (edge overshoot from VCR sharpness circuits and aperture correction) that varies independently across hardware.

The HTML report includes per-metric comparison frames showing all clips at the same frame number, selected near the 90th percentile of the best-scoring clip for each metric. An inline A/B comparison slider lets you directly compare any two images side-by-side. Both the lightbox and A/B comparison support mouse wheel zoom (centered on cursor), drag-to-pan, pinch-to-zoom on mobile, and 'R' key to reset.

## Testing

A synthetic test suite validates all 11 metrics plus brightness invariance:

```bash
python test_cases/test_metrics.py
```

See [test_cases/README.md](test_cases/README.md) for details on the 14 test cases.

## Requirements

- Python 3.8+
- ffmpeg / ffprobe
- numpy, opencv-python-headless, scipy

```bash
python -m venv venv
source venv/bin/activate
pip install numpy opencv-python-headless scipy
```
