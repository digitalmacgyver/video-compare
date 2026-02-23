# Video Compare

No-reference quality metrics for comparing analog video capture pipelines. Analyzes ProRes 422 HQ clips on 14 brightness-agnostic quality metrics, producing interactive HTML reports with Chart.js visualizations, per-metric comparison frames with A/B slider, zoomable lightbox, and sortable heatmaps. No upstream brightness normalization required.

## Example Report

[View a sample report](https://digitalmacgyver.github.io/video-compare/example_report/) comparing 16 VCR/TBC configurations capturing the same source material.

## Workflow

```bash
# Activate the virtual environment
source venv/bin/activate

# Run quality analysis directly on clips (no normalization needed)
python quality_report.py /path/to/clips/ --screenshots 5

# For non-.mov files, use --pattern; use --skip for timing alignment
python quality_report.py /path/to/clips/ --pattern "*_sls.mp4" --name sls_report --skip jvctbc1:1

# Compute metrics only (no HTML) — useful for incremental workflows
python quality_metrics.py /path/to/clips/ --pattern "new_clip*.mov" --name new_clip

# Generate report from pre-computed JSON files (merge multiple runs)
python quality_report.py --from-json existing.json new_clip.json --name combined

# Compare across multiple clip sets
python cross_clip_report.py set1.json set2.json set3.json --output comparison.html
```

## Scripts

### quality_report.py — Full Analysis + Report

Computes metrics and generates HTML/JSON reports in one step. Also supports `--from-json` to generate reports from pre-computed metric JSON files.

```
python quality_report.py <src_dir> [options]
python quality_report.py --from-json <json1> [json2 ...] [options]

Options:
  --output-dir DIR              Where to write reports (default: same as src_dir)
  --name PREFIX                 Report filename prefix (default: "quality_report")
  --pattern GLOB                File glob pattern (default: "*.mov")
  --text                        Also generate a plain-text report
  --screenshots N               Embed N evenly-spaced sample frames per clip in HTML (default: 0)
  --skip PATTERN:N              Skip N initial frames for clips matching PATTERN (repeatable)
  --extra-detail-metrics CSV    Extra detail metrics to compute (default: detail_perceptual,detail_blur_inv,detail_sml)
  --report-metrics CSV          Metric keys to include in report (default: all 14)
  --from-json JSON [JSON ...]   Generate report from pre-computed JSON files instead of analyzing clips
```

**Always produces:**
- `<name>_YYYYMMDD_HHMMSS.json` — timestamped metric data with provenance metadata
- `<name>.html` — interactive HTML report with Chart.js charts, heatmap, comparison frames, and A/B slider

### quality_metrics.py — Metrics Only (No Report)

Computes metrics and outputs a timestamped JSON file. Useful for incremental workflows — analyze a single new clip, then merge with existing data via `quality_report.py --from-json`.

```
python quality_metrics.py <src_dir> [options]

Options:
  --output-dir DIR              Output directory for JSON (default: same as src_dir)
  --name PREFIX                 JSON filename prefix (default: "quality_metrics")
  --pattern GLOB                File glob pattern (default: "*.mov")
  --skip PATTERN:N              Skip N initial frames for clips matching PATTERN (repeatable)
  --extra-detail-metrics CSV    Extra detail metrics to compute (default: detail_perceptual,detail_blur_inv,detail_sml)
```

### cross_clip_report.py — Cross-Clip Comparison

Generates a comparison HTML report from two or more quality report JSONs, showing how the same devices perform across different source material.

```
python cross_clip_report.py <json1> <json2> [json3...] --output PATH
```

### normalize.py / normalize_linear.py — Brightness Normalization (Optional)

Brings clips to a common brightness baseline. Not required for metrics (all are brightness-agnostic), but useful for visual side-by-side comparison. `normalize.py` uses histogram matching; `normalize_linear.py` uses simple linear Y scaling.

```
python normalize.py <src_dir> [--reference CLIP] [--output-dir DIR] [--pattern GLOB]
```

### common.py — Shared Module

Shared constants (ALL_KEYS, METRIC_INFO, color palettes), composite ranking logic, and video I/O functions used by all scripts.

## Metrics

All metrics are brightness-agnostic — they normalize by mean Y or use inherently scale-invariant computations.

### Core Metrics (11)

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

### Extra Detail Metrics (3, computed by default)

| Metric | Method | Direction |
|--------|--------|-----------|
| Detail VHS-SD | Perceptual detail tuned for SD analog VHS | Higher = better |
| Detail BlurInv | Inverse blur effect — directional blur resistance | Higher = better |
| Detail SML | Modified Laplacian — fine transition clarity | Higher = better |

The overall composite uses rank-based scoring: each clip is ranked 1 (best) to N (worst) per metric, and the overall score is the average rank. Metrics with near-zero discrimination (CV < 1%) are automatically excluded from the composite.

## Testing

A synthetic test suite validates all metrics plus brightness invariance:

```bash
python test_cases/test_metrics.py
```

See [test_cases/README.md](test_cases/README.md) for details.

## Requirements

- Python 3.8+
- ffmpeg / ffprobe
- Python dependencies listed in `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
