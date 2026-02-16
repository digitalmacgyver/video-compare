# Video Compare

No-reference quality metrics for comparing analog video capture pipelines.

Analyzes normalized ProRes 422 HQ clips (10-bit yuv422p10le, resolution auto-detected) on 9 quality metrics:
sharpness, edge strength, blocking, detail, texture quality, ringing, temporal stability, colorfulness, naturalness.

Dropped from earlier versions: noise (subordinate to detail, r=0.83), contrast & tonal richness (near-zero discrimination across devices), gradient smoothness (duplicate of texture quality, r=0.94). Ringing is retained despite high correlation with sharpness (r=0.94) because it measures a distinct analog artifact — edge overshoot from VCR sharpness circuits and aperture correction — that varies independently across hardware.

## Example Report

[View a sample report](https://digitalmacgyver.github.io/video-compare/example_report/) comparing 10 VCR/TBC configurations capturing the same source material, with interactive charts, per-metric comparison frames, and a sortable heatmap.

## Workflow

```bash
# Activate the virtual environment
source venv/bin/activate

# 1. Normalize clips to a common brightness/saturation reference
python normalize.py /path/to/clips/

# 2. Run quality analysis on the normalized clips
python quality_report.py /path/to/clips/normalized/

# 3. (Optional) Compare across multiple clip sets
python cross_clip_report.py set1.json set2.json set3.json --output comparison.html
```

### Step 1: Normalize

Brings all clips to a common brightness and saturation baseline using Y histogram matching and chroma saturation scaling. This ensures quality metrics measure artifacts and clarity, not exposure/color differences.

```
python normalize.py <src_dir> [options]

Options:
  --reference CLIP    Reference clip basename (default: first alphabetically)
  --output-dir DIR    Output directory (default: <src_dir>/normalized/)
```

Frame rate is auto-detected from source clips via ffprobe. Output format is ProRes 422 HQ.

### Step 2: Quality Report

Runs all 9 no-reference metrics on each clip, computes a rank-based overall composite score, and generates reports.

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

The `--skip` option handles timing misalignment between capture devices. PATTERN is matched as a substring of the clip filename:
```bash
# Skip 1 frame of JVC clips to align with DS555 clips
python quality_report.py /path/to/normalized/ --skip jvctbc1:1 --screenshots 5
```

**Always produces:**
- `<name>.json` — raw metric data (used by cross_clip_report.py)
- `<name>.html` — interactive HTML report with Chart.js visualizations, per-metric comparison frames with click-to-enlarge lightbox

**Optionally produces:**
- `<name>.txt` — plain-text report with rankings (with `--text`)

Use `--pattern` to select specific file subsets in a directory:
```bash
# Only analyze .mp4 files with a specific suffix
python quality_report.py /path/to/normalized/ --pattern "*_sls.mp4" --name sls_report
python quality_report.py /path/to/normalized/ --pattern "*_starlight_mini.mp4" --name starlight_mini_report
```

### Step 3: Cross-Clip Comparison (Optional)

Generates a comparison HTML report from two or more quality report JSONs, showing how the same devices perform across different source material.

```
python cross_clip_report.py <json1> <json2> [json3...] --output PATH
```

### Utility: Verify Normalization

```bash
python verify_signalstats.py
```

Runs ffprobe signalstats on normalized clips and prints a convergence table to verify normalization quality.

## Metrics

| Metric | Method | Direction |
|--------|--------|-----------|
| Sharpness | Laplacian variance | Higher = better |
| Edge Strength | Sobel gradient mean | Higher = better |
| Blocking | 8x8 DCT boundary ratio | Closer to 1.0 = better |
| Detail | Local variance median (16x16) | Higher = better |
| Texture Quality | Structure/noise ratio | Higher = better |
| Ringing | Near-edge Laplacian energy | Lower = better |
| Temporal Stability | Frame-to-frame diff mean | Lower = better |
| Colorfulness | Hasler-Susstrunk metric | Higher = better |
| Naturalness | MSCN kurtosis | Higher = better |

The overall composite uses rank-based scoring: each clip is ranked 1 (best) to N (worst) per metric, and the overall score is the average rank across all 9 metrics. Equal weight, no single metric can dominate.

## Requirements

- Python 3.8+
- ffmpeg / ffprobe
- numpy, opencv-python-headless, scipy

```bash
python -m venv venv
source venv/bin/activate
pip install numpy opencv-python-headless scipy
```
