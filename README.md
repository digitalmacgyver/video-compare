# Video Compare

No-reference quality metrics for comparing analog video capture pipelines.

Analyzes normalized ProRes 422 HQ clips (10-bit yuv422p10le, resolution auto-detected) on 13 metrics:
- **8 Technical**: sharpness, edge strength, noise, blocking, detail, texture quality, ringing, temporal stability
- **5 Perceptual**: contrast, colorfulness, tonal richness, naturalness, gradient smoothness

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

Runs all 12 no-reference metrics on each clip, computes composite z-scores, and generates reports.

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
| Noise | Flat-region high-freq energy | Lower = better |
| Blocking | 8x8 DCT boundary ratio | Closer to 1.0 = better |
| Detail | Local variance median (16x16) | Higher = better |
| Texture Quality | Structure/noise ratio | Higher = better |
| Ringing | Near-edge Laplacian energy | Lower = better |
| Temporal Stability | Frame-to-frame diff mean | Lower = better |
| Contrast | RMS contrast | Higher = better |
| Colorfulness | Hasler-Susstrunk metric | Higher = better |
| Tonal Richness | Y histogram entropy | Higher = better |
| Naturalness | MSCN kurtosis | Higher = better |
| Gradient Smoothness | Gradient continuity | Higher = better |

Composite scores use discrimination-weighted z-score normalization. Metrics with more spread across devices receive more influence. Technical and perceptual sub-composites are weighted equally in the overall score.

## Requirements

- Python 3.8+
- ffmpeg / ffprobe
- numpy, opencv-python-headless, scipy

```bash
python -m venv venv
source venv/bin/activate
pip install numpy opencv-python-headless scipy
```
