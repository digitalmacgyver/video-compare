# Video Compare — Claude Code Instructions

## Project Overview

No-reference quality metrics pipeline for comparing analog video capture devices. Resolution auto-detected via ffprobe; tested with ProRes 422 HQ 10-bit yuv422p10le clips at various resolutions.

## Scripts

- `normalize.py` — Normalize brightness/saturation across clips
- `quality_report.py` — Analyze clips and produce JSON + HTML reports
- `cross_clip_report.py` — Compare device performance across multiple clip sets
- `verify_signalstats.py` — Validate normalization convergence

## Workflow

```bash
source venv/bin/activate

# Generate quality report directly on raw clips (no normalization needed)
python quality_report.py /path/to/clips/ --screenshots 5

# For non-.mov files, use --pattern; use --skip for timing alignment
python quality_report.py /path/to/clips/ --pattern "*_sls.mp4" --name sls_report --skip jvctbc1:1

# Cross-clip comparison from multiple JSON reports
python cross_clip_report.py report1.json report2.json --output comparison.html

# Normalize clips if needed for visual comparison (not required for metrics)
python normalize.py /path/to/clips/ --reference best_clip.mov
```

## Python Environment

- Virtual environment at `./venv/`; activate with `source venv/bin/activate`
- Dependencies listed in `requirements.txt`: numpy, opencv-python-headless, scipy
- Install with: `pip install -r requirements.txt`
- ffmpeg 4.4.2 available system-wide

## Technical Constraints

- Resolution and frame rate are auto-detected via ffprobe
- ProRes encode uses `-c:v prores_ks -profile:v 3 -vendor apl0`
- OpenCV headless does NOT include `cv2.quality` (no BRISQUE/NIQE)
- Noise estimation uses flat-region approach (Sobel gradient < 0.03) to avoid confusing blur with low noise
- MSCN kurtosis used instead of GGD beta fitting (beta~2.0 for all ProRes clips, no discrimination)
- 11 metrics with rank-based overall composite (average rank across all metrics, equal weight):
  sharpness, edge_strength, blocking, detail, texture_quality, ringing, temporal_stability,
  colorfulness, naturalness, crushed_blacks, blown_whites
- All metrics are brightness-agnostic — no upstream brightness normalization is required
  - sharpness, edge_strength, detail, ringing, temporal_stability: normalized by mean Y
  - blocking, texture_quality: inherently scale-invariant (ratios)
  - naturalness: MSCN divides by local sigma
  - colorfulness: operates on chroma planes
  - crushed_blacks/blown_whites: headroom ratios (fraction of shadow/highlight pixels at floor/ceiling)
- Dropped metrics: noise (r=0.83 with detail), contrast/tonal_richness (near-zero discrimination), grad_smoothness (r=0.94 with texture_quality)
- Ringing retained despite r=0.94 with sharpness — distinct analog artifact (VCR sharpness circuits, aperture correction)
- Comparison frames use p90 percentile selection from the best-scoring clip per metric
- CSS-only lightbox (checkbox hack) for click-to-enlarge on all embedded images

## Data Locations

- Original 14-clip dataset: `/wintmp/analog_video/noref_compare/`
- VCR bakeoff clips: `/wintmp/analog_video/vcr_bakeoff/clip0{2,3,4}/`
- Old reports archive: `/wintmp/analog_video/video_compare/`
- Normalized outputs in `normalized/` subdirectories of each source

## Testing

- Run `python test_cases/test_metrics.py` to validate all 11 metrics before committing changes to `quality_report.py`
- Any metric function change (thresholds, kernels, formulas) must pass the existing test suite
- New metrics must include a corresponding synthetic test case in `test_cases/test_metrics.py`

## Code Style

- Single-file scripts, no package structure
- argparse for CLI interfaces
- NumPy/OpenCV for frame processing, ffmpeg subprocess pipes for I/O
- Chart.js 4.x from CDN for HTML report visualizations
- Dark-themed HTML reports with CSS custom properties
