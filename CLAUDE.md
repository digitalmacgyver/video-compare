# Video Compare — Claude Code Instructions

## Project Overview

No-reference quality metrics pipeline for comparing analog video capture devices. Resolution auto-detected via ffprobe; tested with ProRes 422 HQ 10-bit yuv422p10le clips at various resolutions.

## Scripts

- `quality_report.py` — Full analysis + HTML/JSON report generation; also supports `--from-json` for report-from-JSON mode
- `quality_metrics.py` — Metrics-only computation (outputs timestamped JSON, no HTML)
- `cross_clip_report.py` — Compare device performance across multiple clip sets
- `common.py` — Shared constants, composite ranking, video I/O functions
- `normalize.py` — Histogram-match brightness/saturation normalization
- `normalize_linear.py` — Simple linear Y-scale brightness normalization
- `verify_signalstats.py` — Validate normalization convergence

## Workflow

```bash
source venv/bin/activate

# Full analysis + report (default: 14 metrics including extra detail metrics)
python quality_report.py /path/to/clips/ --screenshots 5

# For non-.mov files, use --pattern; use --skip for timing alignment
python quality_report.py /path/to/clips/ --pattern "*_sls.mp4" --name sls_report --skip jvctbc1:1

# Incremental workflow: compute metrics for a single new clip
python quality_metrics.py /path/to/clips/ --pattern "new_clip*.mov" --name new_clip

# Merge pre-computed JSONs into a combined report
python quality_report.py --from-json existing.json new_clip.json --name combined

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
- MSCN kurtosis used instead of GGD beta fitting (beta~2.0 for all ProRes clips, no discrimination)
- 14 metrics by default (11 core + 3 extra detail), rank-based overall composite:
  - Core: sharpness, edge_strength, blocking, detail, texture_quality, ringing, temporal_stability,
    colorfulness, naturalness, crushed_blacks, blown_whites
  - Extra detail (default on): detail_perceptual, detail_blur_inv, detail_sml
- All metrics are brightness-agnostic — no upstream brightness normalization is required
  - sharpness, edge_strength, detail, ringing, temporal_stability: normalized by mean Y
  - blocking, texture_quality: inherently scale-invariant (ratios)
  - naturalness: MSCN divides by local sigma
  - colorfulness: operates on chroma planes
  - crushed_blacks/blown_whites: headroom ratios
- Discrimination gating: metrics with CV < 1% are automatically excluded from composite ranking
- Dropped metrics: noise (r=0.83 with detail), contrast/tonal_richness (near-zero discrimination), grad_smoothness (r=0.94 with texture_quality)
- Ringing retained despite r=0.94 with sharpness — distinct analog artifact (VCR sharpness circuits, aperture correction)
- JSON filenames include timestamps (_YYYYMMDD_HHMMSS) for data provenance; HTML embeds source JSON reference
- Comparison frames use p90 percentile selection from the best-scoring clip per metric

## Data Locations

- Original 16-clip dataset: `/wintmp/analog_video/noref_compare/`
- Processor comparison clips: `/wintmp/analog_video/noref_compare/processor_comparison/`
- VCR bakeoff clips: `/wintmp/analog_video/vcr_bakeoff/clip0{2,3,4}/`
- Normalized outputs in `normalized/` subdirectories of each source

## Testing

- Run `python test_cases/test_metrics.py` to validate all metrics before committing changes
- Any metric function change (thresholds, kernels, formulas) must pass the existing test suite
- New metrics must include a corresponding synthetic test case in `test_cases/test_metrics.py`

## Code Style

- Single-file scripts with shared module `common.py`
- argparse for CLI interfaces
- NumPy/OpenCV for frame processing, ffmpeg subprocess pipes for I/O
- Chart.js 4.x from CDN for HTML report visualizations
- Dark-themed HTML reports with CSS custom properties
