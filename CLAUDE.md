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

# Normalize a set of clips (auto-detects frame rate and resolution)
python normalize.py /path/to/clips/ --reference best_clip.mov

# Generate quality report (JSON + HTML) with comparison frames and sample screenshots
python quality_report.py /path/to/clips/normalized/ --screenshots 5

# For non-.mov files, use --pattern; use --skip for timing alignment
python quality_report.py /path/to/clips/ --pattern "*_sls.mp4" --name sls_report --skip jvctbc1:1

# Cross-clip comparison from multiple JSON reports
python cross_clip_report.py report1.json report2.json --output comparison.html
```

## Python Environment

- Virtual environment at `./venv/`; activate with `source venv/bin/activate`
- Dependencies: numpy, opencv-python-headless, scipy
- ffmpeg 4.4.2 available system-wide

## Technical Constraints

- Resolution and frame rate are auto-detected via ffprobe
- ProRes encode uses `-c:v prores_ks -profile:v 3 -vendor apl0`
- OpenCV headless does NOT include `cv2.quality` (no BRISQUE/NIQE)
- Noise estimation uses flat-region approach (Sobel gradient < 0.03) to avoid confusing blur with low noise
- MSCN kurtosis used instead of GGD beta fitting (beta~2.0 for all ProRes clips, no discrimination)
- 13 metrics: 8 technical + 5 perceptual; discrimination-weighted composite scores
- Comparison frames use p90 percentile selection from the best-scoring clip per metric
- CSS-only lightbox (checkbox hack) for click-to-enlarge on all embedded images

## Data Locations

- Original 14-clip dataset: `/wintmp/analog_video/noref_compare/`
- VCR bakeoff clips: `/wintmp/analog_video/vcr_bakeoff/clip0{2,3,4}/`
- Old reports archive: `/wintmp/analog_video/video_compare/`
- Normalized outputs in `normalized/` subdirectories of each source

## Code Style

- Single-file scripts, no package structure
- argparse for CLI interfaces
- NumPy/OpenCV for frame processing, ffmpeg subprocess pipes for I/O
- Chart.js 4.x from CDN for HTML report visualizations
- Dark-themed HTML reports with CSS custom properties
