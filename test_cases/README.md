# Metric Validation Test Suite

Synthetic test images that verify each of the 9 quality metrics responds correctly to its target artifact. No external test databases required — all images are generated with NumPy/OpenCV for full reproducibility.

## Running

```bash
source venv/bin/activate
python test_cases/test_metrics.py
```

Exit code 0 if all tests pass, 1 otherwise.

## Test Cases

| Test | Metrics | Good | Bad |
|------|---------|------|-----|
| Sharp vs. Blur | sharpness, edge_strength, detail | Composite scene (edges, checkerboard, zone plate) | Same + Gaussian blur (sigma=4) |
| Ringing | ringing | Smooth-edged bars | Same + damped-cosine oscillations near edges |
| Blocking | blocking | Clean diagonal gradient | Same JPEG-roundtripped at Q=5 |
| Texture Quality | texture_quality | Sinusoidal gratings | Same + Gaussian noise (sigma=0.08) |
| Colorfulness | colorfulness | Full-gamut color bars (10-bit YCbCr) | Same with chroma scaled to 25% |
| Naturalness | naturalness | Multi-scale random rectangles | Same + heavy Gaussian noise (sigma=0.20) |
| Temporal Stability | temporal_stability | Static frames + tiny noise | Same + sinusoidal brightness flicker |

## Assets

The `assets/` directory contains 12 PNG files (good/bad pairs) saved by the script for visual inspection. Temporal stability is tested numerically with frame-pair arrays and has no corresponding PNGs.

## Adding Tests for New Metrics

1. Write a generator function that produces a good/bad image pair where the metric should clearly distinguish the two
2. Save PNGs to `assets/` for visual reference
3. Append a `(test_name, metric_name, v_good, v_bad, expected_direction, passed)` tuple to the results list
4. Run the script and verify the new test passes
