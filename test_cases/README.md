# Metric Validation Test Suite

Synthetic test images that verify each of the 11 quality metrics responds correctly to its target artifact, plus brightness invariance checks. No external test databases required — all images are generated with NumPy/OpenCV for full reproducibility.

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
| Crushed Blacks | crushed_blacks | Smooth gradient (shadows spread) | Same with dark pixels clamped to 0.03 |
| Blown Whites | blown_whites | Smooth gradient (highlights spread) | Same with bright pixels clamped to 0.97 |
| Brightness Inv. | detail, sharpness, edge_strength | Sharp scene at 1.0x brightness | Same scene at 0.3x brightness (~equal) |

## Assets

The `assets/` directory contains 16 PNG files (good/bad pairs) saved by the script for visual inspection. Temporal stability and brightness invariance are tested numerically and share assets with other tests.

## Adding Tests for New Metrics

1. Write a generator function that produces a good/bad image pair where the metric should clearly distinguish the two
2. Save PNGs to `assets/` for visual reference
3. Append a `(test_name, metric_name, v_good, v_bad, expected_direction, passed)` tuple to the results list
4. Run the script and verify the new test passes
