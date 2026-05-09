# Metric Implementation Review

Per-function analysis of all 11 metrics in `quality_report.py` plus the temporal stability computation in `analyze_clip()`, with findings on correctness, brightness invariance, numerical stability, resolution sensitivity, and aggregation.

---

## 1. `sharpness_laplacian(yf)` — Lines 144–147

```python
def sharpness_laplacian(yf):
    mean_y = np.mean(yf)
    return np.var(cv2.Laplacian(yf, cv2.CV_64F)) / (mean_y ** 2 + 1e-10)
```

### Algorithm
Variance of the Laplacian response, normalized by mean brightness squared. The Laplacian (default 3x3 kernel `[[0,1,0],[1,-4,1],[0,1,0]]`) detects second-derivative features — sharp edges and fine texture produce large Laplacian responses. Variance measures the spread of these responses (a globally blurred image has low Laplacian variance even if it has gradients).

### Brightness Invariance
If `yf` is scaled by `k`: `Laplacian(k*yf) = k * Laplacian(yf)`, so `Var(Laplacian(k*yf)) = k² * Var(Laplacian(yf))`. And `mean(k*yf)² = k² * mean(yf)²`. Ratio cancels `k²`. **Correct.**

### Numerical Stability
Division by `(mean_y² + 1e-10)`. For all-black frames, both numerator and denominator are near zero, result ≈ 0. **Adequate.**

### Direction
`higher_better=True`. Higher Laplacian variance = sharper. **Correct.**

### Aggregation
Mean across frames in `analyze_clip()`. Appropriate for a roughly symmetric distribution.

### Issues

**MINOR — Noise confound.** The Laplacian is a high-pass filter. Additive noise (tape noise, head preamp noise) increases Laplacian variance, making noisy clips appear sharper. The metric measures "amount of high-frequency energy" without distinguishing sharpness from noise. This is partially mitigated by `texture_quality` which captures the "quality of detail" dimension, but sharpness itself will over-score noisy captures. No code fix needed — this is an inherent limitation of Laplacian variance as a sharpness measure.

**MINOR — Resolution portability.** The 3x3 kernel has a fixed spatial frequency response. At different resolutions, the same physical content produces different sharpness values. Not an issue when all clips have verified same resolution (enforced at lines 1506–1511).

### Verdict
**Sound implementation.** Well-normalized, numerically stable, correct direction. Noise confound is a known limitation.

---

## 2. `edge_strength_sobel(yf)` — Lines 150–155

```python
def edge_strength_sobel(yf):
    sx = cv2.Sobel(yf, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(yf, cv2.CV_64F, 0, 1, ksize=3)
    mean_y = np.mean(yf)
    return np.mean(np.sqrt(sx * sx + sy * sy)) / (mean_y + 1e-10)
```

### Algorithm
Mean Sobel gradient magnitude (L2 norm of horizontal and vertical first derivatives) normalized by mean brightness. Measures average edge energy — complements sharpness (variance of second derivative) by capturing first-derivative magnitude.

### Brightness Invariance
`Sobel(k*yf) = k * Sobel(yf)`. Mean gradient scales by `k`, mean brightness scales by `k`. Ratio cancels. **Correct.**

### Numerical Stability
Division by `(mean_y + 1e-10)`. **Adequate.**

### Direction
`higher_better=True`. **Correct.**

### Aggregation
Mean across frames. Appropriate.

### Issues

**MINOR — Same noise confound as sharpness.** Noise increases gradient magnitude, inflating edge strength for noisy captures.

**MINOR — Redundancy with sharpness.** Both measure spatial frequency content. Empirical correlation should be checked per the metric selection review. If r > 0.90, the incremental information is minimal.

### Verdict
**Sound implementation.** Correctly normalized and stable. Potential redundancy with sharpness is a metric selection concern, not an implementation issue.

---

## 3. `blocking_artifact_measure(yf)` — Lines 193–204

```python
def blocking_artifact_measure(yf):
    h, w = yf.shape
    h_diffs = np.abs(yf[:, 1:] - yf[:, :-1])
    h_boundary = np.arange(1, w) % 8 == 0
    h_ratio = np.mean(h_diffs[:, h_boundary]) / max(np.mean(h_diffs[:, ~h_boundary]), 1e-10)
    v_diffs = np.abs(yf[1:, :] - yf[:-1, :])
    v_boundary = np.arange(1, h) % 8 == 0
    v_ratio = np.mean(v_diffs[v_boundary, :]) / max(np.mean(v_diffs[~v_boundary, :]), 1e-10)
    return (h_ratio + v_ratio) / 2.0
```

### Algorithm
Wang et al. (2000) blocking metric. Compares mean adjacent-pixel differences at 8-pixel block boundaries versus non-boundary positions. Ratio > 1.0 indicates blocking artifacts; ~1.0 indicates no blocking.

### Boundary Index Verification
`h_diffs[:, j]` = `|yf[:, j+1] - yf[:, j]|` for `j ∈ [0, w-2]`. `h_boundary[j]` = True when `(j+1) % 8 == 0`, i.e., `j ∈ {7, 15, 23, ...}`. These select `|yf[:, 8] - yf[:, 7]|`, `|yf[:, 16] - yf[:, 15]|`, etc. — the correct boundaries between 8-pixel blocks. **Correct.**

### Brightness Invariance
Ratio metric: if all pixel values scale by `k`, all diffs scale by `k`, ratio cancels. **Inherently scale-invariant.**

### Numerical Stability
`max(..., 1e-10)` prevents division by zero. **Adequate.**

### Direction
`higher_better=None` (closer to 1.0 = better). **Correct.**

### Issues

**MINOR — Grid alignment assumption.** The metric assumes the DCT block grid starts at pixel (0, 0). If the frame has been cropped or the active picture doesn't align with the 8-pixel grid, boundary positions are wrong. Not an issue for uncropped captures.

**MINOR — Partial blocks.** For 576x436: width 576 = 72×8 (exact), height 436 = 54×8 + 4 (partial block at bottom). The boundary check doesn't distinguish the partial block. The last "boundary" at row 432 (position 431 in the diff array where (431+1) % 8 = 0) is a real block boundary; the partial bottom rows are simply included in non-boundary statistics. Negligible impact.

**MINOR — Discrimination on ProRes.** ProRes 422 HQ uses DCT at extremely high bitrates. All clips likely score near 1.0 with negligible spread. The metric serves as a safety check for non-ProRes inputs but may contribute noise to the composite for the current dataset.

### Verdict
**Correct implementation.** Clean and efficient. Low discrimination expected for ProRes content.

---

## 4. `detail_texture(yf)` — Lines 207–218

```python
def detail_texture(yf):
    block = 16
    h, w = yf.shape
    cvs = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            blk = yf[r:r+block, c:c+block]
            mu = np.mean(blk)
            if mu > 0.01:
                cvs.append(np.std(blk) / mu)
    return float(np.median(cvs)) if cvs else 0.0
```

### Algorithm
Local coefficient of variation (std/mean) per 16×16 block, median across all non-black blocks. CV is a classic measure of micro-contrast/texture richness.

### Brightness Invariance
`CV = std(k*blk) / mean(k*blk) = k*std(blk) / (k*mean(blk)) = std(blk) / mean(blk)`. **Inherently scale-invariant.**

### Numerical Stability
Blocks with `mu < 0.01` (near-black) are excluded to avoid division by tiny values. If all blocks are excluded, returns 0.0. **Adequate.**

### Direction
`higher_better=True`. **Correct.**

### Issues

**MINOR — Performance.** Python loop over ~972 blocks (576×436 at 16×16) per frame. For thousands of frames, this is the second-slowest metric after `texture_quality_measure`. Could be vectorized using `np.lib.stride_tricks.sliding_window_view` or reshaped block arrays.

**MINOR — Noise sensitivity.** Like sharpness, noise increases local std and thus CV. The median across blocks provides some robustness (noise affects all blocks similarly, so the median is noise-biased but stable). The pairing with `texture_quality` (which penalizes noise) partially compensates in the composite.

### Verdict
**Correct implementation.** Clean algorithm, proper handling of edge cases.

---

## 5. `texture_quality_measure(yf)` — Lines 221–240

```python
def texture_quality_measure(yf):
    block = 16
    h, w = yf.shape
    yf_smooth = cv2.GaussianBlur(yf, (5, 5), 1.0)
    ratios = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            var_orig = np.var(yf[r:r+block, c:c+block])
            var_smooth = np.var(yf_smooth[r:r+block, c:c+block])
            if var_orig > 1e-8:
                ratios.append(var_smooth / var_orig)
    return float(np.median(ratios)) if len(ratios) >= 5 else 0.5
```

### Algorithm
Ratio of smoothed-block variance to original-block variance, median across blocks. Structured detail (edges, gradients) survives mild Gaussian smoothing; unstructured noise does not. Ratio near 1.0 = clean structured detail; lower ratio = noise-contaminated detail.

### Brightness Invariance
`Var(k*yf) = k² * Var(yf)`. Smoothing commutes with scaling: `GaussianBlur(k*yf) = k * GaussianBlur(yf)`. So `Var(k*smooth) = k² * Var(smooth)`. Ratio = `k² * Var(smooth) / (k² * Var(orig))` = `Var(smooth) / Var(orig)`. **Inherently scale-invariant.**

### Numerical Stability
Flat blocks with `var_orig < 1e-8` excluded. Fallback to 0.5 if fewer than 5 valid blocks. **Adequate.**

### Direction
`higher_better=True`. **Correct.**

### Issues

**NOTE — Cross-block smoothing.** The Gaussian blur is computed on the full frame before block extraction. Pixels near block boundaries are influenced by neighboring blocks. This is actually desirable — it ensures the smoothed variance reflects the spatial scale of the Gaussian kernel, not the block size.

**MINOR — Performance.** Same Python-loop pattern as `detail_texture`. Vectorizable.

### Verdict
**Excellent implementation.** Elegant approach to separating signal from noise. Well-normalized, numerically stable.

---

## 6. `ringing_measure(yf)` — Lines 243–253

```python
def ringing_measure(yf):
    edges = cv2.Canny((yf * 255).astype(np.uint8), 50, 150)
    if edges.sum() == 0:
        return 0.0
    near_edge = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    near_edge_only = near_edge & ~edges
    if near_edge_only.sum() == 0:
        return 0.0
    mean_y = np.mean(yf)
    return np.mean(np.abs(cv2.Laplacian(yf, cv2.CV_64F))[near_edge_only > 0]) / (mean_y + 1e-10)
```

### Algorithm
1. Detect edges with Canny (8-bit, thresholds 50/150).
2. Dilate edges by 2 pixels (5×5 kernel) to find the near-edge zone.
3. Subtract the edge pixels to get the near-edge-but-not-on-edge region.
4. Measure mean absolute Laplacian energy in this annular region, normalized by mean brightness.

Ringing (Gibbs phenomenon, edge overshoot from analog sharpness circuits) manifests as high-frequency oscillations in the near-edge zone, which the Laplacian detects.

### Brightness Invariance — Partial

The Laplacian normalization is correct: `|Laplacian(k*yf)| / mean(k*yf) = k * |Laplacian(yf)| / (k * mean(yf))` cancels `k`.

**However**, the Canny edge detection operates on `(yf * 255).astype(np.uint8)` — an 8-bit image. If `yf` is scaled by `k`, the 8-bit image changes, potentially producing a **different edge mask**. The Canny thresholds (50, 150) are absolute on the gradient magnitude of the 8-bit image.

**Impact assessment:** For the intended use case (same content at slightly different capture brightness), the edge maps will be very similar because Canny's hysteresis thresholds tolerate a range of gradient magnitudes. A 20% brightness difference changes gradient magnitudes by 20%, which may shift a few marginal edges in or out of detection, but the dominant edges (high-contrast transitions) will be detected regardless. For extreme brightness differences (e.g., 0.3× vs 1.0×), the edge maps could differ significantly.

**Recommendation:** Use adaptive Canny thresholds (e.g., Otsu's method on the gradient magnitude, or percentile-based thresholds) for better brightness robustness. This is a minor concern for same-content comparisons but would matter for cross-content use.

### Numerical Stability
Two early returns for zero edges / zero near-edge region. Division by `(mean_y + 1e-10)`. **Adequate.**

### Direction
`higher_better=False`. **Correct** — higher near-edge Laplacian = more ringing.

### Issues

**MAJOR — 8-bit quantization of edge detection.** The `(yf * 255).astype(np.uint8)` conversion discards 2 bits of the 10-bit input. While this is adequate for Canny edge detection (the thresholds are coarse enough that 8-bit precision suffices), it creates the brightness-dependent edge mask issue described above.

**MINOR — Fixed dilation radius.** The 5×5 dilation kernel creates a 2-pixel-wide near-edge zone. The physical width of analog ringing depends on the circuit's time constant and varies by device. A wider zone (e.g., 7×7 or 9×9) might capture more of the ringing oscillation. At 576×436, 2 pixels ≈ 1.7% of frame width, which is narrow for typical analog ringing that can span 5–10 pixels (0.87–1.74% at this resolution). The current 2-pixel zone captures only the first oscillation peak.

**Recommendation:** Consider a larger dilation kernel (7×7 or 9×9) to capture the full ringing pattern. Alternatively, use a distance-weighted average that falls off with distance from the edge.

### Verdict
**Functional but imperfect brightness invariance.** The core algorithm is sound and domain-appropriate. The 8-bit Canny quantization creates a brightness-dependent edge mask, making the metric not fully brightness-agnostic despite the Laplacian normalization. Practical impact is minor for same-content comparisons with similar brightness levels.

---

## 7. `colorfulness_metric(u, v)` — Lines 265–273

```python
def colorfulness_metric(u, v):
    cb = u.astype(np.float64) - 512.0
    cr = v.astype(np.float64) - 512.0
    rg = cr - cb
    yb = 0.5 * (cr + cb)
    sigma_rgyb = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mu_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return float(sigma_rgyb + 0.3 * mu_rgyb)
```

### Algorithm
Implements the Hasler & Süsstrunk (2003) colorfulness metric structure (`M = σ_rgyb + 0.3 · μ_rgyb`) but on YCbCr opponent channels instead of the original RGB opponent channels.

### Channel Mapping Deviation

The original Hasler metric defines:
- `rg = R − G`
- `yb = 0.5·(R + G) − B`

The code computes:
- `rg = Cr − Cb`  (proportional to `(R−Y) − (B−Y) = R − B`, not `R − G`)
- `yb = 0.5·(Cr + Cb)`  (proportional to `0.5·(R−Y + B−Y) = 0.5·(R+B) − Y`, not `0.5·(R+G) − B`)

**Mathematical impact:** The quantities `σ_rgyb = √(Var(rg) + Var(yb))` and `μ_rgyb = √(E[rg]² + E[yb]²)` are the trace of the covariance matrix and the L2 norm of the mean vector of a 2D color representation, respectively. These quantities are **rotation-invariant** in color space — measuring them on any orthogonal pair of opponent axes gives the same result, provided the transformation between axis pairs is an isometric rotation.

The transformation from `(Cb, Cr)` to `(Cr−Cb, 0.5(Cr+Cb))` has matrix `[[-1,1],[0.5,0.5]]` with singular values `√(1.25)` and `√(0.5)` — this is **not** an isometry. So the absolute values differ from the true Hasler metric. However, more saturated content produces larger chroma spreads in any linear color representation, so **ordinal rankings are preserved** for typical content.

### Brightness Invariance
Operates entirely on Cb/Cr planes; Y (luma) is not used. **Fully brightness-agnostic.**

### Numerical Stability
No division operations. All arithmetic is straightforward. **No concerns.**

### Direction
`higher_better=True`. **Correct.**

### Issues

**MINOR — Not exactly the Hasler metric.** The non-isometric transformation means absolute values differ from the published formula. The label "Hasler-S. M" in `METRIC_INFO` is technically inaccurate. Rankings are preserved for practical purposes. To match the original formula exactly, convert Cb/Cr to RGB first.

**NOTE — Chroma subsampling.** The input `u, v` are 4:2:2 subsampled (half-width). The metric computes statistics on the subsampled planes without upsampling. This is fine — chroma spread is a global statistic unaffected by spatial resolution.

### Verdict
**Functionally correct for ranking purposes.** The non-standard channel mapping produces different absolute values from the published Hasler metric but preserves ordinal rankings. Adequate for comparison use.

---

## 8. `naturalness_nss(yf)` — Lines 314–325

```python
def naturalness_nss(yf):
    C = 1.0 / 1023.0
    mu = cv2.GaussianBlur(yf, (7, 7), 1.166)
    mu_sq = cv2.GaussianBlur(yf * yf, (7, 7), 1.166)
    sigma = np.sqrt(np.maximum(mu_sq - mu * mu, 0.0))
    mscn = (yf - mu) / (sigma + C)
    data = mscn.ravel()
    s = np.std(data)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((data - np.mean(data)) / s) ** 4) - 3.0)
```

### Algorithm
1. Compute local mean and local standard deviation using Gaussian blur (kernel 7×7, σ=1.166 — from the BRISQUE implementation).
2. Form MSCN (Mean Subtracted Contrast Normalized) coefficients: `(yf − μ_local) / (σ_local + C)`.
3. Compute excess kurtosis of the MSCN distribution.

Natural images have leptokurtic MSCN distributions (kurtosis > 0) due to sparse edges amid smooth regions. Distorted images push toward Gaussian (kurtosis → 0) or below.

### Brightness Invariance — Approximate
MSCN = `(yf − μ) / (σ + C)`. Under scaling by `k`: `μ → k·μ`, `σ → k·σ`. So MSCN → `k·(yf₀ − μ₀) / (k·σ₀ + C)`. For `k·σ₀ >> C`, this ≈ `(yf₀ − μ₀) / σ₀` = original MSCN. For flat regions where `σ₀ ≈ 0`, the constant `C` breaks the cancellation. **Approximately brightness-agnostic** for typical video content with non-trivial local contrast.

### C Constant
`C = 1/1023 ≈ 0.001`. The comment implies this is tied to the 10-bit range, but `yf` is in [0, 1], not [0, 1023]. The local sigma values are typically in [0.01, 0.3], so `C = 0.001` is appropriately small relative to typical sigma values while preventing division by zero. **Functionally fine**, though the naming/comment is slightly misleading.

### Numerical Stability
Early return of 0.0 if the global std of MSCN is < 1e-15 (all-flat frame). The `np.maximum(..., 0.0)` in sigma computation prevents sqrt of negative values from floating-point error. **Adequate.**

### Direction
`higher_better=True`. **Questionable** — see issue below.

### Aggregation
**MEDIAN** across frames (special-cased at line 428). Appropriate because kurtosis is a 4th-power statistic and extremely outlier-sensitive (a single frame with unusual content can dominate the mean). **Good design decision.**

### Issues

**MODERATE — Monotonic direction is weakly justified.** The metric is scored as `higher_better=True`, but empirical testing reveals that pathological distributions can produce extreme kurtosis. A synthetic test of a mostly-flat image with rare sharp spikes yields kurtosis ~69, far exceeding a realistic natural scene (~10). Under the current scoring, the spiky artifact image would rank as "more natural." While this specific artifact pattern is unlikely in analog video, the underlying issue — that kurtosis is unbounded above and extreme values don't correspond to quality — means the monotonic assumption is fragile. A distance-from-target approach (e.g., penalize deviation from a calibrated "natural" kurtosis range of ~5–15) would be more robust.

**MINOR — Limited discrimination.** The CLAUDE.md notes that GGD beta fitting gave β ≈ 2.0 for all ProRes clips (no discrimination), motivating the switch to kurtosis as a more sensitive probe. If kurtosis also shows low CV across clips, this metric adds noise to the composite rather than signal.

**NOTE — Double standardization.** The MSCN values are already approximately zero-mean, unit-variance by construction. The second standardization `(data − mean(data)) / s` is technically redundant for the kurtosis computation, but it ensures exact centering and unit variance, which is correct procedure for computing excess kurtosis.

### Verdict
**Implementation is correct but direction scoring is questionable.** MSCN computation follows BRISQUE faithfully. The median aggregation is a good choice. However, `higher_better=True` is not universally valid — a bounded target-range approach would be safer.

---

## 9. `crushed_blacks(yf)` — Lines 276–288

```python
def crushed_blacks(yf):
    SHADOW_CEIL = 0.15
    CRUSH_FLOOR = 0.07
    n_shadow = np.count_nonzero(yf < SHADOW_CEIL)
    if n_shadow == 0:
        return 0.0
    return float(np.count_nonzero(yf < CRUSH_FLOOR)) / float(n_shadow)
```

### Algorithm
Headroom ratio: fraction of shadow-region pixels (Y < 0.15 ≈ code 153) that are crushed to near-black (Y < 0.07 ≈ code 72). Higher ratio = more shadow detail lost to clipping.

### Threshold Values
- `SHADOW_CEIL = 0.15` → 0.15 × 1023 ≈ 153 (IRE ~15, upper shadow boundary)
- `CRUSH_FLOOR = 0.07` → 0.07 × 1023 ≈ 72 (just above legal black at 64)

These thresholds are appropriate for legal-range 10-bit video. **Reasonable values.**

### Brightness Invariance — By Design, Not Scale-Invariant
This metric uses **absolute thresholds** on the signal level. A uniformly dimmer capture will have more pixels in the shadow region and potentially a higher crushed ratio, even if shadow detail is preserved. This is intentional: the metric measures absolute signal headroom, which IS dependent on the capture chain's black level calibration.

The project documentation calls this metric "brightness-agnostic," which is somewhat misleading. It's more accurately "does not require upstream brightness normalization" — the absolute thresholds are the right approach for measuring signal clipping, but the metric IS sensitive to overall brightness/gain differences between devices.

### Numerical Stability
Returns 0.0 if no shadow pixels exist. **Adequate.**

### Direction
`higher_better=False`. **Correct.**

### Aggregation
Mean across frames, with epsilon suppression (lines 438–441): means < 1e-5 are zeroed. This prevents floating-point noise from creating meaninglessly small non-zero values. **Good.**

### Issues

**NOTE — Content dependence.** Bright content with few shadow pixels will always return 0.0 regardless of capture quality. The metric is only informative for content with significant shadow content. For same-content comparison, this is fine (all clips have the same shadow content).

### Verdict
**Correct implementation.** Simple, interpretable, appropriate thresholds. The absolute nature of the thresholds is correct for the signal-headroom measurement intent.

---

## 10. `blown_whites(yf)` — Lines 291–303

```python
def blown_whites(yf):
    HIGHLIGHT_FLOOR = 0.85
    BLOW_CEIL = 0.93
    n_highlight = np.count_nonzero(yf > HIGHLIGHT_FLOOR)
    if n_highlight == 0:
        return 0.0
    return float(np.count_nonzero(yf > BLOW_CEIL)) / float(n_highlight)
```

### Algorithm
Mirror of `crushed_blacks` for highlights. Headroom ratio: fraction of highlight pixels (Y > 0.85 ≈ code 869) that are blown to near-white (Y > 0.93 ≈ code 951).

### Threshold Values
- `HIGHLIGHT_FLOOR = 0.85` → 0.85 × 1023 ≈ 869 (IRE ~85)
- `BLOW_CEIL = 0.93` → 0.93 × 1023 ≈ 951 (just above legal white at 940)

**Reasonable values** for legal-range video.

### Brightness Invariance
Same analysis as `crushed_blacks` — absolute thresholds, by design.

### Direction
`higher_better=False`. **Correct.**

### Verdict
**Correct implementation.** Symmetric with `crushed_blacks`. Same strengths and characteristics.

---

## 11. Temporal Stability — Lines 416–418 in `analyze_clip()`

```python
if prev_yf is not None:
    mean_y = 0.5 * (np.mean(yf) + np.mean(prev_yf))
    temporal_diffs.append(np.mean(np.abs(yf - prev_yf)) / (mean_y + 1e-10))
prev_yf = yf
```

### Algorithm
Mean absolute frame difference normalized by the average mean brightness of two consecutive frames. Measures frame-to-frame luminance instability (flicker, noise temporal variation, TBC jitter).

### Brightness Invariance
`mean(|k·yf₁ − k·yf₂|) = k · mean(|yf₁ − yf₂|)`. Denominator: `0.5·(mean(k·yf₁) + mean(k·yf₂)) = k · 0.5·(mean(yf₁) + mean(yf₂))`. Ratio cancels `k`. **Correct.**

### Direction
`higher_better=False`. **Correct** — lower frame difference = more stable.

### Aggregation
Mean across frame pairs. Reported as `result["temporal_stability"] = {"mean": float(np.mean(td)), "std": float(np.std(td))}`.

### Issues

**MINOR — Confounds scene motion with capture instability.** Frame differences include both actual content motion and capture artifacts (noise variation, jitter, flicker). For same-content comparison (the primary use case), the motion component is approximately constant across clips, so the differential comparison is valid. For comparing different content, the metric is unreliable.

**MINOR — Sensitivity to scene cuts.** An unexpected scene cut produces an enormous frame difference that dominates the mean. Using median instead of mean would be more robust. However, the current data (analog captures of continuous content) is unlikely to have scene cuts.

**MINOR — Memory: `prev_yf = yf`.** This keeps a reference to the full frame array. Since `yf` is reassigned each iteration (`yf = to_float(y)`), `prev_yf` holds the previous frame. The memory usage is bounded to one extra frame. **Fine.**

### Verdict
**Correct for the intended use case.** Simple, properly normalized. The scene-motion confound is an inherent limitation, not a bug.

---

## 12. `analyze_clip()` — Frame Loop and Aggregation (Lines 379–452)

### Process Management
```python
proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
# ... process frames ...
proc.wait()
```

**Issue — No error handling on ffmpeg failure.** If ffmpeg exits with an error (corrupt file, unsupported codec), the loop reads until EOF and `proc.wait()` returns a non-zero exit code that is never checked. The result would be metrics computed on however many frames were successfully decoded, with no indication of the error. The `proc.returncode` should be checked after `proc.wait()`.

**Issue — No `try/finally` for process cleanup.** If a Python exception occurs during frame processing (e.g., numpy error), the subprocess pipe is not properly closed. The process may hang or become a zombie. Should use a context manager or `try/finally`.

### Skip Frame Implementation (Lines 393–396)
```python
for _ in range(skip_frames):
    frame = read_frame(proc.stdout, width, height)
    if frame is None:
        break
```

**Correct but wasteful.** Reads and decodes skip frames into numpy arrays only to discard them. Could use ffmpeg's `-ss` option for faster seeking, but this requires careful handling of seeking accuracy. The current approach is simple and correct.

### Epsilon Suppression (Lines 438–441)
```python
EPS = 1e-5
for k in ("crushed_blacks", "blown_whites"):
    if result[k]["mean"] < EPS:
        result[k]["mean"] = 0.0
```

**Correct.** Prevents floating-point noise from creating meaninglessly small values for these ratio metrics.

### No-Frame Edge Case (Lines 393–396, 424–429)

If `skip_frames` consumes all frames in the clip (or the clip is empty/corrupt), the frame processing loop never executes. The accumulators remain empty lists, and the aggregation code operates on empty arrays:

```python
for k, vals in accum.items():
    arr = np.array(vals)  # empty array if no frames
    avg = float(np.median(arr)) if k == "naturalness" else float(np.mean(arr))
    # np.mean([]) = nan, np.median([]) = nan
```

This produces `NaN` values in the result dict, which propagate to ranking, z-scores, and JSON output. Downstream code (percentile selection in `find_comparison_frames`, rank computation in `compute_composites`) will fail or produce incorrect results with NaN inputs.

**Fix:** Validate `n_frames > 0` immediately after the frame loop and fail with a clear error message including clip name, skip count, and total frames available.

---

## 12b. Cross-Cutting: Additive Pedestal Sensitivity

This finding spans multiple metrics and is the most significant brightness-invariance concern.

### Background

The project claims all metrics are "brightness-agnostic." The mean-Y normalization in sharpness, edge_strength, detail, ringing, and temporal stability correctly cancels **multiplicative gain** differences (e.g., one clip captured at 1.2× brightness). However, it does NOT cancel **additive pedestal offsets** (e.g., one clip has setup/pedestal shifted by +20 IRE).

In analog capture, additive pedestal offsets are common: different devices calibrate black level differently, creating a constant offset in the signal.

### Mathematical Analysis

For sharpness: `Var(Laplacian(yf + c)) / (mean(yf + c))²`. The Laplacian of a constant is zero, so `Laplacian(yf + c) = Laplacian(yf)` and the numerator is unchanged. But `mean(yf + c) = mean(yf) + c`, so the denominator changes. The sharpness value changes by a factor of `(mean(yf))² / (mean(yf) + c)²`.

Same pattern for edge_strength (denominator is `mean(yf) + c` instead of `mean(yf)`) and detail (CV = `std(blk) / (mean(blk) + c)`).

### Empirical Verification

Tested on a synthetic scene (mean_Y ≈ 0.36):

| Offset | Sharpness | Edge Str | Detail | Ringing |
|--------|-----------|----------|--------|---------|
| -0.10 | 26.86 | 0.958 | 0.711 | 3.351 |
| -0.05 | 21.78 | 0.869 | 0.631 | 3.021 |
| 0.00 | 18.02 | 0.796 | 0.568 | 2.752 |
| +0.05 | 13.92 | 0.700 | 0.229 | 2.416 |
| +0.10 | 11.07 | 0.624 | 0.212 | 2.156 |

A +0.05 offset (realistic for analog pedestal differences) changes sharpness by ~23%, edge_strength by ~12%, and detail by ~60%. Detail is most sensitive because the block means are small (many dark blocks), making the additive offset proportionally large.

### Impact

Capture pipelines with different setup/pedestal calibration can be penalized or rewarded independently of actual detail quality. A device with slightly raised black level will appear less sharp, less detailed, and have less ringing than an identical device with lower pedestal — even when the actual picture quality is the same.

### Severity

**HIGH for datasets with known pedestal variation.** For same-source comparisons where all devices capture from the same signal chain (common pedestal), the impact is minimal. For comparisons across different signal paths or equipment with different setup levels, this can distort rankings significantly.

### Remediation Options

1. **Document the limitation.** Reframe "brightness-agnostic" as "gain-invariant" and document pedestal sensitivity.
2. **Estimate and subtract pedestal.** Before metric computation, estimate the black level (e.g., 5th percentile of Y) and subtract it: `yf_corrected = yf - percentile(yf, 5)`. This makes the metrics invariant to additive offsets.
3. **Use local-contrast normalization.** Replace mean-Y normalization with local contrast measures that inherently reject DC shifts (e.g., normalize Laplacian by local std instead of global mean).
4. **Recommend `normalize_linear.py` preprocessing.** For datasets with known pedestal differences, run linear normalization first.

Option 2 is the most practical near-term fix (small code change, preserves existing metric semantics).

---

## 13. `compute_composites()` — Ranking and Z-Scores (Lines 573–619)

### Tied-Rank Bug

```python
order = np.argsort(-arr)  # or np.argsort(arr)
rank_arr = np.empty(n, dtype=float)
for r, idx in enumerate(order, 1):
    rank_arr[idx] = r
```

**BUG — Tied values receive different ranks.** `np.argsort` breaks ties by array position (index order). For clips with identical metric values, the clip earlier in the `sorted(data.keys())` list (alphabetically first) receives the better rank. This creates a systematic alphabetical bias.

**Example:** If clips "alpha" and "beta" both have sharpness = 0.5000, `np.argsort` assigns alpha rank 1 and beta rank 2 (or vice versa depending on sort direction). The correct treatment is to assign both rank 1.5 (average of 1 and 2).

**Fix:** Replace with `scipy.stats.rankdata`:
```python
from scipy.stats import rankdata

for key in ALL_KEYS:
    arr = np.array([data[c][key]["mean"] for c in clip_names])
    _, _, hb = METRIC_INFO[key]
    if hb is True:
        ranks[key] = rankdata(-arr, method='average')
    elif hb is False:
        ranks[key] = rankdata(arr, method='average')
    else:
        ranks[key] = rankdata(np.abs(arr - 1.0), method='average')
```

**Impact — empirically confirmed:** On the 16-clip noref_compare dataset, `blown_whites` is constant zero for all clips. With the current code, argsort assigns arbitrary ranks 1–16 for these identical values, injecting a full 1–16 rank spread into the composite from a metric with zero discrimination. This alone can shift overall rankings by up to 15/11 ≈ 1.4 rank positions. Combined with potential ties on other metrics (crushed_blacks may also have low variance), the cumulative distortion is significant.

**Recommended companion fix:** In addition to tie-aware ranking, add discrimination gating: exclude metrics with near-zero variance (e.g., CV < 1%) from the composite. Report them as "non-discriminating" rather than silently including them.

### Z-Score Computation
Z-scores are correctly computed: `(value − mean) / std`, sign-flipped for lower-is-better metrics, distance-based for blocking (closer to 1.0). Used for heatmap coloring only (not for the composite score). **Correct.**

---

## 14. Dropped Metrics Still in Code

The following functions remain in `quality_report.py` but are not called:
- `noise_estimate()` (lines 158–190) — dropped, r=0.83 with detail
- `perceptual_contrast()` (lines 260–262) — dropped, near-zero discrimination
- `tonal_richness()` (lines 306–311) — dropped, near-zero discrimination
- `gradient_smoothness()` (lines 328–346) — dropped, r=0.94 with texture_quality

These are dead code. They don't affect correctness or performance (never called), but they add ~80 lines of maintenance burden. Consider removing them or moving them to a separate archive module.

---

## Summary of Findings

### Critical
None.

### High
| # | Issue | Location | Description |
|---|-------|----------|-------------|
| H1 | Additive pedestal sensitivity | sharpness, edge_strength, detail, ringing, temporal | Mean-Y normalization cancels multiplicative gain but not additive offsets. +0.05 pedestal causes 12–60% metric change. |
| H2 | Tied-rank bug + blown_whites | `compute_composites()` L599–612 | `np.argsort` assigns arbitrary 1–16 ranks for blown_whites (constant zero), injecting pure noise. Use `scipy.stats.rankdata(method='average')` + discrimination gating. |
| H3 | No-frame edge case | `analyze_clip()` L393–429 | If skip_frames >= total frames, accumulators are empty → NaN propagation in metrics, ranks, JSON. |

### Major
| # | Issue | Location | Description |
|---|-------|----------|-------------|
| M1 | Ringing brightness dependence | `ringing_measure()` L245 | 8-bit Canny quantization creates brightness-dependent edge mask, causing score swings under brightness changes. |
| M2 | Naturalness direction | `naturalness_nss()` L325 | `higher_better=True` can reward pathological spike distributions (kurtosis ~69) over natural scenes (~10). |
| M3 | No ffmpeg error checking | `analyze_clip()` L386/422 | `proc.returncode` never checked; corrupt files produce partial results silently. |
| M4 | No process cleanup on exception | `analyze_clip()` L386 | Missing try/finally means subprocess leaks on Python exceptions. |

### Minor
| # | Issue | Location | Description |
|---|-------|----------|-------------|
| m1 | Colorfulness channel mapping | `colorfulness_metric()` L269 | Uses (Cr−Cb, 0.5(Cr+Cb)) instead of true Hasler (R−G, 0.5(R+G)−B). Rankings preserved but absolute values differ. |
| m2 | Noise confound on sharpness metrics | `sharpness_laplacian()`, `edge_strength_sobel()` | Additive noise inflates these metrics. Mitigated by texture_quality in the composite. |
| m3 | Dead code | Lines 158–346 | Four dropped metric functions remain in the file (~80 lines). |
| m4 | Narrow ringing detection zone | `ringing_measure()` L248 | 5×5 dilation kernel may miss wider analog ringing oscillations (5–10 pixel span). |

### Notes (informational, no fix needed)
| # | Observation |
|---|-------------|
| n1 | Blocking metric likely has near-zero discrimination on ProRes content — serves as safety check. |
| n2 | Naturalness (MSCN kurtosis) may have limited discrimination — GGD beta was already found non-discriminative. |
| n3 | Temporal stability confounds scene motion with capture instability — acceptable for same-content comparison. |
| n4 | crushed_blacks/blown_whites use absolute thresholds by design — they ARE sensitive to brightness differences, intentionally. |
| n5 | Performance: Python loops in detail_texture and texture_quality_measure are vectorizable for significant speedup. |
| n6 | The naturalness median aggregation (line 428) is a good design choice for an outlier-sensitive 4th-power statistic. |
| n7 | Temporal stability is also pedestal-sensitive: additive offset changes the denominator, making higher-pedestal clips appear more stable. |
