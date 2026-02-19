# Metric Selection Review

Review of the 11 no-reference metrics used by `quality_report.py` for comparing analog video capture pipelines (VCRs, TBCs, capture cards) digitizing standard-definition composite/S-Video sources.

---

## 1. Coverage Analysis

### 1.1 Sharpness (Laplacian variance / mean Y^2)

**Quality dimension:** Spatial frequency energy -- how much high-frequency detail the capture chain preserves. Laplacian variance is a classic focus/sharpness measure (Pertuz et al., 2013, "Analysis of focus measure operators for shape-from-focus").

**Relevance to analog capture:** High. Different VCR heads, TBC sharpness circuits, and capture card bandwidth filters produce measurably different edge transition slopes. This is arguably the single most important differentiator among capture devices for SD analog video.

**Alternatives:** The Tenengrad measure (Sobel gradient variance) is comparable; Brenner gradient and wavelet-ratio sharpness (Vu & Chandler, 2012, "S3: A spectral and spatial measure of local perceived sharpness") offer finer discrimination. The current Laplacian variance approach is standard and well-suited to this application. The mean Y^2 normalization for brightness invariance is appropriate.

**Assessment:** Sound choice. Well-established, relevant, properly normalized.

### 1.2 Edge Strength (mean Sobel gradient / mean Y)

**Quality dimension:** Average gradient magnitude across the frame -- a measure of overall edge definition that complements sharpness. Where sharpness (Laplacian variance) emphasizes variance of second-derivative response (sensitive to crisp edges and fine texture), edge strength measures the first-derivative mean (more sensitive to overall contrast at transitions).

**Relevance to analog capture:** Moderate-high. Captures the same spatial bandwidth dimension as sharpness but from a different mathematical angle. Devices with aggressive coring (noise reduction that zeros out small gradients) will show reduced edge strength relative to sharpness.

**Alternatives:** Could be replaced or supplemented by a multi-scale edge measure (e.g., Canny edge density at multiple thresholds) for better discrimination of soft vs. hard rolloff characteristics.

**Assessment:** Reasonable, but likely provides moderate incremental value over sharpness. The correlation between these two should be checked empirically; if r > 0.9 across the dataset, one could be dropped. The fact that ringing was retained at r=0.94 with sharpness on domain-specific grounds makes keeping edge_strength at a lower correlation more defensible, but worth monitoring.

### 1.3 Blocking (8x8 DCT boundary ratio)

**Quality dimension:** Presence of 8x8 block boundary artifacts from DCT-based compression. The ratio of inter-pixel differences at block boundaries vs. non-boundary positions is a well-known blocking metric (Wang et al., 2000, "Blind measurement of blocking artifacts in images").

**Relevance to analog capture:** Low-moderate for ProRes 422 HQ. ProRes uses DCT but at extremely high bitrates where blocking is perceptually absent. This metric would be relevant if any capture pipeline used MPEG-2, DV, or MJPEG compression, or if comparing lossy transcodes. For ProRes-to-ProRes comparison, all clips should score near 1.0 with negligible discrimination.

**Alternatives:** Liu et al. (2010) "No-reference image quality assessment based on spatial and spectral entropies" proposed a spectral approach to blocking detection. The JPEG-AI blocking measure uses learned features. For this application, the simple boundary-ratio approach is appropriate if blocking artifacts are actually present.

**Assessment:** May have near-zero discrimination across ProRes clips (similar to the dropped contrast/tonal_richness metrics). Worth checking the coefficient of variation across clips. If all values cluster near 1.0 with CV < 5%, this metric contributes noise rather than signal to the composite. However, it serves as a useful safety check if the pipeline ever encounters DV or MPEG-2 intermediate files, so retaining it is defensible for robustness even if it rarely discriminates.

### 1.4 Detail (local coefficient of variation median, 16x16 blocks)

**Quality dimension:** Fine texture preservation. The local CV (std/mean per block) measures how much micro-contrast structure is retained. The median across blocks provides robustness to outlier regions.

**Relevance to analog capture:** High. Capture bandwidth limitations, noise reduction (coring, 3D NR), and analog filter characteristics all affect fine detail. Different TBCs and VCRs have notably different coring aggressiveness, which directly impacts this metric.

**Alternatives:** GLCM (Gray-Level Co-occurrence Matrix) texture features, LBP (Local Binary Pattern) histograms, or wavelet subband energy ratios could provide more nuanced texture characterization. However, the local CV approach is computationally cheap, interpretable, and has demonstrated empirical discrimination in this project.

**Assessment:** Good choice. Well-suited to the application, and the inherent brightness invariance (std/mean) matches the project's design philosophy.

### 1.5 Texture Quality (smoothed/original variance ratio)

**Quality dimension:** Ratio of structured detail to total detail. By comparing variance before and after mild Gaussian smoothing, this separates structured texture (which survives smoothing) from unstructured noise (which is attenuated). Values near 1.0 indicate clean, well-structured detail; lower values indicate noise contamination.

**Relevance to analog capture:** High. Analog video always carries some noise floor (tape noise, head preamp noise, TBC jitter noise). Devices with better SNR or more effective noise reduction will show higher texture quality ratios. This metric effectively captures the "clean detail vs. noisy detail" axis that the dropped noise metric also targeted but from a different angle.

**Alternatives:** The signal-to-noise-plus-distortion ratio from ITU-R BT.1683, or wavelet domain denoising residual measures, could provide similar information. The current approach is elegant and interpretable.

**Assessment:** Excellent choice. Fills an important niche between the dropped noise metric and the detail metric by measuring noise impact on texture structure rather than noise level directly.

### 1.6 Ringing (near-edge Laplacian energy / mean Y)

**Quality dimension:** Edge overshoot / Gibbs phenomenon. Measures high-frequency energy in a narrow annular region around detected edges, normalized by brightness.

**Relevance to analog capture:** Very high. This is one of the most domain-specific and valuable metrics in the suite. VCR sharpness enhancement circuits (detail enhancement, aperture correction) deliberately introduce edge overshoot to create a perceptual sharpness boost. Different VCRs and TBCs apply different amounts. The CLAUDE.md note that ringing was retained despite r=0.94 correlation with sharpness is well-justified -- the correlation is driven by the fact that sharper capture chains tend to also apply more enhancement, but the ringing artifact itself is a distinct quality dimension that some users specifically want to minimize.

**Alternatives:** More sophisticated ringing detectors exist in the MPEG quality assessment literature (e.g., Marziliano et al., 2004, "Perceptual blur and ringing metrics" which separates ringing from blur in the edge profile). The current Canny+dilate approach is simple but effective for the overshoot patterns typical of analog sharpness circuits.

**Assessment:** Excellent domain-specific choice. The decision to retain despite high sharpness correlation is well-reasoned and reflects genuine domain knowledge about analog video processing.

### 1.7 Temporal Stability (mean frame-to-frame difference / mean Y)

**Quality dimension:** Frame-to-frame luminance consistency. Higher values indicate more flicker, noise temporal variation, or instability. The brightness normalization makes this agnostic to static brightness differences.

**Relevance to analog capture:** High. Temporal instability manifests as:
- Tape noise temporal variation (different noise patterns frame-to-frame)
- TBC jitter residual (sub-pixel position shifts between frames)
- Head-switching transients (may cause periodic brightness pulses)
- AGC hunting (slow brightness fluctuation from automatic gain control)

**Alternatives:** More sophisticated temporal metrics exist:
- Temporal Information (TI) from ITU-T P.910 is essentially the same computation (std of frame differences)
- Flicker metrics from VQEG that separate temporal noise from motion
- Optical flow consistency measures that distinguish camera motion from capture instability
The current implementation does not separate true scene motion from capture artifacts, which means content-dependent sequences will inflate this metric. For controlled comparisons of the same source material across devices, this is acceptable since the motion component cancels in the differential comparison.

**Assessment:** Adequate for same-source comparison. The metric would be unreliable for comparing clips of different content. For same-content A/B comparison (which is this project's use case), it works well.

### 1.8 Colorfulness (Hasler-Susstrunk metric on Cb/Cr)

**Quality dimension:** Chroma saturation and variety. The Hasler & Susstrunk (2003) colorfulness metric ("Measuring Colorfulness in Natural Images") combines the standard deviation and mean of opponent-color channels into a single scalar. Higher values indicate more vivid, saturated color reproduction.

**Relevance to analog capture:** Moderate-high. Different capture chains handle chroma bandwidth differently. S-Video separates luma and chroma, reducing cross-contamination. Composite video requires comb filtering, which can reduce chroma bandwidth. TBC chroma processing varies. However, the upstream normalization step (`normalize.py`) already matches chroma saturation across clips, which may reduce this metric's discrimination post-normalization.

**Alternatives:** CIEDE2000 color difference, CIE chroma (C*), or the more recent NIQE-extended color naturalness metrics could provide complementary information. The Hasler-Susstrunk metric is well-established and widely used.

**Assessment:** Sound choice, but its discrimination may be attenuated if clips are saturation-normalized before measurement. The docstring says brightness-sensitive metrics are normalized by mean Y, but it's unclear whether the chroma normalization from `normalize.py` is expected as a prerequisite. UPDATE: The README states normalization is optional ("Not required for metrics"), and colorfulness operates on raw Cb/Cr planes, so it measures actual chroma saturation differences between capture chains. This is appropriate.

**Caveat — fidelity vs. vibrancy:** Colorfulness measures chroma *spread and magnitude*, not chroma *fidelity*. A capture chain with boosted saturation or elevated chroma noise will score higher, even if its color reproduction is less accurate. This means the metric can reward chroma noise or deliberate over-saturation. Pairing with a chroma noise or hue-stability metric would address this gap (see §2.3 below).

### 1.9 Naturalness (MSCN kurtosis)

**Quality dimension:** Natural Scene Statistics (NSS) regularity. The Mean Subtracted Contrast Normalized (MSCN) coefficient distribution of natural images follows a generalized Gaussian with characteristic kurtosis. Departure from this (lower kurtosis, more Gaussian-like) indicates processing artifacts, noise contamination, or unnatural signal characteristics.

**Relevance to analog capture:** Moderate. In the IQA literature, MSCN statistics are a cornerstone of BRISQUE (Mittal et al., 2012) and NIQE (Mittal et al., 2013). However, their effectiveness assumes "natural image" statistics, which analog video content may not always exhibit (test patterns, synthetic graphics, etc.). The CLAUDE.md notes that the GGD beta fitting gave beta approximately 2.0 for all ProRes clips (no discrimination), motivating the switch to kurtosis. This suggests the MSCN distribution is already near-Gaussian for this content type (ProRes preserves statistics well), and kurtosis is being used as a more sensitive probe of the same phenomenon.

**Alternatives:** Full BRISQUE requires the `cv2.quality` module (unavailable in OpenCV headless). NIQE could be implemented from scratch using the MVG model. MUSIQ (Ke et al., 2021) is a transformer-based NR-IQA metric that could run via ONNX but would require a model file. The simple kurtosis approach is a reasonable compromise given the constraints.

**Assessment:** Reasonable given the OpenCV constraint, but may have limited discrimination for this content type. If empirical CV across clips is low, consider whether this metric adds meaningful signal. The use of median aggregation across frames (noted in `analyze_clip`) is appropriate given kurtosis's sensitivity to outliers.

**Caveat — monotonic direction is weakly justified.** The metric is scored as `higher_better=True` (higher MSCN kurtosis = more natural). While this follows the IQA literature (natural images are leptokurtic), empirical testing shows that certain artifact patterns can produce *extremely* high kurtosis: a mostly-flat image with rare sharp spikes yields kurtosis ~69, far exceeding a realistic natural scene (~10). The monotonic "higher is better" assumption could reward pathological distributions. A distance-from-target approach (score distance from a calibrated "natural" kurtosis range) would be more robust, though it requires establishing the target range empirically for this content type.

### 1.10 Crushed Blacks (shadow headroom ratio)

**Quality dimension:** Shadow clipping -- what fraction of shadow-region pixels are crushed to near-black. Measures the capture chain's ability to preserve dark detail below approximately IRE 15.

**Relevance to analog capture:** High. Black level setup, analog clipping, and ADC headroom vary significantly across capture hardware. Some VCRs clip blacks aggressively (especially if setup/pedestal is miscalibrated). TBCs with poor black level tracking will crush shadow detail. This is a genuine quality differentiator.

**Alternatives:** Histogram-based measures of shadow detail (e.g., entropy of the lower 15% of the histogram), or perceptual dark detail visibility metrics from HDR literature. The current ratio approach is simple and interpretable.

**Assessment:** Good domain-specific metric. The thresholds (CRUSH_FLOOR=0.07, SHADOW_CEIL=0.15 in 10-bit normalized range) are appropriate for legal-range video (64-940 in 10-bit). This metric paired with blown_whites provides a useful signal headroom characterization.

### 1.11 Blown Whites (highlight headroom ratio)

**Quality dimension:** Highlight clipping -- fraction of highlight pixels blown to near-white. Measures the capture chain's ability to preserve bright detail near peak white.

**Relevance to analog capture:** High. Similar to crushed blacks but for the upper end. Capture cards with AGC or fixed gain that clips above 100 IRE will show higher blown_whites. Particularly relevant for hot signals (test patterns, bright graphics) where some capture chains clip while others preserve super-white.

**Alternatives:** Same as crushed_blacks -- histogram entropy of upper region, or HDR-derived highlight detail metrics.

**Assessment:** Good complementary metric to crushed_blacks. Together they characterize the signal headroom/dynamic range preservation of the capture chain.

**Empirical finding — zero discrimination on current dataset.** On the 16-clip noref_compare dataset, blown_whites is exactly 0.0 for all clips (std = 0.0). No clip in the test set has highlight clipping. The metric has zero discriminative power and contributes only tied-rank noise to the composite (see §4.1 tied-rank discussion). Consider either (a) excluding metrics with zero variance from the composite scoring, or (b) retaining as a safety check that only enters the composite when its variance exceeds a threshold.

---

## 2. Coverage Gaps

### 2.1 Analog-Specific Artifacts (HIGH priority)

#### 2.1.1 Dropout Detection

**What it is:** Momentary signal loss causing bright white or dark horizontal streaks/lines, typically lasting a fraction of a scan line to several lines. Caused by tape oxide damage, head clogs, or tape-head separation.

**Why it matters:** Dropout is the most visible and objectionable artifact in analog video playback. A single dropout per minute may be acceptable; hundreds per minute indicate severe tape degradation or poor head contact. Devices with dropout compensation (DOC) circuits actively conceal these, and their effectiveness varies dramatically.

**How to detect:** Per-frame horizontal line analysis: scan each row for abrupt luminance jumps exceeding a threshold (e.g., |delta_Y| > 0.3 sustained for > 20 pixels). Count occurrences per frame, report rate per minute. Could also use the temporal stability per-frame data to flag frames with localized extreme differences.

**Difficulty:** Moderate. Requires distinguishing actual picture content (bright horizontal elements) from dropout artifacts. The key discriminator is that dropouts are typically full-brightness (white) or full-black and span a significant horizontal extent without matching surrounding frame content.

#### 2.1.2 Head-Switching Noise

**What it is:** Horizontal disturbance band at the bottom 5-15 lines of each field, caused by the transition between video heads during playback. Appears as a horizontal shift, brightness jump, or noise burst.

**Why it matters:** All VHS/Beta decks exhibit head-switching noise, but TBCs handle it differently. Some TBCs clean it up completely; others leave residual artifacts. The location is predictable (bottom of frame), making detection straightforward.

**How to detect:** Compare the bottom N rows (e.g., bottom 16 lines) against the frame average for temporal instability, horizontal shift (phase correlation), or gradient anomaly. A per-region temporal stability metric would catch this.

**Difficulty:** Low. Fixed spatial location makes this straightforward. Could be a simple extension of the temporal stability metric restricted to the bottom region.

#### 2.1.3 Chroma Delay / Bleed

**What it is:** Horizontal misregistration between luma and chroma, causing color to appear shifted left or right relative to edges. Caused by analog filter group delay mismatch.

**Why it matters:** Chroma delay is a common analog artifact that varies between devices. Some TBCs correct for it; others do not. It directly impacts color edge quality.

**How to detect:** Cross-correlation between Y edge locations and Cb/Cr edge locations. The lag at peak correlation indicates chroma delay in pixels. Alternatively, measure the spatial offset between luma and chroma gradients at vertical edges.

**Difficulty:** Moderate. The implementation exists in broadcast measurement equipment (chroma-luma delay in waveform monitors). Requires analyzing vertical edges with simultaneous chroma transitions.

#### 2.1.4 Cross-Luminance (Dot Crawl) and Cross-Chrominance (Rainbow)

**What it is:** Artifacts from imperfect separation of luma and chroma in composite video signals. Dot crawl appears as a crawling dot pattern along horizontal color transitions; rainbow patterns appear as false color on fine luminance detail.

**Why it matters:** These artifacts are the primary motivation for S-Video vs. composite capture, and for 3D comb filters vs. 2D. Devices with better comb filtering will show less cross-luminance/chrominance.

**How to detect:** Spectral analysis near the colorburst frequency (3.58 MHz for NTSC, 4.43 MHz for PAL) in the luminance channel. Cross-luminance manifests as a specific spatial frequency pattern. Could measure energy at the chroma subcarrier frequency in the Y channel. For cross-chrominance, look for false chroma at fine luminance transitions.

**Difficulty:** High. Requires knowledge of the color encoding standard and spatial frequency analysis at the subcarrier frequency. This is probably the most technically demanding metric to implement correctly, but also one of the most diagnostically valuable for composite video sources.

**Note:** If all clips are captured via S-Video, cross-luminance/chrominance should be absent and this metric would not discriminate. Its relevance depends on whether any composite captures are in the comparison set.

### 2.2 Temporal Artifacts (MEDIUM priority)

#### 2.2.1 Time-Base Stability (Jitter)

**What it is:** Sub-pixel horizontal position instability between frames, caused by tape speed variation. Manifests as a slight horizontal shimmer or wobble, especially visible on vertical edges.

**Why it matters:** TBCs exist specifically to correct time-base errors. Their effectiveness varies, and residual jitter is a key quality differentiator.

**How to detect:** Phase correlation or optical flow between consecutive frames, measuring the horizontal translation component. The variance of horizontal displacement over time indicates jitter magnitude. This is distinct from the current temporal stability metric, which measures absolute luminance differences (conflating jitter with noise and actual motion).

**Difficulty:** Moderate. Phase correlation is well-understood. The challenge is separating jitter (horizontal-only, typically < 1 pixel after TBC) from actual scene motion.

#### 2.2.2 Field Order and Interlacing Artifacts

**What it is:** Incorrect field dominance or field blending artifacts. If the capture or playback chain gets field order wrong, interlaced content shows combing artifacts or reduced vertical resolution.

**Why it matters:** Incorrect deinterlacing or field order errors can halve vertical resolution. Some capture pipelines may apply deinterlacing while others capture raw fields.

**How to detect:** Analyze inter-field correlation vs. inter-frame correlation. Combing artifacts show up as horizontal line-alternating patterns detectable by checking even/odd line correlation.

**Difficulty:** Low-moderate. Well-understood problem. However, the current dataset appears to be pre-processed (IVTC/NOIVTC suffixes in filenames suggest inverse telecine has already been applied), so this may not be relevant for the current workflow.

### 2.3 Signal Integrity (MEDIUM priority)

#### 2.3.1 Dynamic Range Utilization

**What it is:** How effectively the capture chain uses the available digital code range. A capture that maps the full analog signal into a narrow digital range wastes bit depth.

**Why it matters:** Beyond crushed_blacks and blown_whites (which measure clipping at the extremes), the overall histogram spread and its match to the expected signal levels indicates ADC calibration quality.

**How to detect:** Histogram analysis -- effective bit depth (number of unique levels used), histogram spread (percentile range), histogram entropy. Some of this was covered by the dropped tonal_richness metric, but could be reconsidered as a distinct "dynamic range utilization" measure.

**Difficulty:** Low. Simple histogram analysis.

**Note:** The dropped tonal_richness (histogram entropy) and contrast (RMS) metrics covered parts of this. Their near-zero discrimination suggests that all capture chains in the test set produce similar dynamic range utilization, which may mean this is not a differentiator for this specific hardware comparison.

#### 2.3.2 Gamma / Transfer Function Accuracy

**What it is:** Whether the capture chain reproduces the correct tonal curve (gamma 2.2 or BT.1886 for display). An incorrect gamma produces either washed-out or overly contrasty images.

**Why it matters:** Different capture cards may apply different gamma curves. This affects tonal reproduction of the entire image.

**How to detect:** Requires a reference (known-input signal like SMPTE bars or a stepped grayscale) to measure the transfer function. Not feasible as a no-reference metric on arbitrary content.

**Difficulty:** Not applicable as NR metric. Would require full-reference comparison or known test pattern input.

### 2.4 Spatial Artifacts (LOW priority)

#### 2.4.1 Geometric Distortion

**What it is:** Non-linear spatial warping -- barrel/pincushion distortion, skew, trapezoidal distortion. Caused by playback timing errors or capture card sync issues.

**Why it matters:** Mild distortion may not be visible but accumulates in multi-generation copies. Severe distortion (skew, flagging) indicates TBC failure.

**How to detect:** Requires either a reference pattern (registration chart) or sophisticated no-reference approaches (detecting straight lines and measuring curvature). Optical flow between frames could detect time-varying skew.

**Difficulty:** High for NR detection on arbitrary content. Low if using test patterns.

#### 2.4.2 Overscan Handling

**What it is:** How much of the analog signal's active picture area is captured. Some capture configurations crop more aggressively than others.

**Why it matters:** Cropping reduces the captured picture area. For archival purposes, capturing as much of the active picture as possible is preferred.

**How to detect:** Analyze frame borders for static black/blank regions. Measure the active picture area as a fraction of total frame area.

**Difficulty:** Low. Simple border analysis. However, this may be a configuration issue rather than a quality metric (overscan is typically user-adjustable in capture software).

### 2.5 Perceptual NR-IQA Metrics (MEDIUM-LOW priority)

#### 2.5.1 BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

Mittal et al., 2012. Uses MSCN coefficient statistics (mean, variance, shape, and paired-product statistics) with an SVR model trained on LIVE IQA database. The current naturalness metric extracts one feature (kurtosis) from the BRISQUE feature space. Full BRISQUE uses 36 features.

**Status:** Unavailable -- requires `cv2.quality` module not in OpenCV headless. Could be implemented from scratch (the feature extraction is straightforward; the SVR model weights are published), but the training data (LIVE database) contains camera-captured natural images, not analog video. Domain mismatch may reduce effectiveness.

#### 2.5.2 NIQE (Naturalness Image Quality Evaluator)

Mittal et al., 2013. "Making a 'completely blind' image quality analyzer." Uses an MVG (multivariate Gaussian) model of NSS features from pristine natural images. No training on distorted images needed -- quality is measured as the distance from the pristine model.

**Status:** Could be implemented from scratch. The MVG model fitting requires a corpus of "pristine" images. For analog video, what constitutes "pristine" is debatable -- a high-quality S-Video capture with good TBC might be the reference standard, but this circles back to a form of reference-based comparison.

**Assessment:** NIQE is more promising than BRISQUE for this domain because it does not require distortion-specific training. However, the "pristine" model would need to be calibrated for SD analog video content rather than using the standard photographic image model.

#### 2.5.3 MUSIQ / CLIP-IQA / TOPIQ

Recent transformer-based NR-IQA metrics that leverage pre-trained vision models. These are state-of-the-art on standard IQA benchmarks but require GPU inference and large model files.

**Assessment:** Overkill for this application. The computational cost and domain mismatch (trained on camera/compression artifacts, not analog video) make these a poor fit. The simple physics-based metrics in the current suite are more interpretable and more directly relevant to the analog domain.

---

## 3. Metric Redundancy

### 3.1 Assessment of Dropped Metrics

#### Noise (r=0.83 with detail) -- CORRECTLY DROPPED

The noise metric (flat-region high-frequency energy) is conceptually distinct from detail (local coefficient of variation), but at r=0.83 across the clip set, more than 69% of the variance is shared. In a rank-based composite with only 11 metrics, keeping both would effectively double-weight this dimension. The texture_quality metric (smoothed/original variance ratio) captures the noise dimension from a different angle -- "how much of the detail is structured vs. noise" -- which is arguably more informative than the raw noise level. Dropping noise was the right call.

#### Contrast / Tonal Richness (near-zero discrimination) -- CORRECTLY DROPPED

If all capture chains produce similar RMS contrast and histogram entropy (which they should, if they all have similar gain/level settings and are capturing the same source), these metrics add no discriminative information and merely dilute the composite score with noise. This is analogous to a feature with near-zero variance in a classification problem. Correct decision.

#### Gradient Smoothness (r=0.94 with texture quality) -- CORRECTLY DROPPED

At r=0.94, these metrics are effectively measuring the same underlying dimension (quality of tonal gradients vs. noise contamination). Keeping both would heavily double-weight this dimension relative to others. Unlike the ringing/sharpness case, there is no domain-specific argument for why gradient smoothness captures a distinct analog artifact that texture quality misses.

#### Ringing retained despite r=0.94 with sharpness -- CORRECTLY RETAINED

This is the most interesting decision and it is well-justified. The high correlation likely arises because capture chains that preserve more bandwidth (higher sharpness) also tend to pass through more of the sharpness enhancement circuitry (higher ringing). But these are causally distinct: sharpness measures bandwidth preservation while ringing measures artificial edge enhancement. A hypothetical capture chain with wide bandwidth but no sharpness circuit would score high on sharpness and low on ringing. The fact that no such device exists in the current test set (hence the high correlation) does not mean the dimensions are redundant -- it means the test set has limited diversity on this axis. Retaining ringing is the right domain-informed decision.

### 3.2 Potential Redundancy Among Current 11 Metrics

#### Sharpness vs. Edge Strength

These are both first/second derivative measures of spatial frequency content. Sharpness (Laplacian variance) emphasizes high-frequency energy distribution; edge strength (Sobel mean) emphasizes average gradient magnitude.

**Empirical result:** r = 0.768 on the 16-clip noref_compare dataset. This is below the 0.90 redundancy threshold, indicating that the two metrics capture meaningfully different aspects of spatial frequency content. Retention of both is justified.

However, note that edge_strength correlates much more strongly with detail (r = 0.921) — these two share >85% of variance. Since detail is conceptually distinct from edge_strength (local CV vs. mean gradient), this high correlation is likely coincidental for this dataset. Both should be monitored.

**Recommendation:** Retain both sharpness and edge_strength. The r = 0.768 correlation leaves substantial unique variance in each.

#### Detail vs. Texture Quality

Detail measures how much local variation exists (CV); texture quality measures how much of that variation is structured. These are conceptually complementary — a noisy capture could have high detail (lots of variation) but low texture quality (variation is unstructured).

**Empirical result:** r = -0.553 on the 16-clip dataset. These are *negatively* correlated — noisier captures show higher detail (more local variation) but lower texture quality (more of that variation is unstructured noise). This confirms the complementary design: detail measures quantity of high-frequency content while texture quality measures its quality. Both should be retained.

**Recommendation:** Retain both. The negative correlation validates the conceptual distinction.

#### Crushed Blacks vs. Blown Whites

These measure complementary ends of the dynamic range and should have near-zero correlation (shadow clipping is mechanistically independent of highlight clipping). They should both be retained.

**Recommendation:** Verify near-zero correlation. Both should be kept.

#### Naturalness vs. Other Metrics

MSCN kurtosis may correlate with several other metrics (especially texture quality and detail) because natural image statistics are influenced by the same signal characteristics these metrics measure. If naturalness has low unique variance after accounting for the other metrics, it may not add meaningful information.

**Recommendation:** Check partial correlations -- does naturalness carry information not captured by any other metric? If its unique contribution is small, consider dropping it.

---

## 4. Composite Scoring

### 4.1 Rank-Based Averaging

The current approach ranks clips 1 to N per metric and averages the ranks. This is essentially the Borda count from social choice theory.

**Pros:**
- **Robust to outliers and scale differences.** A single extreme metric value cannot dominate the composite because it is compressed to a rank (at most a difference of N-1 rank positions). This is particularly important for metrics with very different scales (e.g., colorfulness in the hundreds vs. blocking near 1.0).
- **No normalization needed.** Unlike z-score approaches, ranking avoids the question of whether the metric distribution is normal, whether the variance is meaningful, etc.
- **Interpretable.** "Average rank 3.2 out of 10" is immediately understandable.
- **Monotonic.** Improving on any single metric cannot worsen the composite score (rank can only decrease or stay the same).

**Cons:**
- **Discards magnitude information.** The difference between rank 1 and rank 2 might be enormous (one device is vastly better) or negligible (statistical noise), but both count the same. A clip that is marginally worst on one metric loses a full rank position.
- **Sensitive to field size.** Adding or removing clips changes all ranks, potentially changing the composite ordering in non-obvious ways. With a small field (10-14 clips), a single new entrant can shift ranks by +/-1 on multiple metrics.
- **Tied ranks handled poorly.** With small field sizes, ties are likely. The code uses `np.argsort` which breaks ties by array order (i.e., alphabetical clip name) rather than assigning averaged ranks. This is a **significant implementation issue** — not merely theoretical. On the 16-clip dataset, `blown_whites` is constant zero for all clips, meaning argsort assigns arbitrary 1..16 ranks for identical values. This metric contributes pure bias to the composite. Combined with any other metrics that have tied values, the cumulative effect can shift overall rankings by 0.5+ positions.
- **Equal weighting assumption.** All 11 metrics receive equal weight (1/11 of the composite). This means crushed_blacks (which may be zero for all clips) has the same influence as sharpness (which may have large meaningful spread).

### 4.2 Alternatives

#### Z-Score Weighted Composite

Normalize each metric to z-scores and compute a weighted sum. The project notes that CV-weighted z-scores were tried but failed because naturalness's near-zero mean inflated its CV to 3.3x weight.

**Assessment:** The CV-weighting failure is a real issue, but the solution is not to abandon z-scores entirely -- rather, use a fixed equal-weight z-score sum or a properly regularized weighting scheme. Z-score composites preserve magnitude information that rank-based composites discard.

**Recommendation:** An equal-weight z-score composite would be a useful complement (not replacement) to the rank composite. Report both.

#### Expert Weighting

Assign weights based on domain knowledge (e.g., sharpness 2x, ringing 1.5x, blocking 0.5x). This allows emphasizing metrics that matter more for the use case.

**Assessment:** Defensible but subjective. Different users have different priorities (archivists may prioritize temporal stability and signal headroom; editors may prioritize sharpness and colorfulness). The equal-weight approach has the virtue of objectivity.

**Recommendation:** If expert weighting is desired, present it as a user-configurable option rather than a default. The current equal-weight default is appropriate for a general-purpose tool.

#### PCA-Based Composite

Extract the first principal component from the metric correlation matrix and use its loadings as weights.

**Assessment:** PCA maximizes explained variance, which means it implicitly up-weights metrics with high inter-clip spread and down-weights metrics with low spread. This has some appeal (it automatically handles the "blocking has no discrimination" problem) but produces weights that are opaque and content-dependent. The first PC may not correspond to "quality" -- it may correspond to whatever axis of hardware variation happens to dominate the dataset.

**Recommendation:** PCA is useful as a diagnostic tool (understanding the dimensionality of the metric space) but not recommended as the primary composite scoring method.

### 4.3 Equal Weighting Assessment

Equal weighting across 11 metrics is defensible but imperfect:

- **Sharpness, detail, and texture quality** collectively receive 3/11 (27%) of the composite weight, all measuring related spatial quality dimensions. If these are highly correlated, this effectively over-weights spatial quality.
- **Crushed blacks and blown whites** receive 2/11 (18%) for signal headroom, which may be disproportionate if both are near-zero for all clips.
- **Temporal stability** receives 1/11 (9%), which may under-weight it for use cases where flicker/jitter is the primary concern.

The rank-based approach partially mitigates the correlated-metrics problem because even if three metrics agree on the ranking, each only contributes 1/11 of the composite. However, a clip that ranks poorly on sharpness is likely to also rank poorly on edge_strength and detail, receiving three "bad rank" penalties for what may be a single underlying deficiency (limited capture bandwidth).

**Recommendation:** Consider grouping correlated metrics into dimensions and applying equal weight per dimension rather than per metric. For example:
- Spatial quality (sharpness, edge_strength, detail): 1/6 weight total
- Signal integrity (texture_quality, crushed_blacks, blown_whites): 1/6 weight total
- Analog artifacts (ringing, blocking): 1/6 weight total
- Temporal (temporal_stability): 1/6 weight total
- Color (colorfulness): 1/6 weight total
- Naturalness (naturalness): 1/6 weight total

This would give each quality dimension equal representation regardless of how many metrics probe it.

### 4.4 Non-Discriminating Metrics in Composite

On the 16-clip dataset, `blown_whites` has exactly zero variance. With the current tied-rank bug (§4.1), this means argsort assigns arbitrary 1..N ranks for identical values, injecting pure noise into the composite. Even with tie-aware ranking (all clips get rank 8.5), a zero-variance metric contributes nothing to discrimination while diluting the weight of informative metrics.

**Recommendation:** Add a discrimination gate: exclude metrics with coefficient of variation below a threshold (e.g., CV < 1%) from the composite scoring. Report the excluded metrics with an annotation ("non-discriminating for this dataset") so they remain visible for diagnostic purposes but don't affect the overall ranking.

---

## 5. Brightness-Agnostic Claim

The project documentation states "all metrics are brightness-agnostic — no upstream brightness normalization is required." This claim is accurate for **multiplicative gain** differences (where one clip is uniformly brighter by a constant factor) but **not for additive pedestal offsets** (where a constant is added to all pixel values).

Empirical testing confirms:
- Multiplicative scaling (0.3× to 1.0×): sharpness, edge_strength, detail are perfectly invariant.
- Additive offset (+0.05, typical for pedestal differences): sharpness changes ~23%, edge_strength ~12%, detail ~60%.

Additive pedestal offsets are common in analog capture (different setup/pedestal calibration across devices). The current mean-Y normalization correctly cancels multiplicative gain but not additive offsets, because Laplacian/Sobel responses are offset-invariant while the normalizing denominator (mean_Y) is not.

**Recommendation:** Reframe the documentation claim to "gain-invariant for metrics normalized by mean Y" and document the known sensitivity to additive pedestal offsets. For users comparing devices with known pedestal differences, recommend running `normalize_linear.py` first, or consider switching to local-contrast-based normalization that rejects DC shifts.

---

## 6. Recommendations

### HIGH IMPACT

**H1. Add dropout detection.** This is the most important missing metric for analog video quality assessment. Dropouts are the single most objectionable artifact and the primary reason to prefer one capture chain over another. A per-frame horizontal line analyzer counting brightness-anomaly events would be straightforward to implement and would add a genuinely new quality dimension. Priority: implement before any other additions.

**H2. Add head-switching region analysis.** Measure temporal instability and horizontal phase error specifically in the bottom 16 lines of each frame (or each field if processing interlaced). This is a low-hanging-fruit extension of the existing temporal stability metric and would capture a major VCR/TBC quality differentiator. Could be reported as a separate metric or as a per-region breakdown of temporal stability.

**H3. Check and report the empirical inter-metric correlation matrix — DONE.** The full correlation matrix has been computed on the 16-clip dataset. Key findings:
- edge_strength vs. detail: r=0.921 (high redundancy)
- sharpness vs. ringing: r=0.901 (expected, justified retention)
- sharpness vs. edge_strength: r=0.768 (moderate, both worth keeping)
- detail vs. texture_quality: r=-0.553 (complementary, validates design)
- blown_whites: zero variance (non-discriminating)
- temporal_stability vs. colorfulness: r=0.881 (unexpectedly high — may indicate confounding)
- temporal_stability vs. blocking: r=-0.824 (unexpectedly high negative)

**H4. Fix tied-rank handling in composite scoring.** The current `np.argsort` approach does not handle ties correctly (it assigns different ranks to tied values based on array position). Use `scipy.stats.rankdata` with `method='average'` for correct fractional ranks. This is a small code change with correctness implications. **Empirically confirmed:** blown_whites is constant zero on the current dataset, so all 16 clips get arbitrary 1..16 ranks for this metric.

**H5. Add discrimination gating for composite.** Exclude zero-variance metrics from composite scoring to prevent tied-rank noise. Report them as "non-discriminating" rather than silently dropping them.

**H6. Clarify brightness-agnostic documentation.** The claim is accurate for multiplicative gain but not for additive pedestal offsets. See §5 for details. Reframe documentation and add pedestal-sensitivity notes.

### MEDIUM IMPACT

**M1. Add chroma delay detection.** Cross-correlate Y and Cb/Cr edge positions to measure horizontal chroma-luma registration error. This is a diagnostic metric specifically relevant to composite video capture and comb filter quality. Moderate implementation complexity.

**M2. Consider dimension-weighted composite.** As discussed in section 4.3, group correlated metrics into quality dimensions and weight by dimension rather than by individual metric. This prevents the spatial quality cluster from dominating the composite. Could be offered as an alternative scoring mode alongside the current equal-weight rank average.

**M3. Report z-score composite alongside rank composite.** A z-score composite (equal weight, sign-corrected) preserves magnitude information that ranks discard. Reporting both gives users a more complete picture. The z-scores are already computed for heatmap coloring; adding a z-score overall is trivial.

**M4. Evaluate blocking metric discrimination.** If blocking shows CV < 5% across the ProRes clip set, consider either (a) dropping it from the composite scoring while retaining it as an informational metric, or (b) keeping it as-is for robustness against non-ProRes inputs. Report the CV in the metric review.

### NICE TO HAVE

**N1. Implement a simplified NIQE.** Extract the full BRISQUE/NIQE feature set (36 MSCN statistics) and fit an MVG model to the "best" clip as the quality anchor. This would replace the current single-feature naturalness metric with a richer perceptual quality estimate. Requires more implementation effort but is doable without external dependencies.

**N2. Add cross-luminance / cross-chrominance detection.** Spectral analysis for NTSC/PAL subcarrier leakage into the luma channel. High implementation complexity but diagnostically valuable for composite video captures. Irrelevant for S-Video captures.

**N3. Add jitter/TBC stability metric.** Phase correlation between consecutive frames to measure horizontal displacement variance (time-base residual). This would complement temporal stability by separating positional jitter from brightness/noise variation. Moderate implementation complexity.

**N4. Per-region temporal analysis.** Break the frame into spatial regions (top, middle, bottom; or a grid) and report temporal stability per region. This would naturally capture head-switching noise (bottom region anomaly) without a dedicated metric.

**N5. Implement optical flow-based temporal quality.** Replace or supplement the simple frame-difference temporal stability with an optical flow consistency measure. This would better distinguish capture instability from scene motion, but adds complexity (optical flow computation is expensive for full-resolution video).

---

## Summary

The current 11-metric suite provides solid coverage of spatial quality (sharpness, edge strength, detail, texture quality), artifact detection (blocking, ringing), temporal behavior (temporal stability), color (colorfulness), signal statistics (naturalness), and signal headroom (crushed blacks, blown whites). The metric selection reflects thoughtful domain knowledge, particularly in the decision to retain ringing despite its correlation with sharpness.

**Strengths:**
- Empirical correlation analysis confirms the key design decisions: sharpness vs. edge_strength (r=0.768, worth keeping both), detail vs. texture_quality (r=-0.553, genuinely complementary), ringing retained despite r=0.901 with sharpness (domain-justified).
- The metric redundancy decisions (dropping noise, contrast, tonal richness, gradient smoothness) are well-reasoned and empirically validated.

**Weaknesses identified through cross-review:**
- The "brightness-agnostic" claim is accurate for multiplicative gain but not for additive pedestal offsets, which are common in analog capture. Empirical testing shows ~23% sharpness change from a +0.05 pedestal offset.
- `blown_whites` has zero variance on the 16-clip dataset, contributing only tied-rank noise to the composite. Discrimination gating is needed.
- `naturalness` direction (`higher_better=True`) can reward pathological distributions. A distance-from-target approach would be more robust.
- `colorfulness` rewards chroma spread without distinguishing fidelity from noise/boost.
- The spatial quality cluster (sharpness, edge_strength, detail) collectively receives 3/11 of the composite weight despite measuring related dimensions. Edge_strength and detail are particularly correlated (r=0.921).

**Coverage gaps:** Analog-specific artifact detection (dropout, head-switching, chroma delay) and more sophisticated temporal analysis (jitter vs. noise separation) are the most impactful missing dimensions.

**Composite scoring:** The rank-based equal-weight approach is robust but needs (a) tie-aware ranking, (b) discrimination gating for zero-variance metrics, and (c) consideration of dimension-based weighting to prevent the spatial cluster from dominating.
