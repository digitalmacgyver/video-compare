# Review Project Plan

## Scope
Comprehensive review of the video-compare project covering metric selection, metric implementation, architecture, and code quality.

## Phase 1: Metric Selection Review (`metric_selection.md`)

### Approach
- Evaluate the 11 selected metrics against the IQA literature for analog video capture assessment
- Identify coverage gaps: what important quality dimensions are missing?
- Assess the dropped metrics rationale (noise, contrast, tonal_richness, grad_smoothness)
- Evaluate the composite scoring approach (rank-based, equal weight)
- Consider domain-specific analog video artifacts not covered (e.g., dropout, head-switching, skew/tracking, chroma bleed, cross-luminance)

### Deliverable
Assessment of metric coverage with recommendations for additions or replacements.

## Phase 2: Metric Implementation Review (`metric_implementation.md`)

### Approach
Review each of the 11 metric functions for:
- **Correctness**: Does the implementation match the stated algorithm?
- **Brightness bias**: Could brightness/gain differences systematically affect rankings? (We experimentally verified detail CV is brightness-agnostic; check all others)
- **Resolution sensitivity**: Are metrics tuned for 576x436 SD content or portable?
- **Numerical stability**: Division by zero guards, edge cases (all-black frames, all-white frames)
- **Direction correctness**: Is higher_better/lower_better correctly assigned in METRIC_INFO?
- **Aggregation**: Is mean vs median appropriate for each metric's distribution?

### Key functions to review
- `sharpness_laplacian()` — Laplacian variance / mean_Y²
- `edge_strength_sobel()` — Sobel gradient / mean_Y
- `blocking_artifact_measure()` — 8x8 boundary ratio
- `detail_texture()` — local CV median
- `texture_quality_measure()` — smoothed/original variance ratio
- `ringing_measure()` — near-edge Laplacian / mean_Y
- `colorfulness_metric()` — Hasler-Süsstrunk on Cb/Cr planes
- `naturalness_nss()` — MSCN kurtosis
- `crushed_blacks()` — shadow headroom ratio
- `blown_whites()` — highlight headroom ratio
- `noise_estimate()` — (dropped but still in code)
- `temporal_stability` computation in `analyze_clip()`

### Deliverable
Per-metric implementation analysis with specific bug reports and improvement recommendations.

## Phase 3: Architecture Review (`architecture.md`)

### Approach
- Evaluate single-file vs modular structure
- Assess code duplication across scripts (METRIC_INFO, compute_composites, CSS duplicated in quality_report.py and cross_clip_report.py)
- Review data flow: ffmpeg → numpy → metrics → JSON → HTML
- Evaluate HTML generation approach (string concatenation with embedded JS)
- Review cross_clip_report.py device name matching limitation
- Assess test coverage and test design
- Consider scalability (clip count, frame count, resolution)

### Deliverable
Architecture assessment with specific refactoring recommendations.

## Phase 4: Code Review (`code_review.md`)

### Approach
Line-by-line review of all Python files for:
- **Bugs**: Logic errors, off-by-one, incorrect formulas
- **Anti-patterns**: Global state, magic numbers, dead code
- **Error handling**: Missing error handling, silent failures
- **Resource management**: Subprocess pipe leaks, memory usage
- **Security**: Command injection via filenames, XSS in HTML output
- **Performance**: Unnecessary recomputation, vectorization opportunities
- **Maintainability**: Code clarity, naming, documentation

### Files to review
1. `quality_report.py` (1576 lines — primary focus)
2. `cross_clip_report.py` (448 lines)
3. `normalize.py` (311 lines)
4. `normalize_linear.py` (264 lines)
5. `verify_signalstats.py` (76 lines)
6. `test_cases/test_metrics.py` (405 lines)

### Deliverable
Categorized list of findings by severity (critical, major, minor, style).

## Execution Order

1. **Phase 2 first** — Implementation review catches bugs that directly affect results
2. **Phase 1 second** — Metric selection builds on understanding from implementation review
3. **Phase 4 third** — Code review covers everything else
4. **Phase 3 last** — Architecture review synthesizes across all findings

This ordering front-loads the most impactful findings (metric correctness) and leaves the broader synthesis for last.
