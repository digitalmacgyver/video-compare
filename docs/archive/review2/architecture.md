# Architecture Review

Assessment of the project's structure, data flow, code organization, HTML generation, test coverage, and scalability. This synthesizes findings from the metric implementation review and code review.

---

## 0. Requirements Traceability

Before assessing architecture, it is worth mapping the project's implicit requirements against current implementation status:

| ID | Requirement | Status | Notes |
|----|------------|--------|-------|
| R-01 | No-reference comparison across capture pipelines | **Met** | 11 metrics, rank-based composite; correctness impacted by tied-rank bug (M2) |
| R-02 | Brightness-agnostic comparison without mandatory normalization | **Partially met** | Multiplicative gain invariance is strong; additive pedestal invariance is weak (see `metric_implementation.md` §12b) |
| R-03 | Cross-clip comparability | **Met** | `cross_clip_report.py` matches devices by name; fragile if naming conventions differ across directories |
| R-04 | Reproducible, auditable reports | **Partially met** | JSON output exists but lacks schema version, metric version, or run parameters metadata |
| R-05 | Operational robustness on real datasets | **Partially met** | Key edge cases (no-frame decode, name collisions, empty inputs) are not consistently guarded |

R-02 and R-04 are the most significant gaps. R-02 is a mathematical limitation documented in `metric_implementation.md`; R-04 is an omission that is straightforward to fix.

---

## 1. Single-File Architecture

### Current Structure
```
quality_report.py      (1576 lines)  — metrics, analysis, HTML generation, CLI
cross_clip_report.py   (448 lines)   — cross-clip comparison, HTML generation, CLI
normalize.py           (311 lines)   — Y histogram matching + chroma scaling
normalize_linear.py    (264 lines)   — linear Y scaling
verify_signalstats.py  (76 lines)    — normalization verification
test_cases/test_metrics.py (405 lines) — synthetic metric tests
```

### Assessment

**Appropriate for current scale.** The project is a specialized analysis pipeline, not a general-purpose library. Each script serves a distinct CLI function. The largest file (`quality_report.py` at 1576 lines) is at the upper end of comfortable single-file maintenance, but it follows a clear section structure (frame I/O → metrics → analysis → scoring → text report → HTML report → main).

**The case against modularizing:** The project's CLAUDE.md explicitly states "Single-file scripts, no package structure." For a pipeline that is run infrequently (ad-hoc quality comparison sessions) rather than deployed as a library, the simplicity of "one file = one tool" has real value. There are no import cycles, no `__init__.py` files to maintain, no package-level concerns.

**The case for partial modularization:** The code duplication identified in the code review (§m7) is the strongest argument. Six functions are identical between `normalize.py` and `normalize_linear.py`. Five constants and one major function are duplicated between `quality_report.py` and `cross_clip_report.py`. A single `common.py` containing the shared elements would eliminate ~150 lines of duplication while maintaining the single-file-per-tool principle for the main scripts.

### Recommendation

**Extract a `common.py` module** containing:
- `ALL_KEYS`, `METRIC_INFO`, `COLORS_*` constants
- `compute_composites()` function
- `probe_video()`, `detect_frame_rate()`, `decode_command()`, `encode_command()`, `read_frame()`, `write_frame()` I/O functions
- The shared `HTML_CSS` string

This reduces duplication without introducing framework-level complexity. Each script remains a self-contained CLI tool that imports shared constants and utilities.

**Priority: Medium.** The duplication is a maintenance risk (changes must be applied in multiple places) but not a correctness risk today.

---

## 2. Data Flow

### Pipeline Architecture

```
ffmpeg decode → raw YUV frames → numpy arrays → metric functions → per-frame values
                                                                        ↓
                                                              aggregation (mean/median)
                                                                        ↓
                                                              JSON output
                                                                        ↓
                                                    compute_composites (ranks, z-scores)
                                                                        ↓
                                                    HTML generation (Chart.js, heatmaps)
```

### Streaming vs. Batch

The current design is **streaming**: frames are read one at a time from ffmpeg's pipe, each metric is computed per-frame, and results are accumulated in lists. This is memory-efficient — only one frame (plus prev_yf for temporal stability) is held in memory at a time.

**Strength:** Handles arbitrarily long clips without memory issues. A 10-minute clip at 59.94fps produces ~36,000 frames; only per-frame scalar results (~100 bytes per frame per metric) are accumulated.

**Weakness:** Metrics that require multi-frame analysis (e.g., the temporal stability computation needs two consecutive frames) are handled with ad-hoc state (`prev_yf`). Adding a metric that needs a sliding window (e.g., a 5-frame jitter detector) would require restructuring the loop.

### Assessment

**Well-designed for current requirements.** The streaming architecture is correct for frame-by-frame metrics on large video files. The separation of per-frame computation from aggregation (mean/median) is clean.

---

## 3. HTML Generation

### Current Approach

HTML is constructed via Python f-strings and string concatenation, with JavaScript data embedded as JSON:

```python
html = f"""<!DOCTYPE html>..."""
html += f"""<div>...</div>"""
# Data injected via json.dumps
html += f"""const data = {json.dumps(data)};"""
```

### Assessment

**Functional but fragile.** The approach works for the current report complexity but has several drawbacks:

1. **No templating.** HTML structure, CSS, and JavaScript are mixed into a single Python string. Modifying the report layout requires editing deeply nested string fragments. There is no syntax highlighting, auto-completion, or validation during development.

2. **Escaping risks.** Python f-string interpolation inside JavaScript template literals creates multiple escaping contexts (Python → HTML → JavaScript). The XSS finding (M1 in code review) demonstrates this risk. A template engine would provide automatic context-aware escaping.

3. **Monolithic output.** The HTML file embeds all data (including base64-encoded screenshots) inline. A report with 14 clips × 11 metrics × 2 comparison frames ≈ 308 JPEG images, each ~50KB base64 ≈ 15MB of HTML. This is large but works in modern browsers.

### Alternatives Considered

- **Jinja2 templates:** Would separate HTML structure from Python logic, provide auto-escaping, and enable syntax highlighting. Adds a dependency.
- **External CSS/JS files:** Would enable proper tooling (linting, minification) but breaks the self-contained single-file report principle.
- **React/Vue static generation:** Overkill for this use case.

### Recommendation

**Keep the current approach** for the self-contained single-file report requirement, but:
1. Fix the XSS issue (M1) by escaping HTML entities in clip names.
2. Extract the Chart.js configuration into a separate function for readability.
3. Consider moving the CSS string to a module-level constant (already done) and the JavaScript to named functions rather than inline blocks.

**Priority: Low.** The current approach works. Templating would be an improvement but adds a dependency for limited benefit.

---

## 4. Test Coverage

### Current Coverage

| Component | Coverage | Notes |
|-----------|----------|-------|
| 11 metric functions | **Good** | 10 of 11 have direct tests; temporal stability tested via manual frame-diff loop |
| Brightness invariance | **Partial** | 3 metrics tested (sharpness, edge_strength, detail). 8 not tested. |
| Edge cases | **None** | No tests for all-black, all-white, constant frames |
| `analyze_clip()` | **None** | Integration test would require video files |
| `compute_composites()` | **None** | Rank computation, z-scores, tied ranks untested |
| `short_name()` | **None** | Name parsing untested |
| HTML generation | **None** | No validation of HTML output |
| `cross_clip_report.py` | **None** | No tests at all |
| `normalize.py` | **None** | Tested indirectly via verify_signalstats.py |

### Assessment

**Metric function tests are solid.** The synthetic test images are well-designed: each tests a specific artifact against a clean reference, with directional assertions (sharp > blurred, clean > noisy). The brightness invariance tests verify the most important property.

**Significant gaps:**

1. **No tests for `compute_composites()`.** The tied-rank bug (M2) would have been caught by a test with tied input values.

2. **No edge case tests.** The epsilon guards in metrics (mean_y < 0.01, var < 1e-8, etc.) are critical for robustness but untested.

3. **No integration tests.** The full pipeline (ffmpeg → analysis → JSON → HTML) is only tested manually. A smoke test with a short synthetic video would catch I/O issues, pipe handling errors, and format mismatches.

### Recommendations

**HIGH: Add `compute_composites()` unit tests.** Test with:
- Simple 3-clip, 2-metric scenario with known ranks
- Tied values (verify correct fractional ranks after the M2 fix)
- All-equal values (all clips should get the same rank)

**MEDIUM: Add edge case metric tests.** Test each metric with:
- `np.zeros((64, 64))` (all-black)
- `np.ones((64, 64))` (all-white)
- `np.full((64, 64), 0.5)` (constant gray)
- Verify no exceptions, no NaN, no infinity

**LOW: Add a pipeline integration test.** Generate a short synthetic video (e.g., 10 frames of a gradient), run `quality_report.py` on it, verify JSON output has expected structure.

---

## 5. Scalability

### Current Performance Profile

For 14 clips at 576×436, ~500 frames each:
- **Metric computation:** ~30 seconds per clip (dominated by frame I/O and Python metric loops)
- **Comparison frame extraction:** ~2 minutes (11 metrics × 14 clips × 1 ffmpeg call each = 154 ffmpeg invocations)
- **HTML generation:** < 1 second
- **Total:** ~10 minutes for the full pipeline

### Scaling Dimensions

| Dimension | Current | Scaling Behavior | Bottleneck |
|-----------|---------|------------------|------------|
| Clip count | 14 | O(N²) for comparison screenshots; O(N·log N) for ranking | Frame extraction (N × M ffmpeg calls) |
| Frame count | ~500 | O(F) linear — streaming architecture | Metric computation loop |
| Resolution | 576×436 | O(W·H) per frame — some metrics are O(W·H·log) | Sobel, Laplacian, GaussianBlur are OpenCV-optimized |
| Metric count | 11 | O(K) linear — each metric is computed independently | Adding metrics is straightforward |

### Performance Bottlenecks

1. **Python loops in `detail_texture` and `texture_quality_measure`.** For 576×436 at 16×16 blocks, there are ~972 blocks × 2 metrics × 500 frames = ~972K loop iterations. Vectorizing these with numpy operations would provide 10–50× speedup on these functions.

2. **Frame extraction via `select` filter.** Each `extract_frame_jpeg` call decodes all frames from the start of the clip to reach the target frame. For frame 400, this means decoding 400 frames to extract 1. With 154 calls across all clips and metrics, the total is ~30K+ frames decoded unnecessarily. Time-based seeking (`-ss`) would reduce this to ~154 seeks.

3. **Sequential clip processing.** Clips are analyzed one at a time. Since each clip's analysis is independent, parallel processing (multiprocessing pool) could provide near-linear speedup on multi-core systems.

### Recommendations

**HIGH: Vectorize block-based metrics.** Replace Python loops with numpy:
```python
# Example: detail_texture vectorized
def detail_texture(yf, block=16):
    h, w = yf.shape
    nh, nw = h // block, w // block
    cropped = yf[:nh*block, :nw*block]
    blocks = cropped.reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(-1, block, block)
    means = blocks.mean(axis=(1, 2))
    stds = blocks.std(axis=(1, 2))
    mask = means > 0.01
    cvs = stds[mask] / means[mask]
    return float(np.median(cvs)) if cvs.size > 0 else 0.0
```

**MEDIUM: Use time-based seeking for frame extraction.** Replace:
```python
"-vf", f"select=eq(n\\,{frame_num})"
```
with:
```python
time_sec = frame_num / fps
"-ss", str(time_sec), "-i", filepath
```

This requires passing fps to the extraction functions but eliminates decoding all preceding frames.

**LOW: Add parallel clip processing.** Use `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` to analyze multiple clips simultaneously. The streaming architecture already keeps per-clip memory usage low, so parallel execution is straightforward.

---

## 6. Error Handling Philosophy

### Current Approach

Errors are handled inconsistently:
- **CLI argument errors:** Proper validation with helpful messages and `sys.exit(1)`.
- **File system errors:** `glob.glob` silently returns empty list, caught by "no files found" check.
- **ffmpeg errors:** Silently ignored (analyze_clip), partially checked (normalize_clip checks encode returncode but not decode).
- **ffprobe errors:** `check_output` raises `CalledProcessError` — unhandled, will crash with a traceback.
- **JSON parse errors:** Unhandled — `json.load` failure gives raw Python traceback.

### Assessment

For a personal analysis tool, this is acceptable. Crashes with tracebacks provide enough diagnostic information for the developer. However, two improvements would significantly improve usability:

1. **Check ffmpeg decode exit codes** (M3 from code review). A corrupt file producing partial results is a subtle failure mode that can lead to misleading quality comparisons.

2. **Wrap ffprobe failures** in `try/except` with a clear error message. Currently, a missing or corrupt file causes an unhandled `CalledProcessError` with a confusing traceback.

---

## 7. Cross-Script Consistency

### Name Resolution

`quality_report.py`'s `short_name()` function extracts device names from filenames by stripping known suffixes. `cross_clip_report.py` then matches devices across clips by exact string comparison of these names. This creates a fragile coupling:

- If clip directories use different naming conventions, device names won't match.
- If `short_name()` is updated, previously generated JSON files will have old-format names that don't match new-format names.

**The JSON files don't store the original filename**, only the short name. There is no way to recover the mapping after the fact.

### Recommendation

Store the original filename in the JSON output alongside the short name:
```json
{
  "device_name": {
    "sharpness": {"mean": ..., "std": ...},
    ...
    "_metadata": {"source_file": "original_filename.mov", "n_frames": 500}
  }
}
```

This preserves the ability to re-derive the name mapping and debug matching issues.

---

## 8. JSON Schema and Reproducibility

### Current State

The JSON output has no schema version, no metric version, and no run parameters:

```json
{
  "device_name": {
    "sharpness": {"mean": 0.123, "std": 0.045, "per_frame": [...]},
    ...
  }
}
```

There is no way to determine from the JSON file alone:
- Which version of the metric definitions produced it
- What parameters were used (skip frames, pattern, etc.)
- Whether the metric set has changed since the file was written

### Impact

The `cross_clip_report.py` script already handles backward compatibility with old 9-metric JSONs (it intersects available keys). But as metrics are added, removed, or recalibrated, comparing JSONs from different tool versions becomes unreliable without explicit versioning.

### Recommendation

Add a `_metadata` block to the JSON output:

```json
{
  "_metadata": {
    "schema_version": 2,
    "metrics": ["sharpness", "edge_strength", ...],
    "n_metrics": 11,
    "source_dir": "/path/to/clips",
    "pattern": "*.mov",
    "skip_frames": {"jvctbc1": 1},
    "timestamp": "2026-02-18T14:30:00"
  },
  "device_name": { ... }
}
```

This enables:
- Version-aware cross-clip comparison (warn if schema versions differ)
- Audit trail for how reports were generated
- Graceful handling of metric set evolution

The `_metadata` key with leading underscore won't collide with device names and is already the convention used by `cross_clip_report.py` for clip-level metadata.

**Priority: Medium.** Low effort, moderate long-term value for reproducibility.

---

## 9. Summary of Recommendations

### Priority Matrix

| Priority | Recommendation | Effort | Impact |
|----------|---------------|--------|--------|
| **HIGH** | Fix tied-rank bug (M2) | Small | Correctness |
| **HIGH** | Fix XSS in HTML (M1) | Small | Security |
| **HIGH** | Add ffmpeg error checking (M3) | Small | Reliability |
| **HIGH** | Add subprocess cleanup (M4) | Small | Reliability |
| **HIGH** | Guard no-frame edge case (M5) | Small | Reliability |
| **HIGH** | Detect clip name collisions (M6) | Small | Correctness |
| **HIGH** | Add `compute_composites()` tests | Medium | Test coverage |
| **MEDIUM** | Add JSON schema versioning (§8) | Small | Reproducibility |
| **MEDIUM** | Extract `common.py` shared module | Medium | Maintainability |
| **MEDIUM** | Vectorize block-based metrics | Medium | Performance (2-5× speedup on detail/texture) |
| **MEDIUM** | Use time-based seeking for screenshots | Small | Performance (screenshot extraction) |
| **MEDIUM** | Fix outdated strings (m2, m6, m13) | Trivial | Accuracy |
| **MEDIUM** | Guard normalize.py division-by-zero (m18) | Small | Reliability |
| **LOW** | Add edge case metric tests | Medium | Robustness |
| **LOW** | Add parallel clip processing | Medium | Performance |
| **LOW** | Store original filenames in JSON | Small | Debuggability |
| **LOW** | Document additive pedestal limitation | Trivial | Accuracy |
