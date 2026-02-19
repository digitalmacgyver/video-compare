# Code Review

Line-by-line review of all Python files, categorized by severity.

---

## Critical Findings

None.

---

## Major Findings

### M1. XSS via clip names in HTML report (`quality_report.py`)

**Location:** `generate_html()`, lines 983, 1031

**Issue:** Clip names (derived from filenames) are inserted into the DOM via `innerHTML` without HTML escaping:

```javascript
// line 983
td.innerHTML='<span class="rank">#'+(idx+1)+'</span> '+row.name;

// line 1031 (sort handler)
fc.innerHTML='<span class="rank">#'+(i+1)+'</span> '+name;
```

The `row.name` value comes from `json.dumps(hm_data)`, which encodes the Python string as JSON. JSON does NOT escape HTML-significant characters (`<`, `>`, `&`). A malicious filename like `x<img onerror=alert(1) src=x>.mov` would execute JavaScript when the report is opened in a browser.

**Risk:** Low in practice — the tool processes local files, so an attacker would need to plant a malicious filename on the local filesystem. However, if reports are shared (e.g., emailed or posted online), the XSS payload would execute in the recipient's browser.

**Fix:** Use `textContent` for the name portion and create the rank span via DOM API, or escape HTML entities before `innerHTML` insertion:

```javascript
function escHtml(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
td.innerHTML = '<span class="rank">#'+(idx+1)+'</span> ' + escHtml(row.name);
```

Note: `cross_clip_report.py` uses `td.textContent = row.name` (line 339), which is safe. Only `quality_report.py`'s heatmap and sort handler are affected.

---

### M2. Tied-rank bug in composite scoring (both scripts)

**Location:** `compute_composites()` in `quality_report.py` line 599–612, `cross_clip_report.py` line 85–98

**Issue:** `np.argsort` assigns different ranks to tied values based on array position, creating systematic alphabetical bias. Covered in detail in `metric_implementation.md` §13.

**Fix:** Use `scipy.stats.rankdata(method='average')`.

---

### M3. No ffmpeg error checking in `analyze_clip()` (`quality_report.py`)

**Location:** Lines 386, 422

**Issue:** The ffmpeg decode process exit code is never checked:

```python
proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
# ... process frames ...
proc.wait()  # return code ignored
```

If ffmpeg fails mid-decode (corrupt file, I/O error), the loop exits on EOF with however many frames were read. The resulting metrics are based on a partial clip, with no warning to the user.

**Fix:**
```python
proc.wait()
if proc.returncode != 0:
    print(f"WARNING: ffmpeg exited with code {proc.returncode} for {filepath}")
```

---

### M4. Subprocess resource leak on exception (`quality_report.py`)

**Location:** `analyze_clip()` line 386

**Issue:** No `try/finally` block around the frame processing loop. If a Python exception occurs (e.g., numpy memory error, unexpected data shape), the ffmpeg subprocess is never terminated:

```python
proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
# If exception here, proc is leaked
while True:
    frame = read_frame(proc.stdout, width, height)
    # ...
proc.wait()
```

**Fix:** Wrap in try/finally:
```python
proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
try:
    while True:
        frame = read_frame(proc.stdout, width, height)
        # ...
finally:
    proc.stdout.close()
    proc.wait()
```

Or use `subprocess.Popen` as a context manager (Python 3.2+).

---

### M5. No-frame edge case produces NaN metrics (`quality_report.py`)

**Location:** Lines 393–396, 424–429

**Issue:** If `skip_frames` is greater than or equal to the total frame count of a clip (or the clip is empty/corrupt), the frame processing loop never executes. All metric accumulators remain empty lists. When `np.mean([])` or `np.median([])` is called, it returns `NaN`, which propagates into the JSON output, rank computation, and z-score calculation. Downstream operations (percentile selection for comparison frames, Chart.js rendering) may silently produce incorrect results or fail.

**Fix:** Validate `n_frames > 0` immediately after the frame loop:
```python
if n_frames == 0:
    print(f"ERROR: No frames decoded for {filepath} (skip={skip_frames})")
    proc.wait()
    return None, None  # or raise an exception
```

---

### M6. Silent clip name collisions (`quality_report.py`)

**Location:** `short_name()` line 353–372, `main()` line 1521

**Issue:** `short_name()` extracts a device name by stripping suffixes and taking the first token. If two different files resolve to the same short name (e.g., `device_a_svideo_direct.mov` and `device_a_svideo_sdi.mov` both become `device_a`), the second clip silently overwrites the first in the `all_results` dict:

```python
name = short_name(clip_path)
# ...
all_results[name] = metrics  # overwrites if name already exists
```

**Impact:** One clip's metrics are silently lost with no warning. The report shows fewer clips than expected.

**Fix:** Detect duplicates before assignment:
```python
if name in all_results:
    print(f"WARNING: Duplicate short name '{name}' from {clip_path}")
    # Append suffix: device_a_2, device_a_3, etc.
    i = 2
    while f"{name}_{i}" in all_results:
        i += 1
    name = f"{name}_{i}"
```

---

## Minor Findings

### m1. Outdated docstring (`quality_report.py` line 2–4)

The module docstring says "576x436, 10-bit yuv422p10le" but the code auto-detects resolution via ffprobe. The hardcoded dimensions are outdated — the tool works with any resolution.

### m2. Outdated text report message (`quality_report.py` line 636)

```python
lines.append("All clips normalized to common brightness/saturation reference before measurement.")
```

The current workflow does not require normalization. This message is a leftover from the pre-brightness-agnostic era.

### m3. Debug leftover: `or True` (`quality_report.py` line 1549)

```python
if args.screenshots > 0 or True:  # always extract comparison frames
```

The `or True` makes the condition always true. The comment explains the intent (always extract comparison frames), but `or True` is a code smell. Should be simplified:

```python
# Always extract metric comparison frames
print("\nExtracting metric comparison frames...")
comparisons = find_comparison_frames(...)

if args.screenshots > 0:
    # ...
```

### m4. Slow frame extraction via select filter (`quality_report.py` line 457–464)

```python
"-vf", f"select=eq(n\\,{frame_num})"
```

The `select` filter decodes all frames from the start to reach frame N. For frames deep into a clip (e.g., frame 400 of a 500-frame clip), this is wasteful. Using `-ss` with time-based seeking would be faster:

```python
# Approximate time-based seek (requires knowing fps)
time_sec = frame_num / fps
cmd = ["ffmpeg", "-ss", str(time_sec), "-i", filepath, "-vframes", "1", ...]
```

This affects the `find_comparison_frames` and `extract_sample_screenshots` functions, which make N × M ffmpeg calls (N screenshots × M clips).

### m5. `extract_frame_jpeg` doesn't validate empty output (`quality_report.py` line 464)

```python
return proc.stdout if proc.returncode == 0 else b""
```

If ffmpeg succeeds but produces zero bytes (e.g., seeking past end of file), the function returns empty bytes. The caller checks `if fdata["jpeg_b64"]:` which handles this, but a zero-byte JPEG should be treated as a failure.

### m6. Hardcoded cross-clip subtitle says "9 metrics" (`cross_clip_report.py` line 269)

```python
"Each clip was independently normalized and evaluated on 9 no-reference metrics."
```

Should be "11" (current metric count), and "independently normalized" is no longer accurate.

### m7. Code duplication across scripts

The following are copy-pasted between `quality_report.py` and `cross_clip_report.py`:
- `ALL_KEYS` (11 entries, must be kept in sync)
- `METRIC_INFO` (11 entries, must be kept in sync)
- `COLORS_7` / `COLORS_14` (color palettes)
- `compute_composites()` (entire function, ~50 lines)
- `HTML_CSS` (CSS block, ~80 lines)

Similarly, between `normalize.py` and `normalize_linear.py`:
- `probe_video()`, `detect_frame_rate()`, `decode_command()`, `encode_command()`, `read_frame()`, `write_frame()` — identical functions

This creates a maintenance burden: any change must be applied to both files. A shared `common.py` module would eliminate the duplication.

### m8. Normalize scripts: reference clip is self-normalized wastefully

**Location:** `normalize.py` line 297, `normalize_linear.py` line 249

Both scripts process the reference clip through the normalization pipeline (two-pass decode + encode). Since the normalization of a clip against itself produces an identity transform (histogram matching CDF to itself → LUT[i] = i; linear scaling with scale=1.0), the output is identical to the input (modulo re-encoding losses). The reference clip should be skipped or simply copied.

### m9. `normalize_linear.py` output filename bug for non-.mov inputs

**Location:** Line 251

```python
out_name = clip_name.replace(".mov", "_linnorm.mov")
```

If the input pattern is `--pattern "*.mp4"`, then `clip_name = "file.mp4"` and `.replace(".mov", ...)` produces `"file.mp4"` (no change). The output goes to a different directory so it won't overwrite the source, but the filename is confusing (no `_linnorm` suffix, and original extension retained).

Same issue in `normalize.py` line 299:
```python
out_name = clip_name.replace(".mov", "_normalized.mov")
```

**Fix:** Use `os.path.splitext()` to handle any extension:
```python
base, ext = os.path.splitext(clip_name)
out_name = f"{base}_linnorm{ext}"
```

### m10. Hardcoded path in `verify_signalstats.py`

**Location:** Line 11

```python
NORM_DIR = "/wintmp/analog_video/noref_compare/normalized"
```

Should accept a command-line argument for the directory path.

### m11. Unused import in `verify_signalstats.py`

**Location:** Line 9

```python
import re
```

Never used. Remove.

### m12. `verify_signalstats.py` ffprobe movie filter quoting

**Location:** Line 18

```python
f"movie='{filepath}',signalstats"
```

If `filepath` contains single quotes (unlikely but possible), this breaks the lavfi filter expression. The filepath should be escaped or the approach changed to use `-i filepath -vf signalstats`.

### m13. `compute_composites()` docstring says "9 metrics" (`quality_report.py` line 577)

```python
"""... The overall score is the average rank across all 9 metrics ..."""
```

Should say "all metrics" or "all 11 metrics."

### m14. `find_comparison_frames` frame index mismatch across clips

**Location:** `quality_report.py` lines 503–515

The `frame_idx` is determined from the best clip's per-frame array. If other clips have different frame counts (different skip offsets or clip lengths), `frame_idx` may exceed their array length:

```python
frame_val = float(pf_arr[frame_idx]) if frame_idx < len(pf_arr) else 0.0
```

The fallback to `0.0` is silent and may misrepresent the metric value for that frame. A warning would be helpful.

### m15. No test for edge cases (all-black, all-white frames)

**Location:** `test_cases/test_metrics.py`

The test suite validates directional correctness (sharp vs. blur, clean vs. noisy) but does not test edge cases:
- All-black frame (Y = 0 everywhere)
- All-white frame (Y = 1 everywhere)
- Constant frame (Y = 0.5 everywhere)
- Single bright pixel in a dark field

These edge cases exercise the epsilon guards and division-by-zero protections. Several metrics return 0.0 or 0.5 for degenerate inputs, but this behavior is not verified by tests.

### m16. `cross_clip_report.py` device name matching

**Location:** Line 166–170

Device matching across clips requires exact name match:
```python
common_devices = device_sets[0]
for ds in device_sets[1:]:
    common_devices &= ds
```

If the same device has slightly different names across reports (e.g., `jvc_tbc` vs `jvctbc`), it will be excluded. The `short_name()` function in `quality_report.py` handles name normalization, but if different clip directories use different naming conventions, the cross-clip comparison silently drops those devices.

### m17. `normalize.py` missing `--pattern` argument

**Location:** `normalize.py` main()

`normalize_linear.py` has a `--pattern` argument for non-.mov files; `normalize.py` does not. It hardcodes `"*.mov"` at line 257:

```python
clips = sorted(glob.glob(os.path.join(src_dir, "*.mov")))
```

### m18. Division-by-zero risk in normalize scripts

**Location:** `normalize.py` lines 148, 197, 203

If ffmpeg decode yields zero frames (corrupt file, wrong path), several divisions by `n_frames` or `sat_count` execute without guards:

```python
# line 148 — histogram accumulation divided by frame count
hist_sum = hist_sum / n_frames  # ZeroDivisionError if n_frames == 0

# line 197 — saturation scale factor
scale = ref_sat / clip_sat  # ZeroDivisionError if clip_sat == 0.0

# line 203 — similar pattern
```

**Impact:** Unhandled `ZeroDivisionError` crashes the normalization pipeline with an unhelpful traceback.

**Fix:** Guard zero-frame and zero-sample states:
```python
if n_frames == 0:
    print(f"ERROR: No frames decoded for {filepath}")
    return
```

### m19. `verify_signalstats.py` crashes on empty input set

**Location:** `verify_signalstats.py` line 70

If no clips match the glob pattern (wrong directory, no `.mov` files), the `vals` list is empty. `min(vals)` and `max(vals)` raise `ValueError: min() arg is an empty sequence`.

**Fix:** Guard empty results:
```python
if not vals:
    print("No clips found — check directory path")
    sys.exit(1)
```

---

## Style / Cleanup

### s1. `compute_composites` duplicated comment style

Both versions (quality_report.py and cross_clip_report.py) have slightly different docstrings for the same function. The quality_report.py version says "9 metrics" while cross_clip_report.py says "all metrics."

### s2. Inconsistent `short_name` implementations

`quality_report.py` and `verify_signalstats.py` both have `short_name()` functions with different logic:
- `quality_report.py`: strips many suffixes, handles spaces, strips "twister_"
- `verify_signalstats.py`: only strips "twister_" and one specific suffix

### s3. Magic numbers without named constants

- Canny thresholds `50, 150` (line 245)
- Dilation kernel `(5, 5)` (line 248)
- Block sizes `16` (detail, texture_quality)
- Block size `8` (blocking)
- MSCN kernel `(7, 7)` with sigma `1.166` (line 317)

These could be module-level constants for readability.

### s4. CSS variable definitions not reused

Both `quality_report.py` and `cross_clip_report.py` define the same CSS custom properties (colors, fonts, layout) in separate `HTML_CSS` strings. A shared CSS string would ensure visual consistency and simplify maintenance.

---

## Summary by File

| File | Critical | Major | Minor | Style |
|------|----------|-------|-------|-------|
| `quality_report.py` | 0 | 6 (M1-M6) | 8 (m1-m5, m7, m13-m14) | 2 (s3-s4) |
| `cross_clip_report.py` | 0 | 1 (M2) | 3 (m6-m7, m16) | 2 (s1, s4) |
| `normalize.py` | 0 | 0 | 4 (m7-m8, m17-m18) | 0 |
| `normalize_linear.py` | 0 | 0 | 2 (m7, m9) | 0 |
| `verify_signalstats.py` | 0 | 0 | 4 (m10-m12, m19) | 1 (s2) |
| `test_cases/test_metrics.py` | 0 | 0 | 1 (m15) | 0 |
| **Total** | **0** | **6** | **19** | **4** |
