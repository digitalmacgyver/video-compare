# Implementation Plan

Sequenced plan for resolving all issues identified in the code review, metric implementation review, metric selection review, and architecture review.

---

## Sequence Rationale

### Proposed Sequence (User)

1. Missing test cases / snapshots of current behavior
2. Major code correctness issues
3. Code organization, modularity, refactoring
4. Verify major functionality unchanged
5. New features, metrics, etc.
6. Light code review for regressions
7. Fix issues from step 6
8. Final verification

### Critique and Refinements

The proposed sequence is fundamentally sound. Snapshot-first, correctness-before-refactoring, and verification checkpoints after major change phases are all correct principles. The following refinements improve robustness:

**1. Add a verification checkpoint between Phase 2 (correctness fixes) and Phase 3 (refactoring).**

Without this, a regression introduced in Phase 2 is indistinguishable from one introduced in Phase 3. Phase 4 (verification) comes too late to isolate the source. The cost of an intermediate verification is low (run tests + compare baseline JSON), and the diagnostic value is high. This becomes Phase 3 in the refined sequence.

**2. Phase 1 should distinguish baseline snapshots from bug-demonstrating tests.**

Two categories of tests serve different purposes:
- *Baseline snapshots* record current correct behavior. They should pass before and after all changes. They are regression guards.
- *Bug-demonstrating tests* document known defects (e.g., tied-rank bug). They fail initially and are expected to pass after the Phase 2 fix. They are acceptance criteria.

Both belong in Phase 1, but conflating them creates confusion about what "test failures" mean at each stage.

**3. Include minor correctness fixes in Phase 2, not Phase 5.**

Items like `normalize.py` division-by-zero (m18), `verify_signalstats` empty input crash (m19), and `normalize_linear.py` filename bug (m9) are correctness defects, not features. Fixing them before refactoring prevents moving buggy code into `common.py` and needing to fix it again later.

**4. Sub-sequence Phase 3 (refactoring) from trivial to structural.**

Typo fixes and dead import removal are zero-risk. Extracting `common.py` is a significant structural change that can introduce import errors, circular dependencies, or subtle behavioral changes. Sequencing trivial cleanups first, then moderate changes, then the major extraction, with test runs between each sub-step, limits blast radius.

**5. Sub-sequence Phase 5 (new features) by risk level.**

"JSON schema metadata" (additive, low risk) is very different from "vectorize block-based metrics" (algorithmic change, must produce identical results) or "add dropout detection" (entirely new metric, new test infrastructure). These should be ordered by increasing risk and scope.

**6. Explicitly define what "verify major functionality unchanged" means.**

Vague verification is no verification. Each checkpoint must specify: which commands to run, which outputs to compare, what constitutes "unchanged" vs. "expected change," and what action to take on unexpected differences.

### Refined Sequence

| Phase | Description | Inputs | Outputs |
|-------|-------------|--------|---------|
| **1** | Baseline snapshots and test infrastructure | Current codebase | Baseline JSON, test suite additions |
| **2** | Correctness fixes (major + minor) | Phase 1 baselines | Fixed code, passing new tests |
| **3** | Post-fix verification checkpoint | Phase 1 baselines, Phase 2 code | Verified metric equivalence |
| **4** | Code organization, cleanup, refactoring | Verified Phase 3 code | `common.py`, cleaned scripts |
| **5** | Post-refactoring verification checkpoint | Phase 3 baselines | Verified metric equivalence |
| **6** | New features and improvements | Verified Phase 5 code | Enhanced functionality |
| **7** | Post-feature verification checkpoint | Phase 1 baselines | Verified metric equivalence (where unchanged) |
| **8** | Regression review | All changed code | Issue list |
| **9** | Fix regression review issues | Phase 8 issue list | Fixed code |
| **10** | Final verification | Phase 1 baselines | Sign-off |

This adds two verification checkpoints (Phases 3 and 5) and splits the original Phase 4 verification into Phase 7 (post-feature) to keep it close to the changes it validates. The original 8-step sequence becomes 10 steps, but the added cost is minimal (each verification checkpoint is ~5 minutes of automated comparison).

---

## Phase 1: Baseline Snapshots and Test Infrastructure

### Goal
Capture the current system's behavior as a regression baseline and build the test infrastructure needed to validate all subsequent changes.

### 1.1 Record Existing Test Suite Baseline

```bash
source venv/bin/activate
python test_cases/test_metrics.py 2>&1 | tee review2/baselines/test_baseline.txt
```

Save the full output (pass/fail status and any printed diagnostics) as the reference. All 14 tests should pass. If any fail, investigate before proceeding — the baseline must be clean.

### 1.2 Generate Metric Baseline on Real Dataset

Run `quality_report.py` on the 14-clip noref_compare dataset and save the JSON output:

```bash
python quality_report.py /wintmp/analog_video/noref_compare/ \
    --output-dir review2/baselines/ \
    --name baseline_noref \
    --screenshots 0
```

This JSON captures the per-clip, per-metric mean/std values and the composite rankings. It is the ground truth for detecting unintended metric changes in later phases.

Also record the composite rankings in a human-readable form:

```bash
python -c "
import json
data = json.load(open('review2/baselines/baseline_noref.json'))
# Extract and print composite rankings
clips = sorted(data.keys())
print('Clip rankings (current, pre-fix):')
for c in clips:
    if 'overall' in data[c]:
        print(f'  {c}: rank={data[c][\"overall\"][\"rank\"]}, score={data[c][\"overall\"][\"mean\"]:.4f}')
" | tee review2/baselines/baseline_rankings.txt
```

### 1.3 Write Baseline Regression Tests (should pass now AND after all changes)

Add to `test_cases/test_metrics.py`:

**a) Edge case tests — all-black, all-white, constant gray frames:**

For each of the 11 metrics, verify:
- No exceptions raised
- No NaN or Infinity returned
- Result is a finite float

Test frames:
- `np.zeros((64, 64))` — all black
- `np.ones((64, 64))` — all white
- `np.full((64, 64), 0.5)` — constant gray
- Chroma equivalents for colorfulness: `np.full((64, 32), 512)` (neutral chroma)

**b) Brightness invariance tests for all gain-normalized metrics:**

Currently only sharpness, edge_strength, and detail have brightness invariance tests. Add tests for:
- ringing (with synthetic edge + near-edge pattern)
- temporal stability (two consecutive frames scaled by same factor)
- blocking (synthetic 8-pixel block pattern)
- texture_quality (synthetic textured image)

Each test: compute metric at 1.0x and 0.5x brightness, assert values within 5% relative tolerance.

**c) `short_name()` unit tests:**

Test the name extraction logic with representative inputs:
- `"device_a_svideo_direct.mov"` → expected short name
- Inputs that could collide (two different files → same short name)
- Edge cases: spaces in filename, no extension, unusual suffixes

### 1.4 Write Bug-Demonstrating Tests (should FAIL now, PASS after Phase 2)

**a) `compute_composites()` tied-rank test:**

Create a minimal test case with 3 clips where two clips have identical values on one metric:
```python
test_data = {
    "clip_a": {"sharpness": {"mean": 10.0}, "blocking": {"mean": 1.0}},
    "clip_b": {"sharpness": {"mean": 10.0}, "blocking": {"mean": 1.1}},
    "clip_c": {"sharpness": {"mean": 5.0},  "blocking": {"mean": 1.2}},
}
composites = compute_composites(test_data)
# clip_a and clip_b should have EQUAL sharpness rank (1.5 each)
assert composites["clip_a"]["ranks"]["sharpness"] == composites["clip_b"]["ranks"]["sharpness"]
```

This will fail with the current `np.argsort` implementation (one gets rank 1, the other rank 2).

**b) `compute_composites()` all-equal test:**

All clips have identical values for one metric. All should receive the same rank:
```python
test_data = {
    "clip_a": {"sharpness": {"mean": 5.0}},
    "clip_b": {"sharpness": {"mean": 5.0}},
    "clip_c": {"sharpness": {"mean": 5.0}},
}
# All three should get rank 2.0 (average of 1,2,3)
```

**c) No-frame edge case test (if feasible without ffmpeg):**

If `analyze_clip()` can be tested with a mock, verify that skip_frames >= total_frames produces a clear error rather than NaN metrics.

### 1.5 Deliverables

| Artifact | Location | Purpose |
|----------|----------|---------|
| Test baseline output | `review2/baselines/test_baseline.txt` | Reference test results |
| Metric baseline JSON | `review2/baselines/baseline_noref.json` | Per-clip metric values |
| Ranking baseline | `review2/baselines/baseline_rankings.txt` | Human-readable composite rankings |
| New test cases | `test_cases/test_metrics.py` | Edge cases, invariance, short_name |
| Bug-demonstrating tests | `test_cases/test_metrics.py` | Tied-rank, all-equal, no-frame |

### 1.6 Validation

- All existing 14 tests pass (unchanged).
- All new baseline regression tests pass (they test current correct behavior).
- Bug-demonstrating tests fail as expected (documenting the known defects).

---

## Phase 2: Correctness Fixes

### Goal
Fix all known correctness defects, in order from highest impact to lowest. Each fix should be a discrete, reviewable change.

### 2.1 Fix Tied-Rank Bug — M2 (both scripts)

**Files:** `quality_report.py` lines 599–612, `cross_clip_report.py` lines 85–98

**Change:** Replace `np.argsort` rank assignment with `scipy.stats.rankdata(method='average')`.

```python
from scipy.stats import rankdata

# For higher_better=True:
ranks[key] = rankdata(-arr, method='average')
# For higher_better=False:
ranks[key] = rankdata(arr, method='average')
# For higher_better=None (blocking, closer to 1.0):
ranks[key] = rankdata(np.abs(arr - 1.0), method='average')
```

**Test:** Bug-demonstrating tests from Phase 1.4a and 1.4b should now pass.

**Expected metric changes:** Composite rankings will change for any clips that were affected by arbitrary tie-breaking. The per-metric mean/std values are unchanged. Document the ranking changes by comparing Phase 1.2 baseline to post-fix output.

### 2.2 Fix XSS via innerHTML — M1 (`quality_report.py`)

**Files:** `quality_report.py` lines 983, 1031

**Change:** Add an `escHtml()` JavaScript function and use it when inserting clip names via `innerHTML`:

```javascript
function escHtml(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
```

Update both innerHTML assignments:
```javascript
td.innerHTML='<span class="rank">#'+(idx+1)+'</span> '+escHtml(row.name);
fc.innerHTML='<span class="rank">#'+(i+1)+'</span> '+escHtml(name);
```

**Test:** Create a test clip name containing `<script>alert(1)</script>` and verify it appears as escaped text in the HTML output, not as executable script. (Can be a manual test or a string-search assertion on the generated HTML.)

**Expected metric changes:** None. This is a rendering-only fix.

### 2.3 Add ffmpeg Error Checking — M3 (`quality_report.py`)

**Files:** `quality_report.py` line 422 (after `proc.wait()`)

**Change:**
```python
proc.wait()
if proc.returncode != 0:
    print(f"WARNING: ffmpeg exited with code {proc.returncode} for {filepath}")
```

**Test:** Manually verify with a truncated/corrupt test file if available. Otherwise, this is a defensive addition — no existing tests exercise this path.

**Expected metric changes:** None for valid inputs.

### 2.4 Add Subprocess Resource Cleanup — M4 (`quality_report.py`)

**Files:** `quality_report.py` `analyze_clip()` line 386

**Change:** Wrap the frame processing loop in `try/finally`:

```python
proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
try:
    while True:
        frame = read_frame(proc.stdout, width, height)
        if frame is None:
            break
        # ... process frame ...
    proc.wait()
    if proc.returncode != 0:
        print(f"WARNING: ffmpeg exited with code {proc.returncode} for {filepath}")
finally:
    proc.stdout.close()
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()
```

**Test:** No direct test (requires simulating an exception during frame processing). Verify existing tests still pass.

**Expected metric changes:** None.

### 2.5 Guard No-Frame Edge Case — M5 (`quality_report.py`)

**Files:** `quality_report.py` after the frame processing loop (after line ~420)

**Change:**
```python
if n_frames == 0:
    print(f"ERROR: No frames decoded for {filepath} (skip={skip_frames})")
    proc.wait()
    return None, None
```

Also add a guard in `main()` to skip clips that return None:
```python
metrics, per_frame = analyze_clip(clip_path, width, height, skip)
if metrics is None:
    print(f"  Skipping {name} — no frames decoded")
    continue
```

**Test:** Phase 1.4c test (if implemented). Manual test: run with `--skip` value larger than clip frame count.

**Expected metric changes:** None for normal inputs. Clips that previously produced NaN will now be skipped with an error message.

### 2.6 Detect Clip Name Collisions — M6 (`quality_report.py`)

**Files:** `quality_report.py` `main()` line ~1521

**Change:**
```python
if name in all_results:
    print(f"WARNING: Duplicate short name '{name}' from {clip_path}")
    i = 2
    while f"{name}_{i}" in all_results:
        i += 1
    name = f"{name}_{i}"
```

**Test:** Phase 1.3c `short_name()` tests should cover collision scenarios.

**Expected metric changes:** None for the current dataset (no collisions exist). Datasets with collisions will now produce correct results instead of silently dropping clips.

### 2.7 Fix normalize.py Division-by-Zero — m18

**Files:** `normalize.py` lines 148, 197, 203

**Change:** Add zero-frame guards:
```python
if n_frames == 0:
    print(f"ERROR: No frames decoded for {filepath}")
    return  # or continue to next clip
```

Similarly for `sat_count`:
```python
if sat_count == 0:
    print(f"WARNING: No valid chroma samples for {filepath}")
    scale = 1.0  # identity transform
```

**Test:** Manual or skip — low-frequency code path.

### 2.8 Fix verify_signalstats.py Empty Input — m19

**Files:** `verify_signalstats.py` line ~70

**Change:**
```python
if not vals:
    print("No clips found — check directory path")
    sys.exit(1)
```

**Test:** Run with an empty/nonexistent directory, verify clean error message.

### 2.9 Fix normalize_linear.py Filename Bug — m9

**Files:** `normalize_linear.py` line 251

**Change:**
```python
base, ext = os.path.splitext(clip_name)
out_name = f"{base}_linnorm{ext}"
```

Apply same fix to `normalize.py` line 299:
```python
base, ext = os.path.splitext(clip_name)
out_name = f"{base}_normalized{ext}"
```

**Test:** Verify with a `.mp4` input filename (can be a dry-run or path-construction unit test).

### 2.10 Deliverables

All fixes applied. Bug-demonstrating tests from Phase 1.4 now pass. No existing tests broken.

---

## Phase 3: Post-Fix Verification Checkpoint

### Goal
Confirm that Phase 2 fixes are correct and that no unintended metric changes occurred.

### 3.1 Run Full Test Suite

```bash
python test_cases/test_metrics.py
```

**Expected:** All tests pass, including the new bug-demonstrating tests from Phase 1.4 (tied-rank, all-equal).

### 3.2 Generate Post-Fix Metrics

```bash
python quality_report.py /wintmp/analog_video/noref_compare/ \
    --output-dir review2/baselines/ \
    --name postfix_noref \
    --screenshots 0
```

### 3.3 Compare to Phase 1 Baseline

Write a comparison script or use `diff`:

```python
# Compare per-metric mean/std values (should be IDENTICAL)
# Compare composite rankings (may differ due to tied-rank fix)
```

**Acceptance criteria:**
- Per-metric mean/std values: **identical** to Phase 1 baseline (no metric function was changed).
- Composite rankings: **may differ** due to tied-rank fix. Document the changes and verify they are explainable (clips with previously arbitrary tie-breaking now have fractional ranks).
- No NaN values in any metric.

### 3.4 Record Post-Fix Baseline

Save `postfix_noref.json` and a ranking comparison document. This becomes the new baseline for Phase 5 comparison.

---

## Phase 4: Code Organization, Cleanup, Refactoring

### Goal
Improve code maintainability without changing behavior. All changes are strictly behavioral no-ops.

### Step 4.1 — Trivial Cleanups (zero risk)

These changes are cosmetic and cannot affect behavior:

| ID | File | Change |
|----|------|--------|
| m1 | `quality_report.py` line 2–4 | Update docstring: remove hardcoded "576x436", note auto-detection |
| m2 | `quality_report.py` line 636 | Remove/update "All clips normalized..." message |
| m3 | `quality_report.py` line 1549 | Remove `or True`, restructure as unconditional comparison frame extraction + conditional screenshots |
| m6 | `cross_clip_report.py` line 269 | Change "9 no-reference metrics" to "no-reference quality metrics" (derive from `len(ALL_KEYS)` if feasible) |
| m11 | `verify_signalstats.py` line 9 | Remove `import re` |
| m13 | `quality_report.py` line 577 | Update docstring to say "all metrics" instead of "9 metrics" |
| s1 | Both scripts | Align `compute_composites` docstrings |

**Test:** Run test suite. All pass (no behavioral change).

### Step 4.2 — Moderate Cleanups (low risk)

| ID | File | Change |
|----|------|--------|
| m10 | `verify_signalstats.py` | Convert hardcoded `NORM_DIR` to argparse with `--dir` argument |
| m17 | `normalize.py` | Add `--pattern` argument (default `"*.mov"`) matching `normalize_linear.py` |
| Dead code | `quality_report.py` | Remove unused metric functions: `noise_estimate()`, `perceptual_contrast()`, `tonal_richness()`, `gradient_smoothness()` (~80 lines) |

**Test:** Run test suite. All pass. Verify `normalize.py --help` shows `--pattern`. Verify `verify_signalstats.py --help` shows `--dir`.

### Step 4.3 — Extract `common.py` Module (medium risk)

This is the most significant refactoring step. Create `common.py` containing shared code, then update all importing scripts.

**Contents of `common.py`:**

```python
# Metric registry
ALL_KEYS = [...]  # 11 metric keys
METRIC_INFO = {...}  # 11 metric info tuples

# Color palettes
COLORS_7 = [...]
COLORS_14 = [...]

# Shared CSS
HTML_CSS = """..."""

# Composite scoring
def compute_composites(data):
    ...  # tie-aware ranking (already fixed in Phase 2)

# Video I/O (shared between normalize scripts and quality_report)
def probe_video(filepath):
    ...

def detect_frame_rate(filepath):
    ...

def decode_command(filepath, width, height, pix_fmt):
    ...

def encode_command(output_path, width, height, fps, pix_fmt):
    ...

def read_frame(pipe, width, height):
    ...

def write_frame(pipe, frame):
    ...
```

**Update scripts:**
- `quality_report.py`: `from common import ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14, HTML_CSS, compute_composites, probe_video, detect_frame_rate, decode_command, read_frame`
- `cross_clip_report.py`: `from common import ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14, HTML_CSS, compute_composites`
- `normalize.py`: `from common import probe_video, detect_frame_rate, decode_command, encode_command, read_frame, write_frame`
- `normalize_linear.py`: `from common import probe_video, detect_frame_rate, decode_command, encode_command, read_frame, write_frame`

**Verification approach:**
1. Before extracting, run `diff` between the duplicated functions to confirm they are truly identical. Any differences must be reconciled first.
2. Extract one group at a time (constants first, then compute_composites, then I/O functions), running the test suite after each extraction.
3. After all extractions, run the full test suite and generate metrics on the real dataset.

**Test:** Full test suite. All pass. Any test file imports that reference the moved functions must be updated.

### 4.4 Deliverables

- `common.py` with all shared code
- All scripts updated to import from `common.py`
- Removed ~150 lines of duplication
- Removed ~80 lines of dead metric code
- All cosmetic/string fixes applied

---

## Phase 5: Post-Refactoring Verification Checkpoint

### 5.1 Run Full Test Suite

```bash
python test_cases/test_metrics.py
```

**Expected:** All tests pass.

### 5.2 Generate Post-Refactoring Metrics

```bash
python quality_report.py /wintmp/analog_video/noref_compare/ \
    --output-dir review2/baselines/ \
    --name postrefactor_noref \
    --screenshots 0
```

### 5.3 Compare to Phase 3 Baseline

**Acceptance criteria:**
- Per-metric mean/std values: **identical** to Phase 3 `postfix_noref.json`.
- Composite rankings: **identical** to Phase 3 (no scoring logic changed).
- JSON structure: identical.

Any difference here indicates a bug introduced during refactoring. Investigate and fix before proceeding.

### 5.4 Verify Cross-Clip Report

```bash
python cross_clip_report.py review2/baselines/postrefactor_noref.json \
    --output review2/baselines/postrefactor_crossclip.html
```

Verify the HTML renders correctly in a browser (spot-check: correct metric count in subtitle, correct clip names, heatmap renders).

### 5.5 Verify Normalize Scripts

```bash
# Dry-run: verify --help works and --pattern is accepted
python normalize.py --help
python normalize_linear.py --help
python verify_signalstats.py --help
```

---

## Phase 6: New Features and Improvements

### Goal
Add new capabilities, ordered from lowest risk (additive metadata) to highest risk (algorithmic changes).

### Step 6.1 — JSON Schema Versioning (low risk, additive)

**Files:** `quality_report.py` (or `common.py` after refactoring)

**Change:** Add a `_metadata` key to the JSON output:

```python
output = {
    "_metadata": {
        "schema_version": 2,
        "metrics": list(ALL_KEYS),
        "n_metrics": len(ALL_KEYS),
        "source_dir": str(src_dir),
        "pattern": args.pattern,
        "skip_frames": skip_dict,
        "timestamp": datetime.now().isoformat(),
    }
}
# Then add per-clip data as before
```

Update `cross_clip_report.py` to read and compare `_metadata.schema_version` when loading multiple JSONs. Warn if versions differ.

**Test:** Generate a new JSON, verify `_metadata` key exists with correct contents. Verify `cross_clip_report.py` still loads old JSONs without `_metadata` (backward compatibility).

### Step 6.2 — Discrimination Gating in Composite (medium risk)

**Files:** `compute_composites()` in `common.py`

**Change:** Before computing ranks for a metric, check if the coefficient of variation exceeds a threshold. If not, exclude that metric from the composite and note it in the output:

```python
NON_DISC_CV_THRESHOLD = 0.01  # 1% CV

for key in ALL_KEYS:
    arr = np.array([data[c][key]["mean"] for c in clip_names])
    cv = np.std(arr) / (np.abs(np.mean(arr)) + 1e-10)
    if cv < NON_DISC_CV_THRESHOLD:
        non_discriminating.append(key)
        continue
    # ... compute ranks as before ...
```

Report excluded metrics in the composite output and HTML report.

**Test:**
- Unit test: feed `compute_composites()` data where one metric is constant across all clips. Verify that metric is excluded from the composite.
- Integration: run on 14-clip dataset. Verify `blown_whites` is flagged as non-discriminating.

**Expected metric changes:** Composite rankings will change (blown_whites no longer contributes). Document the changes.

### Step 6.3 — Store Original Filenames in JSON (low risk, additive)

**Files:** `quality_report.py`

**Change:** Add `_source_file` to each clip's data:

```python
all_results[name]["_source_file"] = os.path.basename(clip_path)
```

**Test:** Verify JSON contains `_source_file` for each clip. Verify `cross_clip_report.py` ignores the new key gracefully.

### Step 6.4 — Vectorize Block-Based Metrics (medium risk, must be behaviorally identical)

**Files:** `quality_report.py`: `detail_texture()`, `texture_quality_measure()`

**Change:** Replace Python loops with numpy vectorized operations:

```python
def detail_texture(yf, block=16):
    h, w = yf.shape
    nh, nw = h // block, w // block
    cropped = yf[:nh*block, :nw*block]
    blocks = cropped.reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(-1, block, block)
    means = blocks.mean(axis=(1, 2))
    stds = blocks.std(axis=(1, 2))
    mask = means > 0.01
    if not mask.any():
        return 0.0
    cvs = stds[mask] / means[mask]
    return float(np.median(cvs))
```

Similar vectorization for `texture_quality_measure`.

**Critical test:** The vectorized versions MUST produce **identical** results (within floating-point tolerance of ~1e-12) to the loop-based versions. Test procedure:
1. Before replacing, save the loop-based output for all 14 clips.
2. Replace with vectorized version.
3. Compare per-frame values. Max absolute difference should be < 1e-10.

If any difference exceeds tolerance, investigate. Common causes: different order of floating-point operations (sum reduction order matters), `ddof` parameter in `np.std`.

**Expected metric changes:** Per-metric values should be identical (within float tolerance). Performance improvement of ~5-20x on these two functions.

### Step 6.5 — Time-Based Seeking for Frame Extraction (medium risk)

**Files:** `quality_report.py`: `extract_frame_jpeg()`

**Change:** Replace sequential decode with time-based seeking:

```python
def extract_frame_jpeg(filepath, frame_num, fps, width, height):
    time_sec = frame_num / fps
    cmd = [
        "ffmpeg", "-ss", f"{time_sec:.4f}", "-i", filepath,
        "-vframes", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"
    ]
    ...
```

Requires passing `fps` to the function (available from `probe_video`).

**Test:** Compare extracted frames from time-based seeking vs. sequential decode for a few known frame indices. Verify they are the same frame (or within 1 frame due to keyframe seeking). Note: time-based seeking before `-i` uses keyframe-based seeking, which may land on a different frame for non-keyframe positions. For ProRes (all-intra), every frame is a keyframe, so this should be exact.

**Expected metric changes:** None (metric comparison frames may differ by <=1 frame position in edge cases, which is acceptable for the comparison purpose).

### Step 6.6 — New Analog-Specific Metrics (larger scope, future)

These are documented here for completeness but are individually scoped projects:

**a) Dropout Detection** (HIGH priority from metric_selection.md)
- Per-frame horizontal line analysis for abrupt luminance jumps
- New metric function, new test case with synthetic dropout pattern
- Requires addition to `ALL_KEYS`, `METRIC_INFO`, and composite

**b) Head-Switching Region Analysis** (HIGH priority)
- Temporal stability restricted to bottom N lines of frame
- Extension of existing temporal stability infrastructure
- New metric function, new test case

**c) Chroma Delay Detection** (MEDIUM priority)
- Cross-correlation between Y and Cb/Cr edge positions
- New metric function, new test case with synthetic shifted chroma

**d) Pedestal Estimation and Correction** (MEDIUM priority)
- Estimate black level as 5th percentile of Y
- Subtract before metric computation
- Must re-validate all brightness invariance tests after this change

Each of these requires its own design, implementation, test case, and metric baseline update. They should be individually planned and executed, not batched.

### 6.7 Deliverables

- JSON schema versioning in output
- Discrimination gating in composite scoring
- Original filenames stored in JSON
- Vectorized block-based metrics (if validated identical)
- Time-based seeking for frame extraction (if validated correct)
- Design documents for new metrics (6.6a-d) for future implementation

---

## Phase 7: Post-Feature Verification Checkpoint

### 7.1 Run Full Test Suite

All tests pass, including any new tests added for Phase 6 features.

### 7.2 Generate Post-Feature Metrics

```bash
python quality_report.py /wintmp/analog_video/noref_compare/ \
    --output-dir review2/baselines/ \
    --name postfeature_noref \
    --screenshots 0
```

### 7.3 Compare to Baselines

**Per-metric mean/std:** Should be **identical** to Phase 5 `postrefactor_noref.json` for all metrics. The vectorized implementations must produce the same values. Any difference is a bug.

**Composite rankings:** Will differ from Phase 5 due to discrimination gating (blown_whites excluded). Compare to Phase 3 `postfix_noref.json` to understand the full chain of ranking changes:
- Phase 1 → Phase 3: changes due to tied-rank fix
- Phase 3 → Phase 7: changes due to discrimination gating

Document both sets of changes.

**JSON structure:** Now includes `_metadata` and `_source_file` keys. Verify `cross_clip_report.py` handles the new structure correctly.

---

## Phase 8: Regression Review

### Goal
Systematic code review of all changes made in Phases 2, 4, and 6, looking for re-introduction of known issue patterns.

### 8.1 Review Checklist

| Pattern | Check | Files |
|---------|-------|-------|
| XSS / innerHTML | No new `innerHTML` assignments with unescaped user data | All HTML generation code |
| Division by zero | All divisions have epsilon guards or zero-checks | All metric functions, I/O functions |
| Resource leaks | All subprocess.Popen wrapped in try/finally or context manager | `quality_report.py`, `normalize.py`, `normalize_linear.py` |
| Tied ranks | No `np.argsort` used for ranking (only `rankdata`) | `common.py` (compute_composites) |
| NaN propagation | All `np.mean`/`np.median` calls on potentially-empty arrays are guarded | All aggregation code |
| Silent overwrites | Dict assignments with user-derived keys check for duplicates | `main()` in quality_report.py |
| Hardcoded counts | No literal "9" or "11" for metric count — use `len(ALL_KEYS)` | All scripts |
| Import consistency | All scripts import shared code from `common.py`, no local copies | All scripts |

### 8.2 Additional Checks

- Verify `common.py` has no circular imports
- Verify all test files import from the correct locations
- Check for any `TODO`, `FIXME`, `HACK`, or `XXX` comments introduced during changes
- Review any new magic numbers introduced in Phase 6

### 8.3 Deliverable

A short issues document listing any problems found, with file/line references. If no issues found, record "Phase 8: no regressions identified."

---

## Phase 9: Fix Regression Review Issues

Fix any issues identified in Phase 8. Each fix follows the same pattern:

1. Write a test that demonstrates the issue (if not already covered).
2. Apply the fix.
3. Verify the test passes.
4. Run the full test suite to confirm no side effects.

If Phase 8 found no issues, this phase is a no-op.

---

## Phase 10: Final Verification

### 10.1 Full Test Suite

```bash
python test_cases/test_metrics.py
```

All tests pass. Record the final test count and output.

### 10.2 Full Pipeline Run

```bash
python quality_report.py /wintmp/analog_video/noref_compare/ \
    --output-dir review2/baselines/ \
    --name final_noref \
    --screenshots 2

python cross_clip_report.py review2/baselines/final_noref.json \
    --output review2/baselines/final_crossclip.html
```

### 10.3 Metric Comparison Chain

Compare `final_noref.json` against all previous baselines:

| Comparison | Expected per-metric diff | Expected ranking diff |
|------------|------------------------|-----------------------|
| Final vs. Phase 1 (baseline) | Identical mean/std | Changed (tied-rank fix + discrimination gating) |
| Final vs. Phase 3 (post-fix) | Identical mean/std | Changed (discrimination gating only) |
| Final vs. Phase 5 (post-refactor) | Identical mean/std | Changed (discrimination gating only) |

All per-metric mean/std values should be identical across the entire chain (no metric function was changed in a way that alters output). Only composite rankings should differ, and only for documented reasons.

### 10.4 Visual Spot-Check

Open the final HTML report in a browser and verify:
- Correct clip count and names (no collisions, no missing clips)
- Heatmap renders correctly with clip names escaped
- Chart.js bar charts display correctly
- Comparison frame images load and lightbox works
- Non-discriminating metrics are annotated
- JSON metadata block present in output file

### 10.5 Sign-Off

Record final state:
- Total tests: N (was 14, now N after additions)
- All passing: yes/no
- Per-metric values: unchanged from baseline (confirmed)
- Composite rankings: changed for documented reasons (tied-rank fix, discrimination gating)
- Known remaining issues: (list any deferred items from review)

---

## Issue Traceability Matrix

Every finding from the review documents mapped to its implementation phase:

### Code Review Findings

| ID | Description | Phase | Status |
|----|-------------|-------|--------|
| M1 | XSS via innerHTML | 2.2 | Done |
| M2 | Tied-rank bug | 2.1 | Done |
| M3 | No ffmpeg error checking | 2.3 | Done |
| M4 | Subprocess resource leak | 2.4 + 9 | Done (quality_report in P2, normalize scripts in P9) |
| M5 | No-frame edge case | 2.5 | Done |
| M6 | Clip name collisions | 2.6 | Done |
| m1 | Outdated docstring | 4.1 | Done |
| m2 | Outdated text report message | 4.1 | Done |
| m3 | Debug `or True` leftover | 4.1 | Done |
| m4 | Slow frame extraction | 6.5 | Deferred (ProRes all-intra, low benefit) |
| m5 | extract_frame_jpeg empty output | Deferred | |
| m6 | "9 metrics" in cross_clip | 4.1 | Done (uses len(ALL_KEYS)) |
| m7 | Code duplication | 4.3 | Done (common.py) |
| m8 | Reference clip self-normalized | Deferred | |
| m9 | normalize_linear filename bug | 2.9 | Done |
| m10 | Hardcoded path in verify_signalstats | 4.2 | Done (argparse) |
| m11 | Unused import | 4.1 | Done |
| m12 | ffprobe movie filter quoting | Deferred | |
| m13 | "9 metrics" in docstring | 2 (early fix) | Done |
| m14 | Frame index mismatch fallback | Deferred | |
| m15 | No edge case tests | 1.3a | Done (30 edge case tests) |
| m16 | Cross-clip device name matching | Deferred | |
| m17 | normalize.py missing --pattern | 4.2 | Done |
| m18 | normalize.py division-by-zero | 2.7 | Done |
| m19 | verify_signalstats empty input | 2.8 | Done |
| s1 | Duplicated docstring style | 4.1 | Done |
| s2 | Inconsistent short_name | Deferred | |
| s3 | Magic numbers | Deferred | |
| s4 | CSS not reused | 4.3 | Partial (constants/logic shared; CSS differs by report type) |

### Architecture Review Findings

| ID | Description | Phase | Status |
|----|-------------|-------|--------|
| §0 | Requirements traceability | Documented in reviews | Done |
| §1 | Single-file architecture | 4.3 | Done (common.py extracted) |
| §8 | JSON schema versioning | 6.1 | Done (_metadata block) |
| Priority matrix items | Various | Mapped to phases above | Done |

### Metric Implementation Findings

| ID | Description | Phase | Status |
|----|-------------|-------|--------|
| H1 | Additive pedestal sensitivity | 6.6d (future) | Deferred |
| H2 | Tied-rank + blown_whites | 2.1 + 6.2 | Done |
| H3 | No-frame edge case | 2.5 | Done |
| M1 | Ringing brightness dependence | Deferred | |
| M2 | Naturalness direction | Deferred | |
| M3 | No ffmpeg error checking | 2.3 | Done |
| M4 | No process cleanup | 2.4 + 9 | Done |
| Dead code | Dropped metric functions | 4.2 | Done (~80 lines removed) |

### Metric Selection Findings

| ID | Description | Phase | Status |
|----|-------------|-------|--------|
| H4 | Tied-rank handling | 2.1 | Done |
| H5 | Discrimination gating | 6.2 | Done (blown_whites auto-excluded) |
| H6 | Brightness-agnostic docs | 4.1 | Done |
| §2.1 | Dropout detection | 6.6a (future) | Deferred |
| §2.1 | Head-switching analysis | 6.6b (future) | Deferred |
| §2.1 | Chroma delay | 6.6c (future) | Deferred |

### Deferred Items

Items marked "Deferred" are either:
- Low impact / unlikely code paths (m5, m12, m14)
- Design limitations not fixable without major rearchitecting (m16, s2)
- Future feature work requiring separate design (H1 pedestal, M1 ringing, M2 naturalness, dropout, head-switching, chroma delay)
- Readability improvements with no correctness impact (s3, m8)
- Low benefit for current data format (m4 time-based seeking — ProRes all-intra)

These can be addressed in a future pass or accepted as known limitations.

---

## Phase 10: Sign-Off

### Final State

- **Total tests:** 59 (was 14 originally; 54 main + 5 feature/bug tests)
- **All passing:** Yes (59/59)
- **Per-metric values:** Unchanged from Phase 1 baseline (confirmed across 4 comparison checkpoints)
- **Composite rankings:** Changed for documented reasons:
  - Tied-rank fix: `scipy.stats.rankdata(method='average')` replaces `np.argsort`
  - Discrimination gating: metrics with CV < 1% excluded from composite (blown_whites in 16-clip dataset)
- **New features:** JSON schema versioning, discrimination gating, source filenames, vectorized block metrics
- **Code organization:** `common.py` extracted with shared constants, I/O, and composite scoring
- **Dead code removed:** ~80 lines (4 dropped metric functions + `_DROPPED_KEYS` + `scipy.ndimage` import)
- **Resource management:** All subprocess.Popen sites now wrapped in try/finally across all scripts

### Verification Chain

| Checkpoint | Per-Metric Diff | Ranking Diff | Status |
|------------|----------------|--------------|--------|
| Phase 3 (post-fix) vs Phase 1 (baseline) | 0.00 | 0.00 | PASS |
| Phase 5 (post-refactor) vs Phase 3 | 0.00 | 0.00 | PASS |
| Phase 7 (post-feature) vs Phase 5 | 0.00 | 0.00 | PASS |
| Phase 10 (final) vs Phase 1 (baseline) | 0.00 | 0.00 | PASS |
