# Video Compare Codebase Cleanup — Design

**Date:** 2026-05-09
**Author:** brainstorming session

## Goal

Reduce code duplication, decompose the 1832-line `quality_report.py`, fix stale documentation, and remove repository cruft. The behavior of every CLI must remain identical — this is a refactor + hygiene pass, not a feature change. The synthetic test suite (`test_cases/test_metrics.py`) is the regression net and must pass at the end.

## Scope (in)

1. **Architecture refactor** — split `quality_report.py` into three modules:
   - `common.py` — keep existing helpers; add `parse_skip_args()`.
   - `metrics.py` (new) — all metric functions, `analyze_clip`, `add_detail_perceptual_metric`, `short_name`, related constants (`DERIVED_KEYS`, `EXTRA_DETAIL_KEYS`, `DETAIL_PERCEPTUAL_KEY`, `DETAIL_PERCEPTUAL_DEPS`, `PCT_KEYS`, `parse_metric_csv`).
   - `html_report.py` (new) — `HTML_CSS`, `generate_html`, `format_text_report`, `find_comparison_frames`, `extract_sample_screenshots`, `extract_frame_jpeg`, `fmt_metric_value`, `short_metric_header`, `METRIC_GUIDE_TEXT`, `CMP_ICON_SVG`.
   - `quality_report.py` — only the CLI orchestration (`main`, `load_and_merge_jsons`).
   - `cross_clip_report.py` — import shared CSS from `html_report.py`.
   - `quality_metrics.py` — import only from `common.py` and `metrics.py`. Stop reaching into `quality_report.py`.

2. **Documentation fixes**
   - Update `quality_report.py` module docstring: "11 metrics" → "14 metrics by default (11 core + 3 extra detail)".
   - Update `test_cases/README.md`: "11 quality metrics" → "11 core quality metrics".
   - Add a one-line clarifying comment near `EXTRA_DETAIL_KEYS` explaining `detail_tenengrad` is available but not on by default.

3. **Bug/anti-pattern fixes**
   - Remove the redundant inner `proc.wait()` in `normalize.py` `compute_reference_stats` and `normalize_clip` (the `finally` block already waits).
   - Move `--skip` parsing into `common.parse_skip_args()` and use it from both `quality_report.py` and `quality_metrics.py`.

4. **Repo hygiene**
   - Move `review2/` → `docs/archive/review2/`.
   - Delete `example_report/index.html` (27 MB) from the main branch. Keep `example_report/index.json`. Add `example_report/*.html` to `.gitignore`. The README's GitHub Pages link continues to work (Pages serves from a separate branch).
   - Delete the local `.playwright-mcp/` directory (already gitignored, just stale).
   - Update `.gitignore`:
     - `**/__pycache__/` (cover subdirs)
     - `*.html` and `*.json` excluding committed example: actually, simpler approach — add `*_quality_report*.html`, `*_quality_report*.json`, `*_quality_metrics*.json`, `comparison.html` to keep generated outputs out.
     - `example_report/*.html`

## Scope (out)

- No metric algorithm changes.
- No CLI flag changes (signatures unchanged; help text may be tweaked for accuracy).
- No JSON schema changes.
- `verify_signalstats.py` left as-is (legacy diagnostic; not worth touching).
- `normalize.py` / `normalize_linear.py` not deduplicated beyond removing the double-wait bug.

## Architecture After Refactor

```
common.py          — constants (ALL_KEYS, METRIC_INFO, COLORS_*),
                     compute_composites, ffprobe/decode/encode helpers,
                     read_frame/write_frame, parse_skip_args (NEW)

metrics.py (NEW)   — pure metric functions (sharpness_laplacian etc.),
                     analyze_clip, add_detail_perceptual_metric,
                     short_name, parse_metric_csv,
                     DERIVED_KEYS, EXTRA_DETAIL_KEYS, DETAIL_PERCEPTUAL_*,
                     PCT_KEYS, METRIC_GUIDE_TEXT (used in HTML help text only — see note),
                     to_float, _zscore_with_params

html_report.py (NEW) — HTML_CSS, CMP_ICON_SVG,
                       generate_html, format_text_report,
                       find_comparison_frames, extract_sample_screenshots,
                       extract_frame_jpeg, fmt_metric_value,
                       short_metric_header

quality_report.py    — argparse main + load_and_merge_jsons.
                       Imports analyze_clip, short_name etc. from metrics;
                       imports generate_html etc. from html_report.

quality_metrics.py   — argparse main only. Imports from metrics + common.

cross_clip_report.py — imports HTML_CSS from html_report (drops local copy).
```

`METRIC_GUIDE_TEXT` lives in `html_report.py` (it is presentation, not metric definition). `PCT_KEYS` and `fmt_metric_value` go in `html_report.py` too — they're formatting concerns.

Wait, `PCT_KEYS` is used inside `analyze_clip` (the EPS clamp) — that's a metric concern. Keep `PCT_KEYS` in `metrics.py`; `fmt_metric_value` (which also uses it) stays in `html_report.py` and imports the set.

## Test Strategy

The synthetic test suite is the spine. Workflow:
1. **Capture baseline** — run `python test_cases/test_metrics.py`, save the output.
2. **Refactor in steps** — see below. After each step, run the test suite. Output must match baseline.
3. **Final smoke test** — run the test suite once more after all changes.

The user's interactive workflows (running quality_report.py against actual clips) cannot be tested without the data on `/wintmp/`, so we rely on the synthetic suite plus visual inspection of help text.

## Implementation Steps

In strict order so each step is independently testable:

1. Add `parse_skip_args` to `common.py`. Use it in both scripts. Run tests.
2. Create `metrics.py`. Move metric functions, `analyze_clip`, `short_name`, `add_detail_perceptual_metric`, related constants. Update `quality_report.py` and `quality_metrics.py` and `test_cases/test_metrics.py` imports. Run tests.
3. Create `html_report.py`. Move `HTML_CSS`, `generate_html`, `format_text_report`, `find_comparison_frames`, `extract_sample_screenshots`, `extract_frame_jpeg`, `fmt_metric_value`, `short_metric_header`, `METRIC_GUIDE_TEXT`, `CMP_ICON_SVG`. Update `quality_report.py` imports. Run tests.
4. Update `cross_clip_report.py` to import `HTML_CSS` from `html_report.py`; delete its local copy. Run a smoke import test.
5. Fix `normalize.py` double-wait bug.
6. Update docstrings in `quality_report.py` (11→14) and `test_cases/README.md`.
7. Add comment near `EXTRA_DETAIL_KEYS`.
8. Move `review2/` to `docs/archive/review2/`. Update `.gitignore`.
9. Delete `example_report/index.html`. Add to `.gitignore`. Delete `.playwright-mcp/` (local, already gitignored).
10. Run the test suite one more time as a final check.

## Risks

- **Circular imports.** `html_report.py` will need `compute_composites` (from `common.py`), `METRIC_INFO`/`ALL_KEYS` (from `common.py`), and the `PCT_KEYS` set (from `metrics.py`). `metrics.py` imports from `common.py`. So the dep graph is `common ← metrics ← html_report`, which is acyclic. `quality_report.py` and `quality_metrics.py` are leaves.
- **Hidden inter-function references.** `add_detail_perceptual_metric` references `DETAIL_PERCEPTUAL_KEY` and `DETAIL_PERCEPTUAL_DEPS`; both move with it. `analyze_clip` references `DERIVED_KEYS`, `PCT_KEYS`, and all metric functions; all move with it.
- **Test import path.** `test_cases/test_metrics.py` does `from quality_report import sharpness_laplacian, ...`. Update to `from metrics import ...`. This is the only test-suite touch.
- **Loss of git history attribution on moved code.** Not really a risk — `git log --follow` still works. We may use `git mv` for `review2/` so history follows.

## Deliverables

- `metrics.py`, `html_report.py` created.
- `quality_report.py` reduced to ~600 lines (CLI + load_and_merge_jsons + thin orchestration).
- `quality_metrics.py` no longer imports from `quality_report.py`.
- `cross_clip_report.py` no longer carries its own CSS copy.
- `common.py` gains `parse_skip_args`.
- `normalize.py` no longer has redundant `proc.wait()`.
- Docstrings/READMEs updated.
- `docs/archive/review2/` exists; root `review2/` does not.
- `example_report/index.html` removed; `.playwright-mcp/` removed.
- `.gitignore` covers subdir caches and generated reports.
- Synthetic test suite passes with the same numbers as baseline.
