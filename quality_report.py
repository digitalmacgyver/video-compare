#!/usr/bin/env python3
"""No-reference video quality metrics for analog video capture comparison.

Analyzes a directory of video clips (resolution auto-detected via ffprobe)
and produces JSON data, an interactive HTML report with Chart.js
visualizations, and optionally a plain-text report.

Usage:
    python quality_report.py <src_dir> [options]

    src_dir             Directory containing normalized .mov files to analyze

Options:
    --output-dir DIR    Where to write reports (default: same as src_dir)
    --name PREFIX       Report filename prefix (default: "quality_report")
    --pattern GLOB      File glob pattern (default: "*.mov")
    --text              Also generate a plain-text report

Examples:
    python quality_report.py /path/to/normalized/
    python quality_report.py /path/to/normalized/ --output-dir /reports/ --name clip02
    python quality_report.py /path/to/normalized/ --pattern "*_sls.mp4" --name sls_report
    python quality_report.py /path/to/normalized/ --text

Metrics (14 by default — 11 core + 3 extra detail; all brightness-agnostic):
  Core: sharpness, edge strength, blocking, detail, texture quality, ringing,
        temporal stability, colorfulness, naturalness, crushed blacks, blown whites
  Extra detail (default on): detail_perceptual (VHS-SD), detail_blur_inv, detail_sml
  Extra detail (opt-in via --extra-detail-metrics): detail_tenengrad

Brightness-sensitive metrics (sharpness, edge strength, detail, ringing, temporal stability) are
normalized by mean Y to eliminate dependence on capture brightness / gain settings.  No upstream
brightness normalization is required.

Dropped metrics: noise (subordinate to detail — correlated r=0.83), contrast and tonal richness
(near-zero discrimination across clips), gradient smoothness (r=0.94 duplicate of texture quality).
Ringing retained despite r=0.94 correlation with sharpness — it measures a distinct analog artifact
(edge overshoot from VCR sharpness circuits / aperture correction) that varies independently of
true edge definition across capture hardware.
"""

import argparse
import os
import sys
import glob
import json
from datetime import datetime

from common import ALL_KEYS, METRIC_INFO, parse_skip_args, probe_video
from metrics import (
    DETAIL_PERCEPTUAL_KEY, DETAIL_PERCEPTUAL_DEPS, EXTRA_DETAIL_KEYS,
    parse_metric_csv, short_name, analyze_clip, add_detail_perceptual_metric,
)
from html_report import generate_html, format_text_report, find_comparison_frames, extract_sample_screenshots


# =====================================================================
# MAIN
# =====================================================================

def load_and_merge_jsons(json_paths):
    """Load one or more quality metric JSON files and merge clip data.

    Returns (merged_data, merged_metric_keys) where merged_data is a dict
    of clip_name -> metric_dict (no _metadata keys), and merged_metric_keys
    is the intersection of metric keys across all files.
    """
    merged = {}
    all_metric_sets = []
    source_jsons = []

    for jp in json_paths:
        if not os.path.exists(jp):
            print(f"ERROR: File not found: {jp}")
            sys.exit(1)
        with open(jp) as f:
            raw = json.load(f)

        meta = raw.get("_metadata", {})
        metrics_in_file = set(meta.get("metrics", []))
        if metrics_in_file:
            all_metric_sets.append(metrics_in_file)
        source_jsons.append(os.path.basename(jp))

        clip_count = 0
        for key, val in raw.items():
            if key.startswith("_"):
                continue
            if key in merged:
                print(f"WARNING: Duplicate clip '{key}' found in {jp}, overwriting previous")
            merged[key] = val
            clip_count += 1

        print(f"  Loaded {clip_count} clip(s) from {os.path.basename(jp)}")

    if not merged:
        print("ERROR: No clip data found in any JSON file")
        sys.exit(1)

    # Metric keys: use intersection so all clips have all reported metrics
    if all_metric_sets:
        common_metrics = all_metric_sets[0]
        for ms in all_metric_sets[1:]:
            common_metrics = common_metrics & ms
        # Preserve a sensible order: ALL_KEYS first, then extras alphabetically
        ordered = [k for k in ALL_KEYS if k in common_metrics]
        extras = sorted(common_metrics - set(ordered))
        merged_metric_keys = ordered + extras
    else:
        # Fallback: infer from first clip's keys
        first_clip = next(iter(merged.values()))
        merged_metric_keys = [k for k in first_clip if isinstance(first_clip.get(k), dict)
                              and "mean" in first_clip.get(k, {})]

    print(f"  Merged: {len(merged)} clips, {len(merged_metric_keys)} common metrics")
    print(f"  Sources: {', '.join(source_jsons)}")
    return merged, merged_metric_keys


def main():
    parser = argparse.ArgumentParser(
        description="No-reference video quality metrics for analog video captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("src_dir", nargs="?", default=None,
                        help="Directory containing video files (not needed with --from-json)")
    parser.add_argument("--from-json", nargs="+", dest="from_json", metavar="JSON",
                        help="Generate report from pre-computed JSON file(s) instead of analyzing clips")
    parser.add_argument("--output-dir", help="Output directory for reports (default: src_dir or current dir)")
    parser.add_argument("--name", default="quality_report", help="Report filename prefix (default: quality_report)")
    parser.add_argument("--pattern", default="*.mov", help="File glob pattern (default: *.mov)")
    parser.add_argument("--text", action="store_true", help="Also generate a plain-text report")
    parser.add_argument("--screenshots", type=int, default=0,
                        help="Number of sample screenshots per clip to embed in HTML (default: 0)")
    parser.add_argument("--skip", action="append", default=[],
                        help="Skip N initial frames for a clip: PATTERN:N (e.g. --skip jvctbc1:1). "
                             "PATTERN is matched as substring of clip filename. Can be repeated.")
    parser.add_argument("--extra-detail-metrics",
                        default="detail_perceptual,detail_blur_inv,detail_sml",
                        help=("Comma-separated extra detail metrics to compute "
                              f"({', '.join(EXTRA_DETAIL_KEYS)}). "
                              "Default: %(default)s"))
    parser.add_argument("--report-metrics",
                        default="detail_perceptual,detail,detail_blur_inv,detail_sml,"
                                "sharpness,edge_strength,blocking,texture_quality,"
                                "ringing,temporal_stability,colorfulness,naturalness,"
                                "crushed_blacks,blown_whites",
                        help="Comma-separated metric keys to include in HTML/TXT report (default: %(default)s).")
    args = parser.parse_args()

    if not args.from_json and not args.src_dir:
        parser.error("src_dir is required unless --from-json is used")

    # --from-json mode: load pre-computed data and generate report
    if args.from_json:
        print(f"Loading {len(args.from_json)} JSON file(s)...")
        all_results, available_metric_keys = load_and_merge_jsons(args.from_json)

        # Determine report metrics
        if args.report_metrics:
            report_metric_keys = parse_metric_csv(args.report_metrics)
            missing = [k for k in report_metric_keys if k not in available_metric_keys]
            if missing:
                print(f"WARNING: Requested report metrics not in JSON data, dropping: {', '.join(missing)}")
                report_metric_keys = [k for k in report_metric_keys if k in available_metric_keys]
        else:
            report_metric_keys = available_metric_keys

        if not report_metric_keys:
            print("ERROR: No valid report metrics available")
            sys.exit(1)

        output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.from_json[0])) or "."
        os.makedirs(output_dir, exist_ok=True)

        # No comparison frames or screenshots in --from-json mode (no per-frame data or video access)
        comparisons = None
        samples = None
        print(f"\nNote: Comparison frames and screenshots not available in --from-json mode.")

        # Generate reports
        run_timestamp = datetime.now()
        ts_suffix = run_timestamp.strftime("%Y%m%d_%H%M%S")
        json_name = f"{args.name}_{ts_suffix}.json"

        # Write merged JSON
        json_output = {
            "_metadata": {
                "schema_version": 3,
                "metrics": list(available_metric_keys),
                "n_metrics": len(available_metric_keys),
                "report_metrics": list(report_metric_keys),
                "source_jsons": [os.path.basename(jp) for jp in args.from_json],
                "timestamp": run_timestamp.isoformat(),
                "json_filename": json_name,
            }
        }
        json_output.update(all_results)
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON:  {json_path}")

        display_name = args.name.replace("_", " ").title() if args.name != "quality_report" else "Merged Report"
        title = f"Video Quality Report — {display_name}"
        html_path = os.path.join(output_dir, f"{args.name}.html")
        with open(html_path, "w") as f:
            f.write(generate_html(all_results, report_metric_keys, title, comparisons, samples,
                                  json_filename=json_name))
        print(f"HTML:  {html_path}")

        if args.text:
            txt_path = os.path.join(output_dir, f"{args.name}.txt")
            with open(txt_path, "w") as f:
                f.write(format_text_report(all_results, "merged", report_metric_keys))
            print(f"Text:  {txt_path}")
        return

    # Standard mode: analyze clips from src_dir
    # Parse --skip arguments into {pattern: n_frames} dict
    skip_patterns = parse_skip_args(args.skip)

    extra_detail_metrics = parse_metric_csv(args.extra_detail_metrics)
    unknown_extra = [k for k in extra_detail_metrics if k not in EXTRA_DETAIL_KEYS]
    if unknown_extra:
        print(f"ERROR: Unknown --extra-detail-metrics: {', '.join(unknown_extra)}")
        print(f"       Allowed: {', '.join(EXTRA_DETAIL_KEYS)}")
        sys.exit(1)

    requested_metric_keys = list(ALL_KEYS)
    for key in extra_detail_metrics:
        if key not in requested_metric_keys:
            requested_metric_keys.append(key)

    analysis_metric_keys = list(requested_metric_keys)
    if DETAIL_PERCEPTUAL_KEY in requested_metric_keys:
        # Ensure dependencies are computed even if only the derived key was requested.
        for dep in DETAIL_PERCEPTUAL_DEPS:
            if dep not in analysis_metric_keys:
                analysis_metric_keys.append(dep)

    if args.report_metrics:
        report_metric_keys = parse_metric_csv(args.report_metrics)
        if not report_metric_keys:
            print("ERROR: --report-metrics is empty after parsing")
            sys.exit(1)
        unknown_report = [k for k in report_metric_keys if k not in METRIC_INFO]
        if unknown_report:
            print(f"ERROR: Unknown --report-metrics keys: {', '.join(unknown_report)}")
            sys.exit(1)
        missing_report = [k for k in report_metric_keys if k not in analysis_metric_keys]
        if missing_report:
            print(f"ERROR: --report-metrics keys not computed in this run: {', '.join(missing_report)}")
            print("       Add them via --extra-detail-metrics (or remove from --report-metrics).")
            sys.exit(1)
    else:
        report_metric_keys = list(requested_metric_keys)

    src_dir = args.src_dir.rstrip("/")
    output_dir = args.output_dir or src_dir
    os.makedirs(output_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(src_dir, args.pattern)))
    if not clips:
        print(f"ERROR: No files matching '{args.pattern}' found in {src_dir}")
        sys.exit(1)

    # Probe all clips and validate consistent resolution
    resolutions = {}
    for clip_path in clips:
        resolutions[clip_path] = probe_video(clip_path)
    res_set = set(resolutions.values())
    if len(res_set) > 1:
        print("ERROR: Not all clips have the same resolution:")
        for path, (w, h) in resolutions.items():
            print(f"  {w}x{h}: {os.path.basename(path)}")
        sys.exit(1)
    vid_w, vid_h = res_set.pop()

    print(f"Analyzing {len(clips)} clips in {src_dir} ({vid_w}x{vid_h})...")
    print(f"Computed metrics ({len(analysis_metric_keys)}): {', '.join(analysis_metric_keys)}")
    if report_metric_keys != analysis_metric_keys:
        print(f"Report metrics ({len(report_metric_keys)}): {', '.join(report_metric_keys)}")
    print("")

    all_results = {}
    all_perframe = {}
    clip_paths = {}  # name -> filepath mapping
    skip_offsets = {}  # name -> number of frames skipped
    for i, clip_path in enumerate(clips, 1):
        name = short_name(clip_path)
        clip_paths[name] = clip_path
        # Determine skip for this clip based on --skip patterns
        skip = 0
        basename = os.path.basename(clip_path)
        for pat, n in skip_patterns.items():
            if pat in basename or pat in name:
                skip = n
                break
        skip_offsets[name] = skip

        # Detect name collisions and append suffix if needed
        if name in all_results:
            orig_name = name
            suffix = 2
            while f"{name}_{suffix}" in all_results:
                suffix += 1
            name = f"{name}_{suffix}"
            print(f"  WARNING: Duplicate short name '{orig_name}' from {clip_path}, using '{name}'")
            skip_offsets[name] = skip_offsets.pop(orig_name, skip)

        skip_msg = f", skip {skip}" if skip > 0 else ""
        print(f"[{i:>2}/{len(clips)}] {name}{skip_msg}...", end=" ", flush=True)
        metrics, perframe = analyze_clip(clip_path, vid_w, vid_h, analysis_metric_keys, skip_frames=skip)

        if metrics is None:
            print(f"skipped (no frames decoded)")
            continue

        metrics["_source_file"] = os.path.basename(clip_path)
        all_results[name] = metrics
        all_perframe[name] = perframe
        print(f"done ({metrics['n_frames']} frames)")

    if DETAIL_PERCEPTUAL_KEY in analysis_metric_keys:
        print("\nComputing derived metric: detail_perceptual")
        if not add_detail_perceptual_metric(all_results, all_perframe):
            print("ERROR: failed to compute derived detail_perceptual metric")
            sys.exit(1)

    # JSON output (always) — timestamped filename for provenance
    run_timestamp = datetime.now()
    ts_suffix = run_timestamp.strftime("%Y%m%d_%H%M%S")
    json_output = {
        "_metadata": {
            "schema_version": 3,
            "metrics": list(analysis_metric_keys),
            "n_metrics": len(analysis_metric_keys),
            "report_metrics": list(report_metric_keys),
            "source_dir": os.path.abspath(src_dir),
            "pattern": args.pattern,
            "skip_frames": dict(skip_patterns),
            "timestamp": run_timestamp.isoformat(),
            "json_filename": f"{args.name}_{ts_suffix}.json",
        }
    }
    json_output.update(all_results)
    json_name = f"{args.name}_{ts_suffix}.json"
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON:  {json_path}")

    # Extract comparison frames and sample screenshots for HTML
    clip_names = sorted(all_results.keys())
    samples = None

    print("\nExtracting metric comparison frames...")
    comparisons = find_comparison_frames(clip_paths, clip_names, all_perframe,
                                         all_results, skip_offsets, report_metric_keys)

    if args.screenshots > 0:
        print(f"\nExtracting {args.screenshots} sample screenshots per clip...")
        samples = extract_sample_screenshots(clip_paths, clip_names, args.screenshots,
                                             skip_offsets)

    # HTML output (always)
    display_name = args.name.replace("_", " ").title() if args.name != "quality_report" else os.path.basename(src_dir)
    title = f"Video Quality Report — {display_name}"
    html_path = os.path.join(output_dir, f"{args.name}.html")
    with open(html_path, "w") as f:
        f.write(generate_html(all_results, report_metric_keys, title, comparisons, samples,
                              json_filename=json_name))
    print(f"HTML:  {html_path}")

    # Text output (optional)
    if args.text:
        txt_path = os.path.join(output_dir, f"{args.name}.txt")
        with open(txt_path, "w") as f:
            f.write(format_text_report(all_results, src_dir, report_metric_keys))
        print(f"Text:  {txt_path}")


if __name__ == "__main__":
    main()
