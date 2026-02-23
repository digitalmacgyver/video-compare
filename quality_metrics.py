#!/usr/bin/env python3
"""Compute no-reference quality metrics for analog video clips.

Analyzes clips and outputs a timestamped JSON file with per-clip metric data.
This is the computation-only counterpart to quality_report.py — it produces
JSON that can be fed into quality_report.py --from-json for report generation.

Usage:
    python quality_metrics.py <src_dir> [options]

    src_dir             Directory containing video files to analyze

Options:
    --output-dir DIR    Output directory (default: same as src_dir)
    --name PREFIX       JSON filename prefix (default: "quality_metrics")
    --pattern GLOB      File glob pattern (default: "*.mov")
    --skip PATTERN:N    Skip N initial frames for clips matching PATTERN (repeatable)
    --extra-detail-metrics CSV  Extra detail metrics to compute (default: detail_perceptual,detail_blur_inv,detail_sml)

Examples:
    # Compute metrics for all clips in a directory
    python quality_metrics.py /path/to/clips/

    # Compute metrics for a single clip using --pattern
    python quality_metrics.py /path/to/clips/ --pattern "specific_clip*.mov"

    # Then generate a report from one or more JSON files
    python quality_report.py --from-json metrics1.json metrics2.json --name combined
"""

import argparse
import os
import sys
import glob
import json
from datetime import datetime

from common import ALL_KEYS, METRIC_INFO, probe_video
from quality_report import (
    analyze_clip, add_detail_perceptual_metric, short_name,
    parse_metric_csv, EXTRA_DETAIL_KEYS, DETAIL_PERCEPTUAL_KEY,
    DETAIL_PERCEPTUAL_DEPS, DERIVED_KEYS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute no-reference quality metrics for analog video clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("src_dir", help="Directory containing video files to analyze")
    parser.add_argument("--output-dir", help="Output directory for JSON (default: src_dir)")
    parser.add_argument("--name", default="quality_metrics",
                        help="JSON filename prefix (default: %(default)s)")
    parser.add_argument("--pattern", default="*.mov",
                        help="File glob pattern (default: %(default)s)")
    parser.add_argument("--skip", action="append", default=[],
                        help="Skip N initial frames for a clip: PATTERN:N (repeatable)")
    parser.add_argument("--extra-detail-metrics",
                        default="detail_perceptual,detail_blur_inv,detail_sml",
                        help=("Comma-separated extra detail metrics to compute "
                              f"({', '.join(EXTRA_DETAIL_KEYS)}). "
                              "Default: %(default)s"))
    args = parser.parse_args()

    # Parse --skip arguments
    skip_patterns = {}
    for s in args.skip:
        if ":" not in s:
            print(f"ERROR: --skip format must be PATTERN:N, got '{s}'")
            sys.exit(1)
        pat, n = s.rsplit(":", 1)
        try:
            skip_patterns[pat] = int(n)
        except ValueError:
            print(f"ERROR: --skip N must be integer, got '{n}'")
            sys.exit(1)

    extra_detail_metrics = parse_metric_csv(args.extra_detail_metrics)
    unknown_extra = [k for k in extra_detail_metrics if k not in EXTRA_DETAIL_KEYS]
    if unknown_extra:
        print(f"ERROR: Unknown --extra-detail-metrics: {', '.join(unknown_extra)}")
        print(f"       Allowed: {', '.join(EXTRA_DETAIL_KEYS)}")
        sys.exit(1)

    analysis_metric_keys = list(ALL_KEYS)
    for key in extra_detail_metrics:
        if key not in analysis_metric_keys:
            analysis_metric_keys.append(key)

    if DETAIL_PERCEPTUAL_KEY in analysis_metric_keys:
        for dep in DETAIL_PERCEPTUAL_DEPS:
            if dep not in analysis_metric_keys:
                analysis_metric_keys.append(dep)

    src_dir = args.src_dir.rstrip("/")
    output_dir = args.output_dir or src_dir
    os.makedirs(output_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(src_dir, args.pattern)))
    if not clips:
        print(f"ERROR: No files matching '{args.pattern}' found in {src_dir}")
        sys.exit(1)

    # Probe and validate resolution
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

    print(f"Computing metrics for {len(clips)} clip(s) in {src_dir} ({vid_w}x{vid_h})...")
    print(f"Metrics ({len(analysis_metric_keys)}): {', '.join(analysis_metric_keys)}")
    print("")

    all_results = {}
    all_perframe = {}
    for i, clip_path in enumerate(clips, 1):
        name = short_name(clip_path)
        basename = os.path.basename(clip_path)

        skip = 0
        for pat, n in skip_patterns.items():
            if pat in basename or pat in name:
                skip = n
                break

        # Handle name collisions
        if name in all_results:
            orig_name = name
            suffix = 2
            while f"{name}_{suffix}" in all_results:
                suffix += 1
            name = f"{name}_{suffix}"
            print(f"  WARNING: Duplicate short name '{orig_name}', using '{name}'")

        skip_msg = f", skip {skip}" if skip > 0 else ""
        print(f"[{i:>2}/{len(clips)}] {name}{skip_msg}...", end=" ", flush=True)
        metrics, perframe = analyze_clip(clip_path, vid_w, vid_h,
                                         analysis_metric_keys, skip_frames=skip)
        if metrics is None:
            print("skipped (no frames decoded)")
            continue

        metrics["_source_file"] = os.path.basename(clip_path)
        all_results[name] = metrics
        all_perframe[name] = perframe
        print(f"done ({metrics['n_frames']} frames)")

    if not all_results:
        print("ERROR: No clips were successfully analyzed")
        sys.exit(1)

    if DETAIL_PERCEPTUAL_KEY in analysis_metric_keys:
        print("\nComputing derived metric: detail_perceptual")
        if not add_detail_perceptual_metric(all_results, all_perframe):
            print("ERROR: failed to compute derived detail_perceptual metric")
            sys.exit(1)

    # JSON output with timestamped filename
    run_timestamp = datetime.now()
    ts_suffix = run_timestamp.strftime("%Y%m%d_%H%M%S")
    json_name = f"{args.name}_{ts_suffix}.json"
    json_output = {
        "_metadata": {
            "schema_version": 3,
            "metrics": list(analysis_metric_keys),
            "n_metrics": len(analysis_metric_keys),
            "source_dir": os.path.abspath(src_dir),
            "pattern": args.pattern,
            "skip_frames": dict(skip_patterns),
            "timestamp": run_timestamp.isoformat(),
            "json_filename": json_name,
        }
    }
    json_output.update(all_results)

    json_path = os.path.join(output_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON:  {json_path}")


if __name__ == "__main__":
    main()
