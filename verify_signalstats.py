#!/usr/bin/env python3
"""Run signalstats on normalized clips and print a comparison table."""

import subprocess
import os
import glob
import json
import re

NORM_DIR = "/wintmp/analog_video/noref_compare/normalized"


def get_signalstats(filepath):
    """Run ffprobe signalstats and return averaged values."""
    cmd = [
        "ffprobe", "-v", "error",
        "-f", "lavfi", "-i",
        f"movie='{filepath}',signalstats",
        "-show_entries", "frame_tags=lavfi.signalstats.YMIN,lavfi.signalstats.YLOW,lavfi.signalstats.YAVG,lavfi.signalstats.YHIGH,lavfi.signalstats.YMAX,lavfi.signalstats.UAVG,lavfi.signalstats.VAVG,lavfi.signalstats.SATAVG,lavfi.signalstats.SATMAX",
        "-of", "json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    data = json.loads(result.stdout)

    keys = ["YMIN", "YLOW", "YAVG", "YHIGH", "YMAX", "UAVG", "VAVG", "SATAVG", "SATMAX"]
    accum = {k: 0.0 for k in keys}
    n = 0
    for frame in data.get("frames", []):
        tags = frame.get("tags", {})
        for k in keys:
            tag_key = f"lavfi.signalstats.{k}"
            if tag_key in tags:
                accum[k] += float(tags[tag_key])
        n += 1

    if n > 0:
        for k in keys:
            accum[k] /= n
    return accum, n


def short_name(filepath):
    """Extract short clip name."""
    name = os.path.basename(filepath)
    # Remove common prefix/suffix
    name = name.replace("twister_", "").replace("_trimmed_VIEW_ACTION_SAFE_IVTC_normalized.mov", "")
    return name


def main():
    clips = sorted(glob.glob(os.path.join(NORM_DIR, "*.mov")))
    print(f"Verifying {len(clips)} normalized clips\n")

    # Header
    header = f"{'Clip':<35} {'YMIN':>6} {'YLOW':>6} {'YAVG':>6} {'YHIGH':>7} {'YMAX':>6} {'UAVG':>6} {'VAVG':>6} {'SATAVG':>7} {'SATMAX':>7}"
    print(header)
    print("-" * len(header))

    results = []
    for clip in clips:
        name = short_name(clip)
        stats, n_frames = get_signalstats(clip)
        row = f"{name:<35} {stats['YMIN']:>6.1f} {stats['YLOW']:>6.1f} {stats['YAVG']:>6.1f} {stats['YHIGH']:>7.1f} {stats['YMAX']:>6.1f} {stats['UAVG']:>6.1f} {stats['VAVG']:>6.1f} {stats['SATAVG']:>7.1f} {stats['SATMAX']:>7.1f}"
        print(row)
        results.append((name, stats))

    # Print convergence summary
    print("\n=== Convergence Summary ===")
    for key in ["YAVG", "SATAVG", "UAVG", "VAVG"]:
        vals = [r[1][key] for r in results]
        print(f"  {key}: min={min(vals):.1f}, max={max(vals):.1f}, "
              f"spread={max(vals)-min(vals):.1f}, mean={sum(vals)/len(vals):.1f}")


if __name__ == "__main__":
    main()
