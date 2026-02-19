#!/usr/bin/env python3
"""Normalize brightness and saturation of analog video captures.

Uses histogram matching on Y (luma) channel and linear chroma saturation
scaling to bring all clips to a common reference level. This removes
brightness/color differences so no-reference quality metrics focus on
artifacts and clarity rather than exposure/color grading differences.

Usage:
    python normalize.py <src_dir> [options]

    src_dir             Directory containing .mov files to normalize

Options:
    --reference CLIP    Reference clip basename (default: first alphabetically)
    --output-dir DIR    Output directory (default: <src_dir>/normalized/)

Examples:
    python normalize.py /path/to/clips/
    python normalize.py /path/to/clips/ --reference best_clip.mov
    python normalize.py /path/to/clips/ --output-dir /path/to/output/
"""

import argparse
import subprocess
import sys
import os
import glob
import numpy as np
from common import (probe_video, detect_frame_rate, decode_command,
                    encode_command, read_frame, write_frame)

# === Pixel format constants ===
BIT_DEPTH = 10
MAX_VAL = (1 << BIT_DEPTH) - 1   # 1023
NEUTRAL = 1 << (BIT_DEPTH - 1)    # 512


def compute_reference_stats(ref_path, width, height):
    """Compute reference clip's average Y histogram CDF and mean chroma saturation."""
    print(f"  Computing reference stats from: {os.path.basename(ref_path)}")
    proc = subprocess.Popen(decode_command(ref_path), stdout=subprocess.PIPE)
    try:
        hist_accum = np.zeros(MAX_VAL + 1, dtype=np.float64)
        sat_sum = 0.0
        sat_count = 0
        n_frames = 0

        while True:
            frame = read_frame(proc.stdout, width, height)
            if frame is None:
                break
            y, u, v = frame
            hist, _ = np.histogram(y, bins=MAX_VAL + 1, range=(0, MAX_VAL + 1))
            hist_accum += hist.astype(np.float64)
            u_f = u.astype(np.float64) - NEUTRAL
            v_f = v.astype(np.float64) - NEUTRAL
            sat = np.sqrt(u_f * u_f + v_f * v_f)
            sat_sum += sat.sum()
            sat_count += sat.size
            n_frames += 1

        proc.wait()
    finally:
        proc.stdout.close()
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()
    print(f"  Reference: {n_frames} frames processed")

    if n_frames == 0:
        print("  ERROR: No frames decoded for reference clip")
        sys.exit(1)

    avg_hist = hist_accum / n_frames
    cdf = np.cumsum(avg_hist)
    cdf = cdf / cdf[-1]

    mean_sat = sat_sum / sat_count
    print(f"  Reference mean saturation: {mean_sat:.2f}")

    return cdf, mean_sat, n_frames


def build_lut(src_cdf, ref_cdf):
    """Build a lookup table mapping source Y values to reference-matched Y values."""
    lut = np.zeros(MAX_VAL + 1, dtype=np.uint16)
    ref_idx = 0
    for src_val in range(MAX_VAL + 1):
        while ref_idx < MAX_VAL and ref_cdf[ref_idx] < src_cdf[src_val]:
            ref_idx += 1
        lut[src_val] = ref_idx
    return lut


def normalize_clip(src_path, out_path, ref_cdf, ref_mean_sat, width, height, frame_rate):
    """Normalize one clip: histogram-match Y, scale chroma saturation."""
    clip_name = os.path.basename(src_path)
    print(f"  Processing: {clip_name}")

    # First pass: compute source clip's average Y CDF and mean saturation
    proc_r = subprocess.Popen(decode_command(src_path), stdout=subprocess.PIPE)
    try:
        src_hist_accum = np.zeros(MAX_VAL + 1, dtype=np.float64)
        src_sat_sum = 0.0
        src_sat_count = 0
        n_frames = 0

        while True:
            frame = read_frame(proc_r.stdout, width, height)
            if frame is None:
                break
            y, u, v = frame
            hist, _ = np.histogram(y, bins=MAX_VAL + 1, range=(0, MAX_VAL + 1))
            src_hist_accum += hist.astype(np.float64)
            u_f = u.astype(np.float64) - NEUTRAL
            v_f = v.astype(np.float64) - NEUTRAL
            sat = np.sqrt(u_f * u_f + v_f * v_f)
            src_sat_sum += sat.sum()
            src_sat_count += sat.size
            n_frames += 1

        proc_r.wait()
    finally:
        proc_r.stdout.close()
        try:
            proc_r.kill()
        except OSError:
            pass
        proc_r.wait()

    if n_frames == 0:
        print(f"    ERROR: No frames decoded for {src_path}")
        return False

    src_avg_hist = src_hist_accum / n_frames
    src_cdf = np.cumsum(src_avg_hist)
    src_cdf = src_cdf / src_cdf[-1]

    lut = build_lut(src_cdf, ref_cdf)

    src_mean_sat = src_sat_sum / src_sat_count
    if src_mean_sat > 0.01:
        chroma_scale = ref_mean_sat / src_mean_sat
    else:
        chroma_scale = 1.0

    print(f"    Frames: {n_frames}, src_mean_sat: {src_mean_sat:.2f}, "
          f"chroma_scale: {chroma_scale:.3f}")

    # Second pass: apply LUT and chroma scaling, encode output
    proc_dec = subprocess.Popen(decode_command(src_path), stdout=subprocess.PIPE)
    proc_enc = subprocess.Popen(encode_command(out_path, width, height, frame_rate), stdin=subprocess.PIPE)
    try:
        frame_num = 0
        while True:
            frame = read_frame(proc_dec.stdout, width, height)
            if frame is None:
                break
            y, u, v = frame
            y_out = lut[y]
            u_f = u.astype(np.float64) - NEUTRAL
            v_f = v.astype(np.float64) - NEUTRAL
            u_out = np.clip(NEUTRAL + u_f * chroma_scale, 0, MAX_VAL).astype(np.uint16)
            v_out = np.clip(NEUTRAL + v_f * chroma_scale, 0, MAX_VAL).astype(np.uint16)
            write_frame(proc_enc.stdin, y_out, u_out, v_out)
            frame_num += 1

        proc_enc.stdin.close()
        proc_dec.wait()
        proc_enc.wait()
    finally:
        proc_dec.stdout.close()
        try:
            proc_dec.kill()
        except OSError:
            pass
        try:
            proc_enc.kill()
        except OSError:
            pass
        proc_dec.wait()
        proc_enc.wait()

    if proc_enc.returncode != 0:
        print(f"    ERROR: encode failed for {clip_name}")
        return False

    print(f"    Done: {frame_num} frames written")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Normalize brightness and saturation of analog video captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("src_dir", help="Directory containing video files to normalize")
    parser.add_argument("--reference", help="Reference clip basename (default: first alphabetically)")
    parser.add_argument("--output-dir", help="Output directory (default: <src_dir>/normalized/)")
    parser.add_argument("--pattern", default="*.mov",
                        help="File glob pattern (default: %(default)s)")
    args = parser.parse_args()

    src_dir = args.src_dir.rstrip("/")
    out_dir = args.output_dir or os.path.join(src_dir, "normalized")
    os.makedirs(out_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(src_dir, args.pattern)))
    if not clips:
        print(f"ERROR: No files matching '{args.pattern}' found in {src_dir}")
        sys.exit(1)

    print(f"Found {len(clips)} clips in {src_dir}")

    # Auto-detect resolution and validate all clips match
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
    print(f"Resolution: {vid_w}x{vid_h}")

    # Pick reference clip
    if args.reference:
        ref_path = os.path.join(src_dir, args.reference)
    else:
        ref_path = clips[0]

    if not os.path.exists(ref_path):
        print(f"ERROR: Reference clip not found: {ref_path}")
        sys.exit(1)

    # Auto-detect frame rate from first source clip
    frame_rate = detect_frame_rate(clips[0])
    print(f"Detected frame rate: {frame_rate}")

    # Compute reference statistics
    print("\n=== Computing Reference Statistics ===")
    ref_cdf, ref_mean_sat, ref_frames = compute_reference_stats(ref_path, vid_w, vid_h)

    # Normalize all clips
    print(f"\n=== Normalizing {len(clips)} Clips ===")
    for i, clip_path in enumerate(clips, 1):
        clip_name = os.path.basename(clip_path)
        base, ext = os.path.splitext(clip_name)
        out_name = f"{base}_normalized{ext}"
        out_path = os.path.join(out_dir, out_name)

        print(f"\n[{i}/{len(clips)}]")
        if not normalize_clip(clip_path, out_path, ref_cdf, ref_mean_sat, vid_w, vid_h, frame_rate):
            print(f"  FAILED: {clip_name}")
            continue

    print(f"\n=== All clips normalized. Outputs in {out_dir} ===")


if __name__ == "__main__":
    main()
