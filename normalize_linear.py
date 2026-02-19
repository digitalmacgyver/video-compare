#!/usr/bin/env python3
"""Linear brightness normalization for analog video captures.

Applies a simple linear Y scale (multiply by constant) so all clips match the
reference clip's mean luma. Chroma channels are passed through unchanged.

This is a minimal brightness correction that preserves the shape of each clip's
luminance distribution — unlike histogram matching, it does not redistribute
values nonlinearly. Useful for isolating brightness differences when evaluating
whether quality metrics are truly brightness-agnostic.

Usage:
    python normalize_linear.py <src_dir> [options]

    src_dir             Directory containing .mov files to normalize

Options:
    --reference CLIP    Reference clip basename (default: first alphabetically)
    --output-dir DIR    Output directory (default: <src_dir>/linear_normalized/)
    --pattern GLOB      File glob pattern (default: "*.mov")

Examples:
    python normalize_linear.py /path/to/clips/
    python normalize_linear.py /path/to/clips/ --reference best_clip.mov
    python normalize_linear.py /path/to/clips/ --output-dir /path/to/output/
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


def compute_mean_y(filepath, width, height):
    """Compute mean Y value across all frames of a clip."""
    proc = subprocess.Popen(decode_command(filepath), stdout=subprocess.PIPE)
    try:
        y_sum = 0.0
        y_count = 0
        n_frames = 0

        while True:
            frame = read_frame(proc.stdout, width, height)
            if frame is None:
                break
            y, _, _ = frame
            y_sum += y.astype(np.float64).sum()
            y_count += y.size
            n_frames += 1

        proc.wait()
    finally:
        proc.stdout.close()
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()
    mean_y = y_sum / y_count if y_count > 0 else 0.0
    return mean_y, n_frames


def normalize_clip_linear(src_path, out_path, ref_mean_y, width, height, frame_rate):
    """Normalize one clip by linear Y scaling to match reference mean brightness.

    Computes scale = ref_mean_y / src_mean_y, then multiplies every Y sample
    by that constant. Chroma channels pass through unchanged.
    """
    clip_name = os.path.basename(src_path)
    print(f"  Processing: {clip_name}")

    # First pass: compute source mean Y
    src_mean_y, n_frames = compute_mean_y(src_path, width, height)
    if src_mean_y < 0.01:
        print(f"    WARNING: Very dark clip (mean_Y={src_mean_y:.2f}), skipping")
        return False

    scale = ref_mean_y / src_mean_y
    print(f"    Frames: {n_frames}, mean_Y: {src_mean_y:.2f}, "
          f"ref_mean_Y: {ref_mean_y:.2f}, scale: {scale:.4f}")

    # Second pass: apply linear scale to Y, pass through U/V
    proc_dec = subprocess.Popen(decode_command(src_path), stdout=subprocess.PIPE)
    proc_enc = subprocess.Popen(encode_command(out_path, width, height, frame_rate),
                                stdin=subprocess.PIPE)
    try:
        frame_num = 0
        while True:
            frame = read_frame(proc_dec.stdout, width, height)
            if frame is None:
                break
            y, u, v = frame
            y_scaled = np.clip(y.astype(np.float64) * scale, 0, MAX_VAL).astype(np.uint16)
            write_frame(proc_enc.stdin, y_scaled, u, v)
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
        description="Linear brightness normalization for analog video captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("src_dir", help="Directory containing video files to normalize")
    parser.add_argument("--reference",
                        help="Reference clip basename (default: first alphabetically)")
    parser.add_argument("--output-dir",
                        help="Output directory (default: <src_dir>/linear_normalized/)")
    parser.add_argument("--pattern", default="*.mov",
                        help="File glob pattern (default: %(default)s)")
    args = parser.parse_args()

    src_dir = args.src_dir.rstrip("/")
    out_dir = args.output_dir or os.path.join(src_dir, "linear_normalized")
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

    # Auto-detect frame rate
    frame_rate = detect_frame_rate(clips[0])
    print(f"Detected frame rate: {frame_rate}")

    # Compute reference mean Y
    print(f"\n=== Computing Reference Mean Y ===")
    print(f"  Reference: {os.path.basename(ref_path)}")
    ref_mean_y, ref_frames = compute_mean_y(ref_path, vid_w, vid_h)
    print(f"  Reference mean Y: {ref_mean_y:.2f} ({ref_frames} frames)")

    # Normalize all clips
    print(f"\n=== Linear-normalizing {len(clips)} clips ===")
    for i, clip_path in enumerate(clips, 1):
        clip_name = os.path.basename(clip_path)
        base, ext = os.path.splitext(clip_name)
        out_name = f"{base}_linnorm{ext}"
        out_path = os.path.join(out_dir, out_name)

        print(f"\n[{i}/{len(clips)}]")
        if not normalize_clip_linear(clip_path, out_path, ref_mean_y,
                                     vid_w, vid_h, frame_rate):
            print(f"  FAILED: {clip_name}")
            continue

    print(f"\n=== All clips linear-normalized. Outputs in {out_dir} ===")


if __name__ == "__main__":
    main()
