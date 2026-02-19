#!/usr/bin/env python3
"""Compare two quality report JSONs — verify per-metric values match and document ranking changes."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quality_report import compute_composites, ALL_KEYS


def compare(baseline_path, postfix_path):
    with open(baseline_path) as f:
        base = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    with open(postfix_path) as f:
        post = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    clips = sorted(base.keys())
    post_clips = sorted(post.keys())
    if clips != post_clips:
        print(f"MISMATCH: different clip sets")
        print(f"  Baseline: {clips}")
        print(f"  Post-fix: {post_clips}")
        return False

    # Compare per-metric mean/std values (should be IDENTICAL)
    max_diff = 0.0
    diffs = []
    for c in clips:
        for k in ALL_KEYS:
            if k not in base[c] or k not in post[c]:
                continue
            bm = base[c][k]["mean"]
            pm = post[c][k]["mean"]
            diff = abs(bm - pm)
            if diff > 1e-10:
                diffs.append((c, k, bm, pm, diff))
            max_diff = max(max_diff, diff)

    print(f"=== Per-Metric Value Comparison ===")
    print(f"Clips: {len(clips)}, Metrics: {len(ALL_KEYS)}")
    print(f"Max absolute difference: {max_diff:.2e}")
    if diffs:
        print(f"\nDifferences above 1e-10:")
        for c, k, bm, pm, d in diffs:
            print(f"  {c}/{k}: {bm} -> {pm} (diff={d:.2e})")
    else:
        print("All per-metric values IDENTICAL.")

    # Compare composite rankings
    base_comp = compute_composites(base)
    post_comp = compute_composites(post)

    print(f"\n=== Composite Ranking Comparison ===")
    print(f"{'Clip':<45} {'Base':>8} {'Post':>8} {'Diff':>8}")
    print("-" * 72)
    any_change = False
    for c in sorted(clips, key=lambda x: post_comp[x]["overall"]):
        bo = base_comp[c]["overall"]
        po = post_comp[c]["overall"]
        diff = po - bo
        marker = " *" if abs(diff) > 0.01 else ""
        if abs(diff) > 0.01:
            any_change = True
        print(f"{c:<45} {bo:>8.2f} {po:>8.2f} {diff:>+8.2f}{marker}")

    if any_change:
        print("\n* = ranking changed (expected due to tied-rank fix)")
    else:
        print("\nNo ranking changes.")

    return len(diffs) == 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline.json> <postfix.json>")
        sys.exit(1)
    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
