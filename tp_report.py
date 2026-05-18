"""tp_report — directory-driven SW2 measurement + combined HTML report.

Walks a directory of captures, runs tp_measure on each (caching per-file
JSON sidecars), then renders a combined HTML report with tp_compare.

Usage:
    python tp_report.py /path/to/captures/ --output report.html

Per-capture JSON is written next to the source by default (or to
--outdir). The JSON name is `<capture_stem>_stage2.json`. Existing JSONs
are reused unless --force is set. Sidecar overlay/fiducial PNGs are
written alongside each JSON.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from typing import List


def _enumerate_captures(directory: str, patterns: List[str]) -> List[str]:
    found: List[str] = []
    for pat in patterns:
        found.extend(sorted(glob.glob(os.path.join(directory, pat))))
    seen = set()
    unique: List[str] = []
    for p in found:
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _write_sidecars(capture_path: str, json_path: str,
                    frame_index: int) -> None:
    """Generate the per-capture sidecar PNGs (overlay / fiducial crops /
    wedge crop). Each is best-effort: failures log but don't abort."""
    stem = os.path.splitext(json_path)[0]
    try:
        import tp_sample_overlay
        import cv2
        bgr = tp_sample_overlay.annotate(capture_path, json_path, frame_index)
        cv2.imwrite(stem + "_overlay.png", bgr)
    except Exception as e:
        print(f"  WARNING: overlay PNG failed: {e}", file=sys.stderr)
    try:
        import tp_fiducial_crops
        import cv2
        fids_bgr = tp_fiducial_crops.build(capture_path, json_path, frame_index)
        cv2.imwrite(stem + "_fiducials.png", fids_bgr)
    except Exception as e:
        print(f"  WARNING: fiducial-crops PNG failed: {e}", file=sys.stderr)
    try:
        import tp_wedge_crops
        import cv2
        wedge_bgr = tp_wedge_crops.build(capture_path, json_path, frame_index)
        cv2.imwrite(stem + "_wedge.png", wedge_bgr)
    except Exception as e:
        print(f"  WARNING: wedge-crops PNG failed: {e}", file=sys.stderr)


def _ensure_json(capture_path: str, json_path: str, frame_index: int,
                 force: bool, with_overlays: bool) -> bool:
    """Run tp_measure if needed and (re-)generate sidecar PNGs. Returns
    True if a usable JSON exists at the end. If `force` is False and
    the JSON already exists, the measurement is skipped but sidecars
    are regenerated unconditionally (cheap, and lets the report pick
    up newly-added sidecar types without forcing a full re-measure)."""
    stem = os.path.splitext(json_path)[0]
    if not os.path.exists(json_path) or force:
        import tp_measure
        try:
            data = tp_measure.measure(capture_path, frame_index)
        except Exception as e:
            print(f"WARNING: tp_measure failed on {capture_path}: {e}",
                  file=sys.stderr)
            return False
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=float)
        print(
            f"  wrote {json_path}: "
            f"registration={data['_meta']['registration']['quality_flag']}"
        )
    if with_overlays:
        # Only regenerate sidecars that are missing (or always if force).
        needs = (force
                 or not os.path.exists(stem + "_overlay.png")
                 or not os.path.exists(stem + "_fiducials.png")
                 or not os.path.exists(stem + "_wedge.png"))
        if needs:
            _write_sidecars(capture_path, json_path, frame_index)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("directory",
                   help="directory containing capture files")
    p.add_argument("--output", required=True,
                   help="combined HTML report path")
    p.add_argument("--pattern",
                   default="*.avi,*.mkv,*.mov,*.mp4",
                   help="comma-separated glob patterns (default: "
                        "*.avi,*.mkv,*.mov,*.mp4)")
    p.add_argument("--outdir", default=None,
                   help="where per-capture JSON sidecars are written "
                        "(default: same directory as the captures)")
    p.add_argument("--force", action="store_true",
                   help="re-run tp_measure even when a JSON already exists")
    p.add_argument("--frame-index", type=int, default=60,
                   help="frame index to sample (default 60)")
    p.add_argument("--no-overlays", action="store_true",
                   help="skip overlay/fiducial sidecar PNG generation")
    args = p.parse_args()

    patterns = [s.strip() for s in args.pattern.split(",") if s.strip()]
    captures = _enumerate_captures(args.directory, patterns)
    if not captures:
        print(f"no captures found in {args.directory} (patterns: {patterns})",
              file=sys.stderr)
        sys.exit(2)

    outdir = args.outdir or args.directory
    os.makedirs(outdir, exist_ok=True)

    print(f"found {len(captures)} captures in {args.directory}")
    json_paths: List[str] = []
    for cap in captures:
        stem = os.path.splitext(os.path.basename(cap))[0]
        json_path = os.path.join(outdir, f"{stem}_stage2.json")
        print(f"processing {os.path.basename(cap)}")
        ok = _ensure_json(cap, json_path, args.frame_index,
                          args.force, not args.no_overlays)
        if ok:
            json_paths.append(json_path)

    if not json_paths:
        print("no usable JSONs produced; nothing to render", file=sys.stderr)
        sys.exit(3)

    import tp_compare
    captures_data = []
    for path in json_paths:
        with open(path) as f:
            cap = json.load(f)
        cap["_source_json_path"] = path
        captures_data.append(cap)

    html = tp_compare.render_page(captures_data)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output} ({len(captures_data)} captures)")


if __name__ == "__main__":
    main()
