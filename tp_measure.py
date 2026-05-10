#!/usr/bin/env python3
"""tp_measure: per-capture SW2 measurement.

Pipeline: probe -> extract single frame (no deinterlace) -> pad to 720x486 ->
register against ideal -> sample tartan + gray patches -> write JSON.

CLI:
    python tp_measure.py <capture> --frame N --output cap.json
"""

from __future__ import annotations
import json
import subprocess
from typing import Any, Dict, Tuple

import numpy as np

import common  # existing module
import tp_chart
import tp_register


_TOOL_VERSION = "tp_measure 0.1"


def extract_frame(
    capture_path: str, frame_index: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Decode a single yuv422p10le frame from `capture_path` at `frame_index`.

    Uses ffmpeg with NO deinterlace filter: interlaced sources are weaved
    (lines as-stored).
    """
    width, height = common.probe_video(capture_path)

    # Probe field order to detect unexpected progressive sources.
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=field_order,codec_name",
        "-of", "default=nw=1", capture_path,
    ]
    probe_out = subprocess.check_output(probe_cmd, text=True)
    field_order = "unknown"
    codec_name = "unknown"
    for line in probe_out.strip().splitlines():
        k, _, v = line.partition("=")
        if k == "field_order":
            field_order = v.strip() or "unknown"
        elif k == "codec_name":
            codec_name = v.strip() or "unknown"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", capture_path,
        "-vf", f"select=eq(n\\,{int(frame_index)})",
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "yuv422p10le",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    y_size = width * height
    u_size = (width // 2) * height
    expected = (y_size + u_size + u_size) * 2
    if len(raw) != expected:
        raise RuntimeError(
            f"ffmpeg returned {len(raw)} bytes; expected {expected} "
            f"for {width}x{height} yuv422p10le"
        )
    arr = np.frombuffer(raw, dtype=np.uint16)
    Y = arr[:y_size].reshape(height, width).copy()
    U = arr[y_size:y_size + u_size].reshape(height, width // 2).copy()
    V = arr[y_size + u_size:].reshape(height, width // 2).copy()

    meta: Dict[str, Any] = {
        "capture": capture_path,
        "frame_index": int(frame_index),
        "raster_in": [width, height],
        "field_order": field_order,
        "codec_name": codec_name,
        "progressive_warning": (field_order == "progressive"),
        "decode": {
            "deinterlace": "none-weave-only",
            "pix_fmt": "yuv422p10le",
        },
    }
    return Y, U, V, meta


def pad_to_486(
    Y: np.ndarray, U: np.ndarray, V: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Pad Y/U/V planes up to height=486 with grey-background. No horizontal
    padding (NTSC active is always 720 wide; mismatches raise).
    """
    height = Y.shape[0]
    width = Y.shape[1]
    if width != 720:
        raise ValueError(f"unexpected width {width}; expected 720")
    if height > 486:
        raise ValueError(f"height {height} exceeds NTSC 486; cannot pad down")
    if height == 486:
        return Y, U, V, {"top": 0, "bottom": 0, "left": 0, "right": 0}
    delta = 486 - height
    top = delta // 2
    bottom = delta - top
    grey_y = tp_chart.GREY_BACKGROUND_Y10
    grey_c = tp_chart.CHROMA_CENTER

    def _pad(plane, c_w, fill):
        new = np.full((486, c_w), fill, dtype=plane.dtype)
        new[top:top + height, :] = plane
        return new

    Y_padded = _pad(Y, width, grey_y)
    U_padded = _pad(U, width // 2, grey_c)
    V_padded = _pad(V, width // 2, grey_c)
    return Y_padded, U_padded, V_padded, {
        "top": top, "bottom": bottom, "left": 0, "right": 0
    }


def _apply_affine(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    a, b, tx = M[0]
    c, d, ty = M[1]
    return float(a * x + b * y + tx), float(c * x + d * y + ty)


def sample_region(
    Y: np.ndarray, U: np.ndarray, V: np.ndarray,
    region: Dict[str, Any], affine: np.ndarray,
) -> Dict[str, Any]:
    """Sample the centre window of `region` from a registered capture.

    `affine` maps ideal coords -> capture coords (2x3, applied as
    [a, b, tx; c, d, ty]).
    """
    x, y, w, h = region["ideal_box"]
    cx_ideal = x + w / 2.0
    cy_ideal = y + h / 2.0
    cx_cap, cy_cap = _apply_affine(affine, cx_ideal, cy_ideal)

    size_frac = float(region["sample"]["size_frac"])
    half_w = max(1, int(round(w * size_frac / 2.0)))
    half_h = max(1, int(round(h * size_frac / 2.0)))

    x0 = max(0, int(round(cx_cap)) - half_w)
    x1 = min(Y.shape[1], int(round(cx_cap)) + half_w)
    y0 = max(0, int(round(cy_cap)) - half_h)
    y1 = min(Y.shape[0], int(round(cy_cap)) + half_h)

    y_patch = Y[y0:y1, x0:x1].astype(np.float64)
    # Chroma subsampled: x is at 2x luma resolution.
    cx0, cx1 = x0 // 2, max(x0 // 2 + 1, x1 // 2)
    u_patch = U[y0:y1, cx0:cx1].astype(np.float64)
    v_patch = V[y0:y1, cx0:cx1].astype(np.float64)

    measured = (
        float(y_patch.mean()) if y_patch.size else float("nan"),
        float(u_patch.mean()) if u_patch.size else float("nan"),
        float(v_patch.mean()) if v_patch.size else float("nan"),
    )
    e = region["expected"]
    delta = (
        measured[0] - e["y10"],
        measured[1] - e["u10"],
        measured[2] - e["v10"],
    )

    # Saturation as ratio of chroma-magnitude vs ideal.
    measured_chroma = ((measured[1] - tp_chart.CHROMA_CENTER) ** 2
                       + (measured[2] - tp_chart.CHROMA_CENTER) ** 2) ** 0.5
    ideal_chroma = ((e["u10"] - tp_chart.CHROMA_CENTER) ** 2
                    + (e["v10"] - tp_chart.CHROMA_CENTER) ** 2) ** 0.5
    sat_pct = (measured_chroma / ideal_chroma * 100.0) if ideal_chroma > 1e-6 else None

    return {
        "id": region["id"],
        "name": region["name"],
        "ideal_yuv10": [e["y10"], e["u10"], e["v10"]],
        "measured_yuv10": list(measured),
        "delta_yuv10": list(delta),
        "sat_pct_vs_ideal": sat_pct,
        "patch_size_px": [x1 - x0, y1 - y0],
        "patch_center_capture_xy": [cx_cap, cy_cap],
    }


import hashlib
import sys


def _ideal_for_md5():
    import tp_synthesize
    return tp_synthesize.synthesize(720, 486)


def _ideal_frame_md5() -> str:
    Y, U, V = _ideal_for_md5()
    h = hashlib.md5()
    h.update(Y.tobytes())
    h.update(U.tobytes())
    h.update(V.tobytes())
    return h.hexdigest()


def measure(capture_path: str, frame_index: int) -> Dict[str, Any]:
    Y, U, V, meta = extract_frame(capture_path, frame_index)
    Y_p, U_p, V_p, padding = pad_to_486(Y, U, V)
    meta["raster_processed"] = [720, 486]
    meta["padding_offsets"] = padding

    if meta["progressive_warning"]:
        print(
            f"WARNING: source field_order={meta['field_order']} indicates "
            f"progressive scan; SW2 analysis assumes interlaced source.",
            file=sys.stderr,
        )

    reg = tp_register.register(Y_p)
    meta["registration"] = {
        "affine": reg["affine_matrix"].tolist() if reg["affine_matrix"] is not None else None,
        "residuals_px": reg["residuals_px"],
        "inliers": reg["inliers"],
        "total": reg["total"],
        "landmarks_used": reg["landmarks_used"],
        "quality_flag": reg["quality_flag"],
        "quality_reason": reg.get("quality_reason"),
    }

    if reg["affine_matrix"] is None:
        raise RuntimeError(
            f"registration failed for {capture_path}: "
            f"{reg.get('quality_reason', 'no affine')}"
        )

    M = reg["affine_matrix"]
    tartan = [sample_region(Y_p, U_p, V_p, r, M) for r in tp_chart.TARTAN_REGIONS]
    grays_raw = [sample_region(Y_p, U_p, V_p, r, M) for r in tp_chart.GRAY_REGIONS]
    # Reshape gray records to the schema in the spec (flat ideal_y10 + delta_y10).
    grays = []
    for r, raw in zip(tp_chart.GRAY_REGIONS, grays_raw):
        grays.append({
            "id": r["id"],
            "name": r["name"],
            "ideal_y10": r["expected"]["y10"],
            "measured_y10": raw["measured_yuv10"][0],
            "delta_y10": raw["delta_yuv10"][0],
            "u10": raw["measured_yuv10"][1],
            "v10": raw["measured_yuv10"][2],
            "patch_size_px": raw["patch_size_px"],
        })

    meta["ideal_frame_md5"] = _ideal_frame_md5()
    meta["tp_chart_version"] = tp_chart.TP_CHART_VERSION
    meta["tool_version"] = _TOOL_VERSION

    return {"_meta": meta, "tartan": tartan, "grays": grays}


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="path to capture (mov/avi/mkv/...)")
    p.add_argument("--frame", type=int, default=60, help="frame index (default 60)")
    p.add_argument("--output", required=True, help="output JSON path")
    args = p.parse_args()
    data = measure(args.capture, args.frame)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(
        f"wrote {args.output}: "
        f"registration={data['_meta']['registration']['quality_flag']}, "
        f"residuals_mean={data['_meta']['registration']['residuals_px']['mean']:.2f}px"
    )


if __name__ == "__main__":
    _main()
