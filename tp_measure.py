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
    if height >= 486:
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
