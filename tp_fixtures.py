"""tp_fixtures: test-only fixture generator with ground-truth fiducial
positions for Stage 2 detector validation.

NOT imported by production code. Tests import this to validate per-
detector accuracy against per-variant thresholds (see Stage 2 design
spec).
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

import tp_chart
import tp_synthesize


def _build_identity_ground_truth() -> Dict[str, Any]:
    """Ground truth for the ideal frame at identity (no degradation)."""
    grids = {
        lm["id"]: (lm["ideal_x"], lm["ideal_y"])
        for lm in tp_chart.GRID_LANDMARKS
    }
    triangles = {}
    for tri in tp_chart.BOUNDARY_TRIANGLES:
        triangles[tri["id"]] = {
            "back_corner_1": tri["ideal_back_corner_1"],
            "back_corner_2": tri["ideal_back_corner_2"],
            "back_midpoint": tri["ideal_back_midpoint"],
            "apex":          tri["ideal_apex"],
            "apex_visible":  True,
        }
    cross = {
        "center": (tp_chart.REGISTRATION_CROSS["ideal_x"],
                   tp_chart.REGISTRATION_CROSS["ideal_y"]),
    }
    circle = {
        "center": (tp_chart.BLACK_CIRCLE["ideal_cx"],
                   tp_chart.BLACK_CIRCLE["ideal_cy"]),
        "radius": tp_chart.BLACK_CIRCLE["expected_radius_px"],
    }
    return {
        "grid_intersections": grids,
        "triangles": triangles,
        "cross": cross,
        "circle": circle,
    }


def _shift_planes(Y, U, V, dx, dy):
    """Roll planes by (dx, dy) with grey-background fill on the exposed
    rows/columns. dx is in Y-coords (full-rate); chroma dx is dx // 2 since
    yuv422p10le subsamples horizontally."""
    grey_y = tp_chart.GREY_BACKGROUND_Y10
    grey_c = tp_chart.CHROMA_CENTER

    def _shift(plane, dx_p, dy_p, fill):
        h, w = plane.shape
        out = np.full_like(plane, fill)
        sx0 = max(0, -dx_p); sx1 = min(w, w - dx_p)
        sy0 = max(0, -dy_p); sy1 = min(h, h - dy_p)
        dx0 = max(0, dx_p); dy0 = max(0, dy_p)
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = plane[sy0:sy1, sx0:sx1]
        return out

    Y = _shift(Y, dx, dy, grey_y)
    U = _shift(U, dx // 2, dy, grey_c)
    V = _shift(V, dx // 2, dy, grey_c)
    return Y, U, V


def _apply_shift_to_ground_truth(gt, dx, dy):
    """Translate every ground-truth position by (dx, dy)."""
    def _t(pt):
        return (pt[0] + dx, pt[1] + dy)
    out = {
        "grid_intersections": {k: _t(v) for k, v in gt["grid_intersections"].items()},
        "triangles": {},
        "cross": {"center": _t(gt["cross"]["center"])},
        "circle": {
            "center": _t(gt["circle"]["center"]),
            "radius": gt["circle"]["radius"],
        },
    }
    for tid, t in gt["triangles"].items():
        out["triangles"][tid] = {
            "back_corner_1": _t(t["back_corner_1"]),
            "back_corner_2": _t(t["back_corner_2"]),
            "back_midpoint": _t(t["back_midpoint"]),
            "apex":          _t(t["apex"]),
            "apex_visible":  t["apex_visible"],
        }
    return out


def _rotation_matrix_about_center(theta_deg, cx=360.0, cy=243.0):
    """2D rotation matrix as a 2x3 affine for use with cv2.warpAffine and
    for transforming ground-truth points. Uses cv2's convention: positive
    angle rotates the image CCW on screen (y-down)."""
    import cv2
    return cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)


def _rotate_planes(Y, U, V, theta_deg):
    import cv2
    grey_y = int(tp_chart.GREY_BACKGROUND_Y10)
    grey_c = int(tp_chart.CHROMA_CENTER)
    h, w = Y.shape
    M = _rotation_matrix_about_center(theta_deg, cx=w / 2.0, cy=h / 2.0)
    Y2 = cv2.warpAffine(Y, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=grey_y, flags=cv2.INTER_LINEAR)
    h2, w2 = U.shape
    Mc = _rotation_matrix_about_center(theta_deg, cx=w2 / 2.0, cy=h2 / 2.0)
    U2 = cv2.warpAffine(U, Mc, (w2, h2), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=grey_c, flags=cv2.INTER_LINEAR)
    V2 = cv2.warpAffine(V, Mc, (w2, h2), borderMode=cv2.BORDER_CONSTANT,
                        borderValue=grey_c, flags=cv2.INTER_LINEAR)
    return Y2, U2, V2


def _apply_rotation_to_ground_truth(gt, theta_deg, cx=360.0, cy=243.0):
    M = _rotation_matrix_about_center(theta_deg, cx, cy)
    def _t(pt):
        x, y = pt
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return (float(nx), float(ny))
    out = {
        "grid_intersections": {k: _t(v) for k, v in gt["grid_intersections"].items()},
        "triangles": {},
        "cross": {"center": _t(gt["cross"]["center"])},
        "circle": {
            "center": _t(gt["circle"]["center"]),
            "radius": gt["circle"]["radius"],
        },
    }
    for tid, t in gt["triangles"].items():
        out["triangles"][tid] = {
            "back_corner_1": _t(t["back_corner_1"]),
            "back_corner_2": _t(t["back_corner_2"]),
            "back_midpoint": _t(t["back_midpoint"]),
            "apex":          _t(t["apex"]),
            "apex_visible":  t["apex_visible"],
        }
    return out


def synthesize_with_ground_truth(
    width: int = 720,
    height: int = 486,
    shift: Tuple[int, int] = (0, 0),
    rotation_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    Y, U, V = tp_synthesize.synthesize(width, height)
    gt = _build_identity_ground_truth()
    if rotation_deg != 0.0:
        Y, U, V = _rotate_planes(Y, U, V, rotation_deg)
        gt = _apply_rotation_to_ground_truth(gt, rotation_deg,
                                             cx=width / 2.0, cy=height / 2.0)
    dx, dy = shift
    if dx != 0 or dy != 0:
        Y, U, V = _shift_planes(Y, U, V, dx, dy)
        gt = _apply_shift_to_ground_truth(gt, dx, dy)
    return Y, U, V, gt
