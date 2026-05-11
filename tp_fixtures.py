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


def synthesize_with_ground_truth(
    width: int = 720,
    height: int = 486,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synthesize the ideal SW2 frame and return per-fiducial ground truth.

    Identity baseline. Subsequent tasks add shift/rotation/noise/blur/clip
    keyword arguments.
    """
    Y, U, V = tp_synthesize.synthesize(width, height)
    gt = _build_identity_ground_truth()
    return Y, U, V, gt
