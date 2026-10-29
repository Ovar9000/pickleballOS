"""Pickleball court geometry in court feet.

Origin is the near-left baseline corner as seen from the camera. x runs across the
court (0..20 ft), y runs along it (0..44 ft), the net is at y = 22.
"""

from __future__ import annotations

WIDTH_FT = 20.0
LENGTH_FT = 44.0
NET_Y = LENGTH_FT / 2
KITCHEN_DEPTH_FT = 7.0
NEAR_KITCHEN_Y = NET_Y - KITCHEN_DEPTH_FT
FAR_KITCHEN_Y = NET_Y + KITCHEN_DEPTH_FT
CENTER_X = WIDTH_FT / 2

# The twelve painted line intersections, in the order the calibration tool asks for them.
KEYPOINTS: dict[str, tuple[float, float]] = {
    "near_left_corner": (0.0, 0.0),
    "near_center_baseline": (CENTER_X, 0.0),
    "near_right_corner": (WIDTH_FT, 0.0),
    "near_left_kitchen": (0.0, NEAR_KITCHEN_Y),
    "near_center_kitchen": (CENTER_X, NEAR_KITCHEN_Y),
    "near_right_kitchen": (WIDTH_FT, NEAR_KITCHEN_Y),
    "far_left_kitchen": (0.0, FAR_KITCHEN_Y),
    "far_center_kitchen": (CENTER_X, FAR_KITCHEN_Y),
    "far_right_kitchen": (WIDTH_FT, FAR_KITCHEN_Y),
    "far_left_corner": (0.0, LENGTH_FT),
    "far_center_baseline": (CENTER_X, LENGTH_FT),
    "far_right_corner": (WIDTH_FT, LENGTH_FT),
}

Segment = tuple[tuple[float, float], tuple[float, float]]

# Painted lines (the net is listed separately; it is not a ground line).
LINES: list[Segment] = [
    ((0.0, 0.0), (0.0, LENGTH_FT)),
    ((WIDTH_FT, 0.0), (WIDTH_FT, LENGTH_FT)),
    ((0.0, 0.0), (WIDTH_FT, 0.0)),
    ((0.0, LENGTH_FT), (WIDTH_FT, LENGTH_FT)),
    ((0.0, NEAR_KITCHEN_Y), (WIDTH_FT, NEAR_KITCHEN_Y)),
    ((0.0, FAR_KITCHEN_Y), (WIDTH_FT, FAR_KITCHEN_Y)),
    ((CENTER_X, 0.0), (CENTER_X, NEAR_KITCHEN_Y)),
    ((CENTER_X, FAR_KITCHEN_Y), (CENTER_X, LENGTH_FT)),
]
NET: Segment = ((0.0, NET_Y), (WIDTH_FT, NET_Y))


def in_court(x: float, y: float, side_margin: float = 0.0, end_margin: float = 0.0) -> bool:
    """True if (x, y) lies inside the court expanded by the given margins (feet)."""
    return -side_margin <= x <= WIDTH_FT + side_margin and -end_margin <= y <= LENGTH_FT + end_margin


def in_kitchen(x: float, y: float) -> bool:
    """True if (x, y) is inside either non-volley zone (lines count as in)."""
    return 0.0 <= x <= WIDTH_FT and NEAR_KITCHEN_Y <= y <= FAR_KITCHEN_Y
