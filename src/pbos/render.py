"""Overlay drawing for the tracking video: floor rings, P1-P4 tags, court map, distance panel.

Styled after the reference badminton HUD (dark translucent cards, small caps labels, `~` on every
estimate). Phase 6 replaces this with the full HUD renderer driven by HudState; this module draws
straight from tracker state.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import court
from .calibration import Calibration

# BGR. P1 lime, P2 teal, P3 violet, P4 pink, as in the reference.
SLOT_COLORS = {"P1": (50, 240, 200), "P2": (200, 200, 40), "P3": (255, 140, 170), "P4": (150, 110, 255)}
PANEL_BG = (24, 22, 20)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.78) -> None:
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(np.full_like(roi, PANEL_BG), alpha, roi, 1 - alpha, 0)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (70, 70, 70), 1, cv2.LINE_AA)


def _text(img, s, org, scale=0.45, color=TEXT, thick=1) -> None:
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_floor_ring(img: np.ndarray, calib: Calibration, xy, color, radius_ft: float = 1.4) -> None:
    """A circle on the court plane, drawn in perspective (the ellipses under players' feet)."""
    a = np.linspace(0, 2 * np.pi, 48)
    pts, ok = calib.court_to_image(np.stack([xy[0] + radius_ft * np.cos(a), xy[1] + radius_ft * np.sin(a)], 1))
    if not ok.all():
        return
    poly = [np.round(pts).astype(np.int32)]
    glow = img.copy()
    cv2.polylines(glow, poly, True, color, 6, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35, img, 0.65, 0, dst=img)
    cv2.polylines(img, poly, True, color, 2, cv2.LINE_AA)


def draw_tag(img: np.ndarray, box: np.ndarray, label: str, color) -> None:
    x1, y1 = int(box[0]), int(box[1])
    w, h = 40, 22
    x, y = max(x1, 0), max(y1 - h - 6, 0)
    _panel(img, x, y, w, h, alpha=0.85)
    cv2.line(img, (x + 4, y + 4), (x + 4, y + h - 4), color, 2)
    _text(img, label, (x + 10, y + 16), 0.5, TEXT, 1)


def draw_court_map(img: np.ndarray, players: dict[str, np.ndarray], x: int, y: int, px_per_ft: float = 4.0) -> None:
    cw, ch = int(court.WIDTH_FT * px_per_ft), int(court.LENGTH_FT * px_per_ft)
    pad, top = 14, 26
    w, h = cw + 2 * pad, ch + top + pad + 16
    _panel(img, x, y, w, h)
    _text(img, "COURT MAP", (x + 10, y + 17), 0.42, TEXT)
    ox, oy = x + pad, y + top

    def p(cx, cy):  # far baseline at the top
        return int(round(ox + cx * px_per_ft)), int(round(oy + (court.LENGTH_FT - cy) * px_per_ft))

    for a, b in court.LINES:
        cv2.line(img, p(*a), p(*b), (170, 170, 170), 1, cv2.LINE_AA)
    cv2.line(img, p(*court.NET[0]), p(*court.NET[1]), (210, 210, 210), 1, cv2.LINE_AA)
    for slot, xy in players.items():
        cx = float(np.clip(xy[0], -2, court.WIDTH_FT + 2))
        cy = float(np.clip(xy[1], -3, court.LENGTH_FT + 3))
        cv2.circle(img, p(cx, cy), 5, SLOT_COLORS[slot], -1, cv2.LINE_AA)
        _text(img, slot[1], (p(cx, cy)[0] - 3, p(cx, cy)[1] + 4), 0.32, (20, 20, 20))
    _text(img, "PLAYER POSITIONS", (x + 10, y + h - 6), 0.32, MUTED)


def draw_distance_panel(img: np.ndarray, distances: dict[str, float], x: int, y: int) -> None:
    w, h = 250, 34 + 26 * 4 + 10
    _panel(img, x, y, w, h)
    _text(img, "DISTANCE COVERED", (x + 12, y + 20), 0.5, TEXT, 1)
    _text(img, "SESSION - COURT ESTIMATE", (x + 12, y + 34), 0.33, MUTED)
    for i, slot in enumerate(("P1", "P2", "P3", "P4")):
        yy = y + 58 + 26 * i
        cv2.circle(img, (x + 18, yy - 5), 4, SLOT_COLORS[slot], -1, cv2.LINE_AA)
        _text(img, slot, (x + 30, yy), 0.5, TEXT)
        val = f"~{distances[slot]:.0f}" if slot in distances else "--"
        _text(img, val, (x + 150, yy), 0.62, SLOT_COLORS[slot] if slot in distances else MUTED, 2)
        _text(img, "ft", (x + 215, yy), 0.42, MUTED)


def render_frame(frame: np.ndarray, calib: Calibration, players: dict[str, dict], distances: dict[str, float],
                 camera_status: str = "steady") -> np.ndarray:
    """players: slot -> {"xy": court ft, "box": px box or None}."""
    img = frame.copy()
    for slot, p in players.items():
        draw_floor_ring(img, calib, p["xy"], SLOT_COLORS[slot])
    for slot, p in players.items():
        if p.get("box") is not None:
            draw_tag(img, p["box"], slot, SLOT_COLORS[slot])
    H, W = img.shape[:2]
    draw_distance_panel(img, distances, 16, H - 160)
    draw_court_map(img, {s: p["xy"] for s, p in players.items()}, W - 16 - 108, H - 16 - 234)
    footer = "ESTIMATED MOTION - pickleballOS"
    if camera_status == "rebased":
        footer = "CAMERA MOVED - CALIBRATION RE-ALIGNED - " + footer
    elif camera_status == "lost":
        footer = "CAMERA VIEW LOST - POSITIONS PAUSED"
    _text(img, footer, (16, H - 6), 0.36, (200, 200, 200))
    return img
