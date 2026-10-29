"""Interactive click-to-calibrate window (OpenCV HighGUI).

Keys: left-click = place the highlighted keypoint, s = skip it (not visible), u = undo,
Enter = fit with what you have (>= 4), r = restart, q / Esc = quit without saving.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import court
from .calibration import MIN_POINTS, NET_COLOR, Calibration, CalibrationError, verification_image

WINDOW = "pbos calibrate"
LOUPE_PX = 180
LOUPE_ZOOM = 4
CURRENT = (0, 255, 255)
DONE = (80, 255, 120)
PENDING = (130, 130, 130)


def _diagram(current: str | None, done: set[str], w: int = 120, h: int = 240) -> np.ndarray:
    """Small top-down court with the keypoint to click highlighted. Camera side is at the bottom."""
    pad = 12
    img = np.full((h, w, 3), 25, np.uint8)
    sx = (w - 2 * pad) / court.WIDTH_FT
    sy = (h - 2 * pad - 10) / court.LENGTH_FT

    def p(x, y):
        return int(pad + x * sx), int(h - pad - 10 - y * sy)

    for a, b in court.LINES:
        cv2.line(img, p(*a), p(*b), (170, 170, 170), 1)
    cv2.line(img, p(*court.NET[0]), p(*court.NET[1]), NET_COLOR, 1)
    for name, xy in court.KEYPOINTS.items():
        color = CURRENT if name == current else DONE if name in done else PENDING
        cv2.circle(img, p(*xy), 6 if name == current else 3, color, -1)
    cv2.putText(img, "camera side", (18, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return img


def _loupe(frame: np.ndarray, x: int, y: int) -> np.ndarray:
    half = LOUPE_PX // (2 * LOUPE_ZOOM)
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = padded[y:y + 2 * half, x:x + 2 * half]
    big = cv2.resize(crop, (LOUPE_PX, LOUPE_PX), interpolation=cv2.INTER_NEAREST)
    c = LOUPE_PX // 2
    cv2.line(big, (c, 0), (c, LOUPE_PX), CURRENT, 1)
    cv2.line(big, (0, c), (LOUPE_PX, c), CURRENT, 1)
    cv2.rectangle(big, (0, 0), (LOUPE_PX - 1, LOUPE_PX - 1), (255, 255, 255), 1)
    return big


def _paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    dst[y:y + h, x:x + w] = src


def _render_click_view(frame, names, history, points, mouse, message) -> np.ndarray:
    view = frame.copy()
    for name, (u, v) in points.items():
        cv2.circle(view, (int(u), int(v)), 5, DONE, 2, cv2.LINE_AA)
        cv2.putText(view, name, (int(u) + 7, int(v) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DONE, 1, cv2.LINE_AA)

    current = names[len(history)] if len(history) < len(names) else None
    if current:
        text = f"Click: {current}  ({len(history) + 1}/{len(names)})   s=skip  u=undo  Enter=fit({len(points)} pts)  q=quit"
    else:
        text = "All keypoints visited. Enter=fit  u=undo  r=restart  q=quit"
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(view, message or text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (80, 80, 255) if message else (255, 255, 255), 1, cv2.LINE_AA)

    diagram = _diagram(current, set(points))
    _paste(view, diagram, view.shape[1] - diagram.shape[1] - 10, 40)

    if mouse is not None:
        x, y = mouse
        lp = _loupe(frame, x, y)
        # Put the loupe in the bottom corner away from the cursor.
        lx = 10 if x > view.shape[1] // 2 else view.shape[1] - LOUPE_PX - 10
        _paste(view, lp, lx, view.shape[0] - LOUPE_PX - 10)
    return view


def run(frame: np.ndarray, court_id: str = "court1") -> Calibration | None:
    """Show the click UI; return the accepted Calibration, or None if the user quit."""
    names = list(court.KEYPOINTS)
    history: list[str] = []  # keypoint names visited (clicked or skipped), in order
    points: dict[str, tuple[float, float]] = {}
    state = {"mouse": None, "message": ""}

    def on_mouse(event, x, y, _flags, _param):
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(history) < len(names):
            name = names[len(history)]
            points[name] = (float(x), float(y))
            history.append(name)
            state["message"] = ""

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, frame.shape[1], frame.shape[0])
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW, _render_click_view(frame, names, history, points, state["mouse"], state["message"]))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key == ord("s") and len(history) < len(names):
                history.append(names[len(history)])
            elif key == ord("u") and history:
                points.pop(history.pop(), None)
            elif key == ord("r"):
                history.clear()
                points.clear()
            elif key == 13:
                if len(points) < MIN_POINTS:
                    state["message"] = f"Need at least {MIN_POINTS} points (have {len(points)})."
                    continue
                try:
                    calib = Calibration.from_points(points, (frame.shape[1], frame.shape[0]), court_id)
                except CalibrationError as e:
                    state["message"] = str(e)
                    continue
                accepted = _review(frame, calib)
                if accepted:
                    return calib
                cv2.setMouseCallback(WINDOW, on_mouse)
    finally:
        cv2.destroyWindow(WINDOW)


def _review(frame: np.ndarray, calib: Calibration) -> bool:
    """Show the verification view. Enter accepts, any of u/r/Esc goes back to clicking."""
    img = verification_image(frame, calib)
    cv2.putText(img, "Enter = save   u = back to clicking", (10, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.resizeWindow(WINDOW, img.shape[1], img.shape[0])
    while True:
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            return True
        if key in (ord("u"), ord("r"), ord("q"), 27):
            cv2.resizeWindow(WINDOW, frame.shape[1], frame.shape[0])
            return False
