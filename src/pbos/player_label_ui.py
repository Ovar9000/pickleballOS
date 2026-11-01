"""Player position ground truth: click where each player stands on random frames.

For every player of *this* court (not other courts, not spectators), click the floor midway between
their feet. Left-click when that point is inside the painted court lines (on a line counts as in),
right-click when it is outside. Shift+click when the feet are hidden (behind the net post, another
player): click your best guess; it counts for "was the player found" but not for position error.
Judge in/out from the painted lines only; that is what makes the label independent of calibration.

Keys: u/Backspace = undo last point · c = clear frame · Enter/Space = frame done (saves) · s = skip
(cut-aways, unclear frames) · b = back · q/Esc = save and quit. A loupe magnifies the area under the
cursor. Progress is saved after every frame, so a session can be resumed any time; --review walks the
frames already labelled to check or fix them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .label_ui import _loupe, _read
from .playereval import FootPoint, PlayerFrame, load_player_labels, save_player_labels

WINDOW = "pbos player-label"
IN, OUT, HIDDEN = (60, 220, 60), (60, 60, 230), (180, 180, 180)


def _draw_point(view, p: FootPoint, k: int) -> None:
    c = HIDDEN if p.hidden else (IN if p.inside else OUT)
    q = (int(round(p.u)), int(round(p.v)))
    cv2.drawMarker(view, q, c, cv2.MARKER_TILTED_CROSS if p.hidden else cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    tag = f"{k + 1} " + ("hidden" if p.hidden else ("in" if p.inside else "out"))
    cv2.putText(view, tag, (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)


def run(video: str, out: str, frames: list[int], fps: float, review: bool = False) -> int:
    """Label the given frames in order. Returns how many labelled frames the file holds afterwards."""
    done: dict[int, PlayerFrame] = {}
    if Path(out).exists():
        done = {f.frame: f for f in load_player_labels(out)[1]}
    todo = [f for f in frames if (f in done) == review]
    cap = cv2.VideoCapture(video)
    state = {"mouse": None, "click": None}

    def on_mouse(event, x, y, flags, _p):
        state["mouse"] = (x, y)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            hidden = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            state["click"] = FootPoint(float(x), float(y), None if hidden else event == cv2.EVENT_LBUTTONDOWN, hidden)

    def save():
        save_player_labels(out, video, fps, [done[f] for f in sorted(done)])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    i, history = 0, []
    try:
        while 0 <= i < len(todo):
            f = todo[i]
            img = _read(cap, f)
            if img is None:
                i += 1
                continue
            points = list(done[f].points) if f in done else []
            state["click"] = None
            decided = None
            while decided is None:
                if state["click"] is not None:
                    points.append(state["click"])
                    state["click"] = None
                view = img.copy()
                for k, p in enumerate(points):
                    _draw_point(view, p, k)
                bar = (f"frame {f} ({f / fps:.2f}s)  {len(done)} done, {len(todo) - i} to go  |  {len(points)} players  "
                       "L-click=in  R-click=out  shift+click=feet hidden  u=undo  Enter=done  s=skip  b=back  q=quit")
                cv2.rectangle(view, (0, 0), (view.shape[1], 28), (20, 20, 20), -1)
                cv2.putText(view, bar, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if state["mouse"]:
                    mx, my = state["mouse"]
                    lp = _loupe(img, mx, my)
                    lx = 10 if mx > view.shape[1] // 2 else view.shape[1] - lp.shape[1] - 10
                    view[view.shape[0] - lp.shape[0] - 10: view.shape[0] - 10, lx:lx + lp.shape[1]] = lp
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("u"), 8) and points:
                    points.pop()
                elif key == ord("c"):
                    points = []
                elif key in (13, 32):
                    decided = PlayerFrame(f, points)
                elif key == ord("s"):
                    decided = "skip"
                elif key == ord("b") and history:
                    history.pop()
                    i -= 2
                    decided = "back"
                elif key in (ord("q"), 27):
                    return len(done)
            if isinstance(decided, PlayerFrame):
                done[f] = decided
                history.append(f)
                save()
            i += 1
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW)
        save()
    return len(done)
