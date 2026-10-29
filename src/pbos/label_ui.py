"""Ball ground-truth labelling: review random frames, click the ball or mark it not visible.

What counts as visible: you can see at least part of the ball (a blurred streak counts). Press n when
it is hidden behind a player or the paddle, held in a hand or pocket between points, or out of frame.
Never click where you think the ball is: a guessed position teaches a detector to fire on shirts.

Keys: left-click = ball here (saves, next frame) · Enter/Space = accept the suggestion ring ·
n = ball not visible · s = skip · b = back · q/Esc = save and quit. A loupe magnifies the area
under the cursor. Progress is saved after every frame, so a session can be resumed any time.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from .balldata import BallLabel, load_labels, save_labels

WINDOW = "pbos label"


def _read(cap, frame: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    return img if ok else None


def _suggest(cap, frame: int, make_detector) -> np.ndarray | None:
    """Run a fresh detector on frames t-2, t-1, t and return its strongest candidate."""
    if make_detector is None or frame < 2:
        return None
    det = make_detector()
    cands = []
    for f in (frame - 2, frame - 1, frame):
        img = _read(cap, f)
        if img is None:
            return None
        cands = det(img)
    return max(cands, key=lambda c: c.area).uv if cands else None


def _loupe(img, x, y, size=180, zoom=4):
    half = size // (2 * zoom)
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
    big = cv2.resize(pad[y:y + 2 * half, x:x + 2 * half], (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.line(big, (size // 2, 0), (size // 2, size), (0, 255, 255), 1)
    cv2.line(big, (0, size // 2), (size, size // 2), (0, 255, 255), 1)
    return big


def run(video: str, out: str, frames: list[int], fps: float, make_detector=None, review: bool = False) -> int:
    """Label the given frames (in order). Returns how many labels the file holds afterwards.

    review=True walks frames that already have a label, showing that label as the ring (red when it
    says "not visible"), so earlier answers can be confirmed (Enter) or corrected.
    """
    done: dict[int, BallLabel] = {}
    if Path(out).exists():
        _, prev = load_labels(out)
        done = {lb.frame: lb for lb in prev}
    todo = [f for f in frames if (f in done) == review]
    cap = cv2.VideoCapture(video)
    state = {"mouse": None, "click": None}

    def on_mouse(event, x, y, _f, _p):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (float(x), float(y))

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
            prev = done.get(f) if review else None
            sug = np.array([prev.u, prev.v]) if prev is not None and prev.visible else (
                None if review else _suggest(cap, f, make_detector))
            state["click"] = None
            decided = None
            while decided is None:
                view = img.copy()
                if sug is not None:
                    cv2.circle(view, tuple(np.round(sug).astype(int)), 12, (0, 255, 255), 1, cv2.LINE_AA)
                if prev is not None and not prev.visible:
                    cv2.putText(view, "labelled: NOT VISIBLE (Enter keeps it)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)
                bar = (f"frame {f} ({f / fps:.2f}s)  {len(done)} labelled, {len(todo) - i} to go  |  click=ball  "
                       "Enter=accept  n=hidden/held/out of frame  s=skip  b=back  q=quit")
                cv2.rectangle(view, (0, 0), (view.shape[1], 28), (20, 20, 20), -1)
                cv2.putText(view, bar, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if state["mouse"]:
                    mx, my = state["mouse"]
                    lp = _loupe(img, mx, my)
                    lx = 10 if mx > view.shape[1] // 2 else view.shape[1] - lp.shape[1] - 10
                    view[view.shape[0] - lp.shape[0] - 10: view.shape[0] - 10, lx:lx + lp.shape[1]] = lp
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(20) & 0xFF
                if state["click"] is not None:
                    decided = BallLabel(f, *state["click"], True, "human")
                elif key in (13, 32) and sug is not None:
                    decided = BallLabel(f, float(sug[0]), float(sug[1]), True, "human")
                elif key in (13, 32) and prev is not None:
                    decided = prev  # review: keep the earlier answer
                elif key == ord("n"):
                    decided = BallLabel(f, -1.0, -1.0, False, "human")
                elif key == ord("s"):
                    decided = "skip"
                elif key == ord("b") and history:
                    back_to = history.pop()
                    if not review:  # in review the earlier answer stays until it is replaced
                        done.pop(back_to, None)
                    i -= 2
                    decided = "back"
                elif key in (ord("q"), 27):
                    return len(done)
            if isinstance(decided, BallLabel):
                done[f] = decided
                history.append(f)
                save_labels(out, video, fps, sorted(done.values(), key=lambda lb: lb.frame))
            i += 1
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW)
        save_labels(out, video, fps, sorted(done.values(), key=lambda lb: lb.frame))
    return len(done)


def sample_frames(fps: float, start: float, end: float, n: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    lo, hi = int(start * fps) + 2, int(end * fps)
    return sorted(rng.sample(range(lo, hi), min(n, hi - lo)))
