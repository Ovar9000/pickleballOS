"""Ball detection, tracking and 3D flight estimation from one camera.

Detection is classical and causal: a pixel is a ball candidate if it differs from *both* of the
two previous frames (so the ball's ghost at its old position is rejected) and is optic-yellow, in
a blob of ball-like size. On a fixed camera this needs no training data and runs in a few ms.

The court homography cannot place a ball that is in the air: it only maps ground points. Instead,
each flight between two velocity changes (paddle contact or bounce) is fitted with a ballistic
arc under known gravity, using the full 3D camera. Gravity is what makes this solvable from one
view: how fast the ball "falls" in the image depends on how far away it is. The fit is linear least
squares in the six unknowns (start position, start velocity). Air drag is ignored, so speeds and
heights are estimates; bounces are placed with the homography from the image point at touchdown,
which is exact because the ball is on the ground there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import court
from .calibration import Calibration
from .camera3d import Camera3D

G_FT = 32.174  # ft/s^2
FPS_TO_MPH = 3600 / 5280


@dataclass
class BallCandidate:
    uv: np.ndarray
    area: float


class BallDetector:
    def __init__(self, hsv_lo=(22, 70, 140), hsv_hi=(45, 255, 255), diff_thr: int = 18,
                 min_area: int = 2, max_area: int = 150, max_side: int = 25):
        self.hsv_lo, self.hsv_hi = hsv_lo, hsv_hi
        self.diff_thr = diff_thr
        self.min_area, self.max_area, self.max_side = min_area, max_area, max_side
        self.roi: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1; None = whole frame
        self._prev: list[np.ndarray] = []

    def reset(self) -> None:
        self._prev.clear()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self.roi = roi
        self.reset()

    def __call__(self, frame: np.ndarray) -> list[BallCandidate]:
        x0, y0 = 0, 0
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            frame = frame[y0:y1, x0:x1]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        prev = self._prev
        self._prev = (prev + [gray])[-2:]
        if len(prev) < 2 or prev[0].shape != gray.shape:
            return []
        _, m1 = cv2.threshold(cv2.absdiff(gray, prev[-1]), self.diff_thr, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(cv2.absdiff(gray, prev[-2]), self.diff_thr, 255, cv2.THRESH_BINARY)
        moving = cv2.bitwise_and(m1, m2)
        yellow = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.hsv_lo, self.hsv_hi)
        mask = cv2.bitwise_and(moving, cv2.dilate(yellow, np.ones((3, 3), np.uint8)))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            area, w, h = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if self.min_area <= area <= self.max_area and max(w, h) <= self.max_side:
                out.append(BallCandidate(cents[i].astype(np.float64) + (x0, y0), float(area)))
        return out


@dataclass
class Flight:
    t0: float
    p0: np.ndarray  # ft, (x, y, height) at t0
    v0: np.ndarray  # ft/s at t0
    rms_px: float

    def position(self, t: float) -> np.ndarray:
        tau = t - self.t0
        return self.p0 + self.v0 * tau + np.array([0.0, 0.0, -0.5 * G_FT * tau * tau])

    @property
    def horizontal_mph(self) -> float:
        return float(np.linalg.norm(self.v0[:2]) * FPS_TO_MPH)

    def plausible(self) -> bool:
        return (self.rms_px < 4.0 and -2.0 < self.p0[2] < 25.0 and self.horizontal_mph < 90.0
                and -30 < self.p0[0] < 50 and -40 < self.p0[1] < 85)


def fit_ballistic(ts, uvs, P: np.ndarray, iterations: int = 3) -> Flight:
    """Least-squares ballistic arc through 2D detections, given the 3x4 camera matrix P."""
    ts = np.asarray(ts, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    t0 = ts[0]
    tau = ts - t0
    w = np.ones(len(ts))
    x = np.zeros(6)
    for _ in range(iterations):
        rows, rhs = [], []
        for i, (u, v) in enumerate(uvs):
            c = np.array([0.0, 0.0, -0.5 * G_FT * tau[i] ** 2])
            for row in (P[0] - u * P[2], P[1] - v * P[2]):
                a, b = row[:3], row[3]
                rows.append(w[i] * np.concatenate([a, a * tau[i]]))
                rhs.append(-w[i] * (a @ c + b))
        x, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        # Re-weight by 1/depth so the algebraic error approximates pixel error.
        X = x[:3] + np.outer(tau, x[3:]) + np.stack([np.zeros_like(tau), np.zeros_like(tau), -0.5 * G_FT * tau**2], 1)
        depth = (np.hstack([X, np.ones((len(X), 1))]) @ P[2])
        w = 1.0 / np.maximum(np.abs(depth), 1e-6)
    flight = Flight(t0, x[:3], x[3:], 0.0)
    proj = np.array([flight.position(t) for t in ts])
    h = np.hstack([proj, np.ones((len(proj), 1))]) @ P.T
    flight.rms_px = float(np.sqrt(np.mean(np.sum((h[:, :2] / h[:, 2:3] - uvs) ** 2, axis=1))))
    return flight


@dataclass
class BallEvent:
    kind: str  # "bounce" | "hit"
    t: float
    uv: np.ndarray
    court_xy: np.ndarray | None  # bounce: exact ground point; hit: estimate from the 3D arc, if fitted
    speed_mph: float | None  # horizontal speed of the flight that ended here, if it was fitted


@dataclass
class BallState:
    uv: np.ndarray | None = None  # detected this frame
    trail: list = field(default_factory=list)
    xyz: np.ndarray | None = None  # 3D estimate (ft) from the current flight fit
    flight_mph: float | None = None  # current flight, horizontal
    last_mph: float | None = None  # last completed fitted flight, horizontal
    events: list[BallEvent] = field(default_factory=list)  # emitted this frame


class BallTracker:
    """Links candidates into a track and splits it into flights at velocity changes."""

    def __init__(self, calib: Calibration, camera: Camera3D | None, min_fit: int = 5, break_px: float = 7.0,
                 gate_px: float = 45.0, max_gap_s: float = 0.3, acquire_mask: np.ndarray | None = None):
        self.calib = calib
        self.camera = camera
        self.min_fit = min_fit
        self.break_px = break_px
        self.gate_px = gate_px
        self.max_gap_s = max_gap_s
        self.acquire_mask = acquire_mask if acquire_mask is not None else acquisition_mask(calib)
        self._seg: list[tuple[float, np.ndarray]] = []
        self._pending: list[tuple[float, np.ndarray]] = []
        self._flight: Flight | None = None
        self._trail: list[np.ndarray] = []
        self.last_mph: float | None = None

    def set_geometry(self, calib: Calibration, camera: Camera3D | None) -> None:
        """New calibration after the camera moved. The current arc is refitted on the next point."""
        self.calib, self.camera = calib, camera
        self.acquire_mask = acquisition_mask(calib)
        self._flight = None

    def _predict(self, t: float) -> np.ndarray | None:
        if self._flight is not None and self.camera is not None:
            return self.camera.project(self._flight.position(t))[0]
        if len(self._seg) >= 2:
            (t1, p1), (t2, p2) = self._seg[-2], self._seg[-1]
            return p2 + (p2 - p1) * (t - t2) / max(t2 - t1, 1e-6)
        if self._seg:
            return self._seg[-1][1]
        return None

    def _fit(self, pts) -> Flight | None:
        if self.camera is None or len(pts) < self.min_fit:
            return None
        f = fit_ballistic([p[0] for p in pts], [p[1] for p in pts], self.camera.P)
        return f if f.plausible() else None

    def _close_segment(self, t_end: float, uv_end: np.ndarray, events: list[BallEvent]) -> None:
        flight = self._fit(self._seg) if len(self._seg) >= self.min_fit else None
        mph = flight.horizontal_mph if flight else None
        if mph is not None:
            self.last_mph = mph
        # Ending low and falling = bounce (ground point -> exact court position); otherwise a hit.
        is_bounce = False
        if flight is not None:
            z_end = flight.position(t_end)[2]
            vz_end = flight.v0[2] - G_FT * (t_end - flight.t0)
            is_bounce = z_end < 1.5 and vz_end < 0
        if is_bounce:
            xy = self.calib.image_to_court([uv_end])[0]
        else:
            xy = flight.position(t_end)[:2] if flight is not None else None
        events.append(BallEvent("bounce" if is_bounce else "hit", t_end, uv_end, xy, mph))

    def update(self, t: float, candidates: list[BallCandidate]) -> BallState:
        events: list[BallEvent] = []
        if self._seg and t - self._seg[-1][0] > self.max_gap_s:
            self._seg, self._pending, self._flight = [], [], None  # lost: ball dead or occluded too long
            self._trail = []

        pred = self._predict(t)
        chosen = None
        if candidates:
            if pred is not None:
                d = [np.linalg.norm(c.uv - pred) for c in candidates]
                k = int(np.argmin(d))
                gate = self.gate_px if self._flight is None else max(self.gate_px, 4 * self.break_px)
                if d[k] <= gate:
                    chosen = candidates[k].uv
            if chosen is None and not self._seg:
                ok = [c for c in candidates if self._acquirable(c.uv)]
                if ok:
                    chosen = max(ok, key=lambda c: c.area).uv

        if chosen is not None:
            if self._flight is not None and np.linalg.norm(chosen - pred) > self.break_px:
                self._pending.append((t, chosen))
                if len(self._pending) >= 2:
                    # Two points in a row off the arc: the ball changed velocity between the last
                    # good point and the first pending one.
                    self._close_segment(self._seg[-1][0], self._seg[-1][1], events)
                    self._seg, self._pending, self._flight = list(self._pending), [], None
            else:
                if self._pending:  # a single outlier: drop it
                    self._pending = []
                self._seg.append((t, chosen))
                fit = self._fit(self._seg)
                if fit is not None:
                    self._flight = fit
                elif self._flight is not None and len(self._seg) >= self.min_fit:
                    self._flight = None
            self._trail = (self._trail + [chosen])[-10:]

        state = BallState(uv=chosen, trail=list(self._trail), last_mph=self.last_mph, events=events)
        if self._flight is not None:
            state.xyz = self._flight.position(t)
            state.flight_mph = self._flight.horizontal_mph
        return state

    def _acquirable(self, uv: np.ndarray) -> bool:
        u, v = int(round(uv[0])), int(round(uv[1]))
        h, w = self.acquire_mask.shape
        return 0 <= u < w and 0 <= v < h and self.acquire_mask[v, u] > 0


def detection_roi(calib: Calibration, lift_px: int = 260, pad_px: int = 40) -> tuple[int, int, int, int]:
    """Bounding box of where the ball can be: the court plus margins, and high lobs above it."""
    ys, xs = np.nonzero(acquisition_mask(calib, lift_px))
    w, h = calib.image_size
    if len(xs) == 0:
        return 0, 0, w, h
    return (max(int(xs.min()) - pad_px, 0), max(int(ys.min()) - pad_px, 0),
            min(int(xs.max()) + pad_px, w), min(int(ys.max()) + pad_px, h))


def acquisition_mask(calib: Calibration, lift_px: int = 150) -> np.ndarray:
    """Where a new ball track may start: over the court (plus margins) and up to `lift_px` above it,
    since the ball is usually in the air. Balls on neighbouring courts start outside this."""
    ground = calib.court_mask(side_margin=6.0, end_margin=12.0)
    kernel = np.zeros((2 * lift_px + 1, 1), np.uint8)
    kernel[lift_px:, 0] = 1  # dilate upwards only
    return cv2.dilate(ground, kernel, anchor=(0, lift_px))
