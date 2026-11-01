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
FT_PER_S_TO_KMH = court.FT_TO_M * 3.6


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
        self._grays: list[np.ndarray] = []  # last three blurred grey ROI crops, oldest first
        self._frame: np.ndarray | None = None  # current colour ROI crop

    def reset(self) -> None:
        self._grays.clear()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self.roi = roi
        self.reset()

    def _blobs(self, grays, colour, diff_thr, hsv_lo, hsv_hi, min_area, max_area, offset) -> list[BallCandidate]:
        cur, p1, p2 = grays[-1], grays[-2], grays[-3]
        _, m1 = cv2.threshold(cv2.absdiff(cur, p1), diff_thr, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(cv2.absdiff(cur, p2), diff_thr, 255, cv2.THRESH_BINARY)
        moving = cv2.bitwise_and(m1, m2)
        yellow = cv2.inRange(cv2.cvtColor(colour, cv2.COLOR_BGR2HSV), hsv_lo, hsv_hi)
        mask = cv2.bitwise_and(moving, cv2.dilate(yellow, np.ones((3, 3), np.uint8)))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            area, w, h = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if min_area <= area <= max_area and max(w, h) <= self.max_side:
                out.append(BallCandidate(cents[i].astype(np.float64) + offset, float(area)))
        return out

    def __call__(self, frame: np.ndarray) -> list[BallCandidate]:
        x0, y0 = 0, 0
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            frame = frame[y0:y1, x0:x1]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if self._grays and self._grays[-1].shape != gray.shape:
            self._grays.clear()
        self._grays = (self._grays + [gray])[-3:]
        self._frame = frame
        if len(self._grays) < 3:
            return []
        return self._blobs(self._grays, frame, self.diff_thr, self.hsv_lo, self.hsv_hi,
                           self.min_area, self.max_area, np.array([x0, y0], dtype=np.float64))

    def search_near(self, uv: np.ndarray, radius: float) -> list[BallCandidate]:
        """Relaxed second look in a window where the tracker expects the ball (call after __call__).

        Motion blur washes out the ball's colour and contrast, so the global thresholds miss it.
        Near a confident prediction a fainter, less saturated blob is still most likely the ball.
        """
        if len(self._grays) < 3 or self._frame is None:
            return []
        x0, y0 = (self.roi[0], self.roi[1]) if self.roi is not None else (0, 0)
        h, w = self._grays[-1].shape
        cx, cy = uv[0] - x0, uv[1] - y0
        r = int(radius)
        a0, b0 = max(int(cx) - r, 0), max(int(cy) - r, 0)
        a1, b1 = min(int(cx) + r, w), min(int(cy) + r, h)
        if a1 - a0 < 4 or b1 - b0 < 4:
            return []
        crops = [g[b0:b1, a0:a1] for g in self._grays]
        return self._blobs(crops, self._frame[b0:b1, a0:a1], max(self.diff_thr // 2, 6), (18, 35, 100),
                           (50, 255, 255), 1, self.max_area, np.array([x0 + a0, y0 + b0], dtype=np.float64))


class LearnedBallDetector:
    """Heatmap ball detector (ballnet.py, exported to ONNX), a drop-in for BallDetector.

    Keeps the last three frames of the ROI, runs the network on (t-2, t-1, t) and returns heatmap
    peaks as candidates. `search_near` re-reads the last heatmap with a lower threshold where the
    tracker expects the ball, mirroring the classical detector's relaxed second look.
    """

    def __init__(self, model_path: str, device: str = "cpu", threshold: float = 0.7,
                 search_threshold: float = 0.2, max_candidates: int = 5):
        from .players import ort_session

        self.session = ort_session(str(model_path), device)
        self.threshold, self.search_threshold, self.max_candidates = threshold, search_threshold, max_candidates
        self.roi: tuple[int, int, int, int] | None = None
        self._frames: list[np.ndarray] = []
        self._heat: np.ndarray | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._heat = None

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        if roi is not None:  # the network needs sizes divisible by 8
            x0, y0, x1, y1 = roi
            roi = (x0, y0, x0 + (x1 - x0) // 8 * 8, y0 + (y1 - y0) // 8 * 8)
        self.roi = roi
        self.reset()

    def _offset(self) -> np.ndarray:
        return np.array(self.roi[:2] if self.roi is not None else (0, 0), dtype=np.float64)

    def _peaks(self, heat: np.ndarray, thr: float, window=None) -> list[BallCandidate]:
        x0 = y0 = 0
        if window is not None:
            x0, y0, x1, y1 = window
            heat = heat[y0:y1, x0:x1]
        if heat.size == 0:
            return []
        local_max = heat >= cv2.dilate(heat, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(local_max & (heat >= thr))
        kept: list[int] = []
        for k in np.argsort(-heat[ys, xs]):  # greedy: drop peaks next to a stronger one (plateaus)
            if all((xs[k] - xs[j]) ** 2 + (ys[k] - ys[j]) ** 2 > 36 for j in kept):
                kept.append(k)
            if len(kept) == self.max_candidates:
                break
        off = self._offset() + (x0, y0)
        out = []
        for k in kept:
            # Sub-pixel position: heat-weighted centroid of the 5x5 neighbourhood.
            a0, b0 = max(xs[k] - 2, 0), max(ys[k] - 2, 0)
            win = heat[b0:ys[k] + 3, a0:xs[k] + 3]
            gy, gx = np.mgrid[0:win.shape[0], 0:win.shape[1]]
            uv = np.array([(gx * win).sum(), (gy * win).sum()]) / max(float(win.sum()), 1e-9) + (a0, b0)
            # `area` carries the confidence, so acquisition prefers the strongest peak.
            out.append(BallCandidate(uv + off, float(heat[ys[k], xs[k]]) * 100))
        return out

    def __call__(self, frame: np.ndarray) -> list[BallCandidate]:
        if self.roi is None:
            h, w = frame.shape[:2]
            self.set_roi((0, 0, w, h))
        x0, y0, x1, y1 = self.roi
        crop = frame[y0:y1, x0:x1]
        if self._frames and self._frames[-1].shape != crop.shape:
            self._frames.clear()
        self._frames = (self._frames + [crop])[-3:]
        if len(self._frames) < 3:
            return []
        stacked = np.concatenate(self._frames, axis=2)[None]  # (1, H, W, 9) uint8
        name = self.session.get_inputs()[0].name
        if name == "frames":  # older export: float NCHW input
            stacked = np.ascontiguousarray(stacked.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        self._heat = self.session.run(None, {name: stacked})[0][0, 0]
        return self._peaks(self._heat, self.threshold)

    def search_near(self, uv: np.ndarray, radius: float) -> list[BallCandidate]:
        if self._heat is None:
            return []
        cx, cy = uv - self._offset()
        r = int(radius)
        h, w = self._heat.shape
        window = (max(int(cx) - r, 0), max(int(cy) - r, 0), min(int(cx) + r, w), min(int(cy) + r, h))
        return self._peaks(self._heat, self.search_threshold, window)


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
    def horizontal_kmh(self) -> float:
        return float(np.linalg.norm(self.v0[:2]) * FT_PER_S_TO_KMH)

    def plausible(self, t_end: float | None = None) -> bool:
        """Small residual, sane speed, and the arc stays around the court (and above the floor)
        for the whole time it was observed."""
        if not (self.rms_px < 4.0 and self.horizontal_kmh < 145.0):
            return False
        for t in (self.t0, t_end if t_end is not None else self.t0):
            x, y, z = self.position(t)
            if not (-12 <= x <= court.WIDTH_FT + 12 and -15 <= y <= court.LENGTH_FT + 15 and -1.0 <= z <= 25.0):
                return False
        return True


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
    speed_kmh: float | None  # horizontal speed of the flight that ended here, if it was fitted


@dataclass
class BallState:
    uv: np.ndarray | None = None  # detected this frame
    predicted_uv: np.ndarray | None = None  # where the fitted arc puts it when not detected (coasting)
    trail: list = field(default_factory=list)
    xyz: np.ndarray | None = None  # 3D estimate (ft) from the current flight fit
    flight_kmh: float | None = None  # current flight, horizontal
    last_kmh: float | None = None  # last completed fitted flight, horizontal
    events: list[BallEvent] = field(default_factory=list)  # emitted this frame


@dataclass(eq=False)  # compared by identity
class _Hypothesis:
    """A side track kept next to the main one, so the tracker can switch to the real ball."""
    pts: list = field(default_factory=list)
    flight: Flight | None = None
    junk: bool = False  # a retired main track: keeps absorbing its blob so it is not picked up again


class BallTracker:
    """Links candidates into a track and splits it into flights at velocity changes.

    Temporal context comes from three places: the fitted arc predicts where the ball will be (and
    lets the track coast through short occlusions); after a hit, when the ball leaves the arc, it
    is searched for around its last position in a radius that grows with elapsed time; and
    `search_hint` tells the detector where a relaxed second look is worthwhile.

    Candidates the main track doesn't take feed side hypotheses. A blob that keeps moving without
    ever fitting a ballistic arc (a shoe, a shirt, a ball on the next court) is not the ball in
    play, so a main track like that is retired, and the tracker switches to a hypothesis that does
    fit an arc whenever the main track has none or has lost the ball. Coasting on an arc is limited
    to about as long as the arc was observed, since a short fit extrapolates badly.
    """

    def __init__(self, calib: Calibration, camera: Camera3D | None, min_fit: int = 5, break_px: float = 7.0,
                 gate_px: float = 45.0, max_gap_s: float = 0.3, max_coast_s: float = 0.5, min_coast_s: float = 0.15,
                 reach_base_px: float = 20.0, reach_px_per_s: float = 1500.0, reach_max_px: float = 160.0,
                 speed_min_points: int = 8, speed_max_kmh: float = 110.0, stale_points: int = 8,
                 max_hypotheses: int = 8, hyp_window: int = 12, acquire_mask: np.ndarray | None = None):
        self.calib = calib
        self.speed_min_points = speed_min_points
        self.speed_max_kmh = speed_max_kmh
        self.camera = camera
        self.min_fit = min_fit
        self.break_px = break_px
        self.gate_px = gate_px
        self.max_gap_s = max_gap_s
        self.max_coast_s = max_coast_s
        self.min_coast_s = min_coast_s
        self.reach_base_px = reach_base_px
        self.reach_px_per_s = reach_px_per_s
        self.reach_max_px = reach_max_px
        self.stale_points = stale_points
        self.max_hypotheses = max_hypotheses
        self.hyp_window = hyp_window
        self.acquire_mask = acquire_mask if acquire_mask is not None else acquisition_mask(calib)
        self._seg: list[tuple[float, np.ndarray]] = []
        self._pending: list[tuple[float, np.ndarray]] = []
        self._flight: Flight | None = None
        self._trail: list[np.ndarray] = []
        self._hyps: list[_Hypothesis] = []
        self.last_kmh: float | None = None

    def set_geometry(self, calib: Calibration, camera: Camera3D | None) -> None:
        """New calibration after the camera moved. The current arc is refitted on the next point."""
        self.calib, self.camera = calib, camera
        self.acquire_mask = acquisition_mask(calib)
        self._flight = None
        for h in self._hyps:
            h.flight = None

    def _project(self, flight: Flight, t: float) -> np.ndarray:
        return self.camera.project(flight.position(t))[0]

    def _predict(self, t: float) -> np.ndarray | None:
        if self._flight is not None and self.camera is not None:
            return self._project(self._flight, t)
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
        return f if f.plausible(pts[-1][0]) else None

    def _close_segment(self, t_end: float, uv_end: np.ndarray, events: list[BallEvent]) -> None:
        pts = [p for p in self._seg if p[0] <= t_end]
        flight = self._fit(pts) if len(pts) >= self.min_fit else None
        if flight is None:
            return  # no verified arc: not evidence of a hit or bounce (rolling, carried, noise)
        kmh = flight.horizontal_kmh
        if len(pts) >= self.speed_min_points and kmh <= self.speed_max_kmh:
            self.last_kmh = kmh  # only well-observed flights drive the speed card
        # Ending low and falling = bounce (ground point -> exact court position); otherwise a hit.
        z_end = flight.position(t_end)[2]
        vz_end = flight.v0[2] - G_FT * (t_end - flight.t0)
        is_bounce = z_end < 1.5 and vz_end < 0
        if is_bounce:
            xy = self.calib.image_to_court([uv_end])[0]
        else:
            xy = flight.position(t_end)[:2]
        events.append(BallEvent("bounce" if is_bounce else "hit", t_end, uv_end, xy, kmh))

    def _reach(self, dt: float) -> float:
        return min(self.reach_base_px + self.reach_px_per_s * max(dt, 0.0), self.reach_max_px)

    def search_hint(self, t: float) -> tuple[np.ndarray, float] | None:
        """(where, radius) the ball should be this frame, for a relaxed second detection pass."""
        if not self._seg:
            return None
        if self._flight is not None:
            return self._predict(t), 3 * self.break_px
        return self._seg[-1][1], self._reach(t - self._seg[-1][0])

    def _reset(self) -> None:
        self._seg, self._pending, self._flight, self._trail = [], [], None, []

    def _lost(self, t: float) -> bool:
        """Main track gone: not seen for too long, or coasting on an arc for longer than the arc was
        observed, or out of the picture."""
        if not self._seg:
            return False
        gap = t - self._seg[-1][0]
        if self._flight is None:
            return gap > self.max_gap_s
        span = self._seg[-1][0] - self._seg[0][0]
        if gap > min(self.max_coast_s, max(self.min_coast_s, span)):
            return True
        if gap > 0 and self.camera is not None:
            u, v = self._project(self._flight, t)
            w, h = self.calib.image_size
            return not (-50 <= u <= w + 50 and -50 <= v <= h + 50)
        return False

    def update(self, t: float, candidates: list[BallCandidate]) -> BallState:
        events: list[BallEvent] = []
        if self._lost(t):
            self._reset()  # ball dead, occluded too long, or its arc no longer trustworthy

        rest = list(candidates)
        chosen = None
        if self._seg and rest:
            pick = self._choose(t, rest)
            if pick is not None:
                cand, off_arc = pick
                rest = [c for c in rest if c is not cand]
                chosen = cand.uv
                self._add_point(t, chosen, off_arc, events)
                self._trail = (self._trail + [chosen])[-10:]

        rest = self._feed_hypotheses(t, rest)
        if not self._seg:
            h = self._best_hypothesis(t, need_arc=False)
            if h is not None:
                chosen = self._adopt(h)
            else:
                ok = [c for c in rest if self._acquirable(c.uv)]
                if ok:
                    cand = max(ok, key=lambda c: c.area)
                    rest = [c for c in rest if c is not cand]
                    chosen = cand.uv
                    self._seg, self._trail = [(t, chosen)], [chosen]
        elif self._flight is None or chosen is None:
            # The main track has no arc, or lost the ball this frame: take a side track that has one.
            h = self._best_hypothesis(t, need_arc=True)
            if h is not None:
                chosen = self._adopt(h)
        for c in rest:
            if len(self._hyps) < self.max_hypotheses and self._acquirable(c.uv):
                self._hyps.append(_Hypothesis([(t, c.uv)]))

        if self._seg and self._flight is None and len(self._seg) >= self.stale_points:
            # Many points and still no ballistic arc: not the ball in play.
            self._hyps.append(_Hypothesis(self._seg[-self.hyp_window:], junk=True))
            self._reset()
            chosen = None

        state = BallState(uv=chosen, trail=list(self._trail), last_kmh=self.last_kmh, events=events)
        if self._flight is not None:
            state.xyz = self._flight.position(t)
            state.flight_kmh = self._flight.horizontal_kmh
            if chosen is None and self.camera is not None:
                state.predicted_uv = self.camera.project(state.xyz)[0]
        return state

    def _choose(self, t: float, candidates: list[BallCandidate]) -> tuple[BallCandidate, bool] | None:
        """The main track's candidate this frame, and whether it is off the current arc."""
        pred = self._predict(t)
        last_t, last_uv = (self._pending or self._seg)[-1]
        if pred is not None:
            d_pred = [np.linalg.norm(c.uv - pred) for c in candidates]
            k = int(np.argmin(d_pred))
            if d_pred[k] <= self.gate_px:
                return candidates[k], self._flight is not None and d_pred[k] > self.break_px
        d_last = [np.linalg.norm(c.uv - last_uv) for c in candidates]
        k = int(np.argmin(d_last))
        if d_last[k] <= self._reach(t - last_t):
            # Not where the arc said: after a hit the ball restarts from where it was.
            return candidates[k], self._flight is not None
        return None

    def _add_point(self, t: float, uv: np.ndarray, off_arc: bool, events: list[BallEvent]) -> None:
        if off_arc:
            self._pending.append((t, uv))
            if len(self._pending) >= 2:
                # Two points in a row off the arc: the ball changed velocity between the last
                # good point and the first pending one.
                self._close_segment(self._seg[-1][0], self._seg[-1][1], events)
                self._seg, self._pending, self._flight = list(self._pending), [], None
        else:
            self._pending = []  # a single off-arc point was an outlier
            self._seg.append((t, uv))
            self._refit(events)

    def _feed_hypotheses(self, t: float, candidates: list[BallCandidate]) -> list[BallCandidate]:
        """Extend the side tracks with candidates the main track didn't take; returns the unused ones."""
        self._hyps = [h for h in self._hyps if t - h.pts[-1][0] <= self.max_gap_s]
        free = list(candidates)
        for h in sorted(self._hyps, key=lambda h: -len(h.pts)):
            if not free:
                break
            last_t, last_uv = h.pts[-1]
            k = None
            if h.flight is not None and self.camera is not None:
                pred = self._project(h.flight, t)
                dists = [np.linalg.norm(c.uv - pred) for c in free]
                if min(dists) <= self.gate_px:
                    k = int(np.argmin(dists))
            if k is None:
                dists = [np.linalg.norm(c.uv - last_uv) for c in free]
                if min(dists) > self._reach(t - last_t):
                    continue
                k = int(np.argmin(dists))
            h.pts = (h.pts + [(t, free.pop(k).uv)])[-self.hyp_window:]
            h.flight = self._fit(h.pts)
            if h.flight is None and len(h.pts) > self.min_fit:
                tail = h.pts[-self.min_fit:]
                h.flight = self._fit(tail)
                if h.flight is not None:
                    h.pts = tail  # the ball changed velocity before the tail
            if h.flight is not None:
                h.junk = False
        return free

    def _best_hypothesis(self, t: float, need_arc: bool) -> _Hypothesis | None:
        live = [h for h in self._hyps if h.pts[-1][0] == t and not h.junk
                and (h.flight is not None or not need_arc)]
        if not live:
            return None
        return max(live, key=lambda h: (h.flight is not None, len(h.pts)))

    def _adopt(self, h: _Hypothesis) -> np.ndarray:
        """Make a side track the main one. A main track without an arc goes back to the side tracks."""
        self._hyps.remove(h)
        if self._seg and self._flight is None:
            self._hyps.append(_Hypothesis(self._seg[-self.hyp_window:], junk=len(self._seg) >= self.stale_points))
        self._seg, self._pending, self._flight = list(h.pts), [], h.flight
        self._trail = [p[1] for p in h.pts][-10:]
        return h.pts[-1][1]

    def _refit(self, events: list[BallEvent]) -> None:
        fit = self._fit(self._seg)
        if fit is not None:
            self._flight = fit
            return
        self._flight = None
        # The segment may span a hit that happened before any arc was established (short flights).
        # If the most recent points alone make a good arc, split there.
        if len(self._seg) > self.min_fit:
            tail = self._seg[-self.min_fit:]
            fit = self._fit(tail)
            if fit is not None:
                head_end = self._seg[-self.min_fit - 1]
                self._close_segment(head_end[0], head_end[1], events)
                self._seg, self._flight = list(tail), fit

    def _acquirable(self, uv: np.ndarray) -> bool:
        u, v = int(round(uv[0])), int(round(uv[1]))
        h, w = self.acquire_mask.shape
        return 0 <= u < w and 0 <= v < h and self.acquire_mask[v, u] > 0


def detect_and_track(detector: BallDetector, tracker: BallTracker, frame: np.ndarray, t: float) -> BallState:
    """One frame: global detection, a relaxed second look where the track expects the ball, update."""
    cands = detector(frame)
    hint = tracker.search_hint(t)
    if hint is not None and hint[0] is not None:
        uv, radius = hint
        if not any(np.linalg.norm(c.uv - uv) <= radius for c in cands):
            cands = cands + detector.search_near(uv, radius)
    return tracker.update(t, cands)


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
