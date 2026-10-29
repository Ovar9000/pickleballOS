"""Multi-player tracking in court space (feet), plus distance covered.

Tracking happens on the court plane rather than in pixels: four players in a 20 x 44 ft box with a
known maximum speed is an easy association problem there, and court distance is the same near and
far from the camera. Each track is a constant-velocity Kalman filter; detections are assigned with
the Hungarian algorithm inside a distance gate. Confirmed tracks hold one of four session slots
(P1-P4). When a track is lost and a new one appears, it takes the free slot whose last known
position is nearest, so a short occlusion doesn't create a "new player". No appearance features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court

SLOTS = ("P1", "P2", "P3", "P4")
_H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


@dataclass
class Measurement:
    xy: np.ndarray  # court feet
    sigma_ft: float  # measurement noise; grows with distance from the camera
    box: np.ndarray | None = None
    cue: str = "box"
    keypoints: np.ndarray | None = None  # (17, 2) pixels, COCO order
    kp_scores: np.ndarray | None = None


@dataclass
class Track:
    id: int
    x: np.ndarray  # [x, y, vx, vy]
    P: np.ndarray
    t: float
    last_update: float
    hits: int = 1
    confirmed: bool = False
    slot: str | None = None
    box: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    kp_scores: np.ndarray | None = None

    @property
    def xy(self) -> np.ndarray:
        return self.x[:2]


@dataclass
class _Slot:
    track_id: int | None = None
    last_xy: np.ndarray | None = None


class CourtTracker:
    def __init__(self, max_players: int = 4, gate_ft: float = 5.0, confirm_hits: int = 3,
                 max_age_s: float = 1.5, tentative_age_s: float = 0.5, accel_sigma: float = 8.0,
                 reacquire_ft: float = 8.0, spawn_side_margin: float = 1.5, spawn_end_margin: float = 10.0):
        self.max_players = max_players
        # New tracks start only near the lines (servers stand behind the baseline, so the end
        # margin is generous). Referees and ball crew beside the court never become players, but a
        # tracked player can still chase a wide ball outside this zone.
        self.spawn_side_margin = spawn_side_margin
        self.spawn_end_margin = spawn_end_margin
        self.gate_ft = gate_ft
        self.confirm_hits = confirm_hits
        self.max_age_s = max_age_s
        self.tentative_age_s = tentative_age_s
        self.accel_sigma = accel_sigma
        self.reacquire_ft = reacquire_ft
        self.tracks: list[Track] = []
        self.slots = {s: _Slot() for s in SLOTS[:max_players]}
        self._next_id = 1

    # --- Kalman --------------------------------------------------------------------------

    def _predict(self, tr: Track, t: float) -> None:
        dt = t - tr.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        tr.x = F @ tr.x
        tr.P = F @ tr.P @ F.T + G @ G.T * self.accel_sigma**2
        tr.t = t

    @staticmethod
    def _update(tr: Track, m: Measurement) -> None:
        R = np.eye(2) * m.sigma_ft**2
        S = _H @ tr.P @ _H.T + R
        K = tr.P @ _H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (m.xy - _H @ tr.x)
        tr.P = (np.eye(4) - K @ _H) @ tr.P

    # --- Step ----------------------------------------------------------------------------

    def step(self, t: float, measurements: list[Measurement] | None) -> None:
        """Advance to time t. Pass None on frames where the detector didn't run."""
        for tr in self.tracks:
            self._predict(tr, t)
        if measurements is not None:
            unmatched = self._associate(t, measurements)
            for m in unmatched:
                if court.in_court(*m.xy, side_margin=self.spawn_side_margin, end_margin=self.spawn_end_margin):
                    self._spawn(t, m)
        self._confirm_and_assign()
        self._expire(t)

    def _associate(self, t: float, ms: list[Measurement]) -> list[Measurement]:
        remaining = list(range(len(ms)))
        # Confirmed tracks pick first, so a tentative track (a passer-by, a ghost) can't steal a player.
        for group in ([tr for tr in self.tracks if tr.confirmed], [tr for tr in self.tracks if not tr.confirmed]):
            if not group or not remaining:
                continue
            cost = np.array([[np.linalg.norm(tr.xy - ms[j].xy) for j in remaining] for tr in group])
            rows, cols = linear_sum_assignment(np.where(cost > self.gate_ft, 1e6, cost))
            used = set()
            for r, c in zip(rows, cols):
                if cost[r, c] > self.gate_ft:
                    continue
                tr, m = group[r], ms[remaining[c]]
                self._update(tr, m)
                tr.last_update, tr.hits, tr.box = t, tr.hits + 1, m.box
                tr.keypoints, tr.kp_scores = m.keypoints, m.kp_scores
                used.add(remaining[c])
            remaining = [j for j in remaining if j not in used]
        return [ms[j] for j in remaining]

    def _spawn(self, t: float, m: Measurement) -> None:
        P = np.diag([m.sigma_ft**2, m.sigma_ft**2, 25.0, 25.0])  # unknown velocity, up to ~5 ft/s
        self.tracks.append(Track(self._next_id, np.array([*m.xy, 0.0, 0.0]), P, t, t, box=m.box))
        self._next_id += 1

    def _confirm_and_assign(self) -> None:
        """Give newly confirmed tracks a free slot, jointly (Hungarian), not first-come-first-served.

        Cost tiers, cheapest first: a released slot on the same side of the net whose last position
        is close (<= reacquire_ft); a released same-side slot further away; a never-used slot; a
        slot from the other end. Players only change ends between games, and which side of the net
        someone is on stays reliable even where the calibration is poor. If more tracks are ready
        than slots are free, the rest stay tentative: four players are already on court.
        """
        ready = [tr for tr in self.tracks if not tr.confirmed and tr.hits >= self.confirm_hits]
        free = [s for s, info in self.slots.items() if info.track_id is None]
        if not ready or not free:
            return
        # Near side first, then left to right, so a fresh session gets readable P1-P4 labels.
        ready.sort(key=lambda tr: (tr.x[1], tr.x[0]))
        cost = np.zeros((len(ready), len(free)))
        for i, tr in enumerate(ready):
            x, y = tr.xy
            # Ties go to whoever is inside the lines rather than standing well outside them.
            outside = max(0.0, -x, x - court.WIDTH_FT) + max(0.0, -y, y - court.LENGTH_FT)
            for j, s in enumerate(free):
                last = self.slots[s].last_xy
                if last is None:
                    cost[i, j] = 300.0 + (i - j) ** 2 + 10.0 * outside
                    continue
                d = float(np.linalg.norm(last - tr.xy))
                if (last[1] < court.NET_Y) != (tr.x[1] < court.NET_Y):
                    cost[i, j] = 1000.0 + d
                else:
                    cost[i, j] = d if d <= self.reacquire_ft else 200.0 + d
        # Doubles has two players per side. A third person confirming on one side is a bystander or a
        # duplicate, so it stays tentative. (Existing tracks may still cross, e.g. at a change of ends.)
        per_side = self.max_players // 2
        on_side = {True: 0, False: 0}
        for tr in self.tracks:
            if tr.confirmed:
                on_side[bool(tr.x[1] < court.NET_Y)] += 1
        rows, cols = linear_sum_assignment(cost)
        for r, c in sorted(zip(rows, cols), key=lambda rc: cost[rc]):
            tr, slot = ready[r], free[c]
            side = bool(tr.x[1] < court.NET_Y)
            if on_side[side] >= per_side:
                continue
            on_side[side] += 1
            tr.confirmed, tr.slot = True, slot
            self.slots[slot].track_id = tr.id

    def _expire(self, t: float) -> None:
        keep = []
        for tr in self.tracks:
            age = t - tr.last_update
            if tr.confirmed and age > self.max_age_s:
                self.slots[tr.slot].track_id = None
                self.slots[tr.slot].last_xy = tr.xy.copy()
            elif not tr.confirmed and age > self.tentative_age_s:
                pass
            else:
                keep.append(tr)
        self.tracks = keep

    def players(self) -> dict[str, Track]:
        """Confirmed tracks by slot."""
        out = {}
        for tr in self.tracks:
            if tr.confirmed:
                self.slots[tr.slot].last_xy = tr.xy.copy()
                out[tr.slot] = tr
        return out


@dataclass
class DistanceMeter:
    """Distance covered from a smoothed position stream, with a deadband against jitter.

    Positions are first smoothed with an exponential moving average (`alpha` per sample), then
    distance is added only once the player has moved `deadband_ft` from the last anchor. Without
    both, a player standing still at the kitchen line accumulates feet of fake distance from
    detection noise. The cost is a small undercount of very slow shuffling; the evaluation measures
    both effects.
    """

    deadband_ft: float = 0.3
    alpha: float = 0.3
    max_speed_fps: float = 25.0  # faster than any player sprints: an identity or geometry glitch
    total_ft: float = 0.0
    rejected_jumps: int = 0
    _smooth: np.ndarray | None = field(default=None, repr=False)
    _anchor: np.ndarray | None = field(default=None, repr=False)
    _last: tuple[float, np.ndarray] | None = field(default=None, repr=False)

    def restart(self) -> None:
        """Forget the anchor (e.g. the slot now follows a different track) without losing the total."""
        self._smooth = self._anchor = self._last = None

    def add(self, xy: np.ndarray, t: float | None = None) -> float:
        xy = np.asarray(xy, dtype=np.float64)
        if t is not None and self._last is not None:
            dt = t - self._last[0]
            if dt > 0 and np.linalg.norm(xy - self._last[1]) / dt > self.max_speed_fps:
                self.rejected_jumps += 1
                self.restart()
        if t is not None:
            self._last = (t, xy.copy())
        if self._anchor is None:
            self._smooth, self._anchor = xy.copy(), xy.copy()
            return self.total_ft
        self._smooth = self.alpha * xy + (1 - self.alpha) * self._smooth
        xy = self._smooth
        step = float(np.linalg.norm(xy - self._anchor))
        if step >= self.deadband_ft:
            self.total_ft += step
            self._anchor = xy.copy()
        return self.total_ft
