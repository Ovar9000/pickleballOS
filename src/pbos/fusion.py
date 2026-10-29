"""Fuse audio pops, ball-track direction changes and player wrists into contacts, then rallies.

Each cue fails differently:
- the ball track's direction changes are where the ball is, but can't tell a low dink hit from a
  bounce and also fire on a rolling or carried dead ball;
- audio pops pin the moment to milliseconds, but neighbouring courts pop too;
- wrists (from pose) say whether a paddle could be there.

Rules (all thresholds are parameters, to be tuned against labelled rallies):
- direction change near a wrist, with a pop                   -> hit (credited to that player);
  without audio in the recording, a vision "hit" near a wrist is accepted instead
- direction change away from wrists, low and falling           -> bounce (court position exact)
- pop with no direction change, the ball at a wrist, and the ball in flight (on a fitted arc)
  just before -> hit (contact hidden from the camera). "In flight" rules out a ball held in a hand
  between points while a neighbouring court pops.
- anything else -> dropped (dead ball, other court, noise)

A rally is a run of hits with gaps under `gap_s`; it closes once no hit has followed for `gap_s`.
Everything here is causal with a short settle delay, so it also works live.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .audio import Onset
from .ball import BallEvent


@dataclass
class PlayerSnapshot:
    box: np.ndarray  # x1, y1, x2, y2
    wrists: np.ndarray  # (k, 2) confident wrist pixels, k in 0..2
    court_xy: np.ndarray


@dataclass
class Contact:
    t: float
    kind: str  # "hit" | "bounce"
    source: str  # "audio+vision" | "vision" | "audio"
    slot: str | None = None
    court_xy: np.ndarray | None = None
    uv: np.ndarray | None = None
    speed_mph: float | None = None


@dataclass
class Rally:
    start: float
    end: float
    hits: int
    bounces: int


@dataclass
class _Frame:
    t: float
    ball_uv: np.ndarray | None
    players: dict[str, PlayerSnapshot]
    in_flight: bool = False


class ContactFusion:
    def __init__(self, window_s: float = 0.12, settle_s: float = 0.3, reach: float = 0.5,
                 min_hit_gap_s: float = 0.15, history_s: float = 1.0, require_audio: bool = True,
                 flight_lookback_s: float = 0.35):
        self.require_audio = require_audio  # set False when the recording has no usable audio
        self.flight_lookback_s = flight_lookback_s
        self.window_s = window_s
        self.settle_s = settle_s
        self.reach = reach  # paddle reach from the wrist, as a fraction of the player's box height
        self.min_hit_gap_s = min_hit_gap_s
        self._frames: deque[_Frame] = deque()
        self._history_s = history_s
        self._onsets: list[Onset] = []
        self._vision: list[BallEvent] = []
        self._last_hit_t = -1e9

    def add_frame(self, t: float, ball_uv: np.ndarray | None, players: dict[str, PlayerSnapshot],
                  in_flight: bool = False) -> None:
        self._frames.append(_Frame(t, ball_uv, players, in_flight))
        while self._frames and self._frames[0].t < t - self._history_s:
            self._frames.popleft()

    def add_onsets(self, onsets: list[Onset]) -> None:
        self._onsets.extend(onsets)

    def add_vision(self, events: list[BallEvent]) -> None:
        self._vision.extend(events)

    def _frame_at(self, t: float) -> _Frame | None:
        if not self._frames:
            return None
        return min(self._frames, key=lambda f: abs(f.t - t))

    def _nearest_wrist(self, uv: np.ndarray, players: dict[str, PlayerSnapshot]) -> tuple[str | None, float]:
        """Owner of the wrist nearest to uv, and that distance in units of the owner's box height."""
        best, best_d = None, np.inf
        for slot, p in players.items():
            if len(p.wrists) == 0:
                continue
            h = max(p.box[3] - p.box[1], 1.0)
            d = float(np.min(np.linalg.norm(p.wrists - uv, axis=1))) / h
            if d < best_d:
                best, best_d = slot, d
        return best, best_d

    def resolve(self, t: float) -> list[Contact]:
        """Contacts whose evidence is complete (older than the settle delay), in time order."""
        horizon = t - self.settle_s
        out: list[Contact] = []
        ready = [e for e in self._vision if e.t <= horizon]
        self._vision = [e for e in self._vision if e.t > horizon]
        used: set[int] = set()

        for e in ready:
            k, onset = self._match_onset(e.t, used)
            frame = self._frame_at(e.t)
            players = frame.players if frame else {}
            slot, d = self._nearest_wrist(e.uv, players)
            near = d <= self.reach
            when = onset.t if onset else e.t
            if near and (onset is not None or (e.kind == "hit" and not self.require_audio)):
                out.append(Contact(when, "hit", "audio+vision" if onset else "vision", slot,
                                   players[slot].court_xy if slot else None, e.uv, e.speed_mph))
            elif not near and e.kind == "bounce":
                out.append(Contact(when, "bounce", "audio+vision" if onset else "vision", None, e.court_xy, e.uv,
                                   e.speed_mph))
            else:
                if k is not None:
                    used.discard(k)  # the pop may still explain a hidden contact below
                continue

        # Pops with no direction change: a hit the camera didn't see, if the ball was at a paddle.
        for k, o in enumerate(self._onsets):
            if o.t > horizon or k in used:
                continue
            frame = self._frame_at(o.t)
            if frame is None or frame.ball_uv is None:
                continue
            slot, d = self._nearest_wrist(frame.ball_uv, frame.players)
            flying = any(f.in_flight for f in self._frames if o.t - self.flight_lookback_s <= f.t <= o.t)
            if d <= self.reach and flying:
                out.append(Contact(o.t, "hit", "audio", slot, frame.players[slot].court_xy, frame.ball_uv))
        self._onsets = [o for o in self._onsets if o.t > horizon]

        out.sort(key=lambda c: c.t)
        kept = []
        for c in out:  # one hit per stroke: a pop and a direction change can both describe it
            if c.kind == "hit":
                if c.t - self._last_hit_t < self.min_hit_gap_s:
                    continue
                self._last_hit_t = c.t
            kept.append(c)
        return kept

    def _match_onset(self, t: float, used: set[int]) -> tuple[int | None, Onset | None]:
        best, best_dt = None, self.window_s
        for k, o in enumerate(self._onsets):
            if k in used:
                continue
            dt = abs(o.t - t)
            if dt <= best_dt:
                best, best_dt = k, dt
        if best is None:
            return None, None
        used.add(best)
        return best, self._onsets[best]


@dataclass
class RallyTracker:
    gap_s: float = 3.0
    current: Rally | None = None
    rallies: list[Rally] = field(default_factory=list)

    def on_contact(self, c: Contact) -> Rally | None:
        """Feed contacts in time order. Returns a rally if this contact closed the previous one."""
        closed = self.step(c.t)
        if c.kind == "hit":
            if self.current is None:
                self.current = Rally(c.t, c.t, 0, 0)
            self.current.hits += 1
            self.current.end = c.t
        elif self.current is not None:
            self.current.bounces += 1
            self.current.end = c.t
        return closed

    def step(self, t: float) -> Rally | None:
        if self.current is not None and t - self.current.end > self.gap_s:
            done, self.current = self.current, None
            self.rallies.append(done)
            return done
        return None
