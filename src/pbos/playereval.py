"""Player position ground truth and evaluation.

A human clicks, on random frames, the point on the floor midway between each player's feet, and
says whether that point is inside the painted court lines (lines count as in). Labels are image
pixels, so they stay valid if the calibration is redone; they are converted to court feet with
whatever calibration is being evaluated.

Two different things are measured:
- **Position error** (feet): the tracker's map position against the clicked point mapped through the
  same calibration. This is the ground-point estimate (ankles, box bottom) plus the tracker; a wrong
  calibration moves both alike, so it does not show here.
- **In/out agreement**: the human's in/out call comes from the painted lines, not from the
  calibration, so it does catch calibration errors. It is reported twice: the clicked point mapped
  through the calibration (calibration alone) and the tracker's map position (what the map shows).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import court


@dataclass
class FootPoint:
    u: float
    v: float
    inside: bool | None  # None when the feet are hidden (the click is a best guess)
    hidden: bool = False


@dataclass
class PlayerFrame:
    frame: int
    points: list[FootPoint] = field(default_factory=list)


def save_player_labels(path: str | Path, video: str, fps: float, frames: list[PlayerFrame]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"video": Path(video).name, "fps": fps, "frames": [asdict(f) for f in frames]},
                               indent=1))


def load_player_labels(path: str | Path) -> tuple[dict, list[PlayerFrame]]:
    d = json.loads(Path(path).read_text())
    return d, [PlayerFrame(f["frame"], [FootPoint(**p) for p in f["points"]]) for f in d["frames"]]


def boundary_distance_ft(xy) -> float:
    """Signed distance to the court outline: positive inside, negative outside."""
    x, y = float(xy[0]), float(xy[1])
    inside = min(x, court.WIDTH_FT - x, y, court.LENGTH_FT - y)
    if inside >= 0:
        return inside
    dx = max(-x, 0.0, x - court.WIDTH_FT)
    dy = max(-y, 0.0, y - court.LENGTH_FT)
    return -float(np.hypot(dx, dy))


def match(truth: list[np.ndarray], pred: list[np.ndarray], max_ft: float) -> list[tuple[int, int]]:
    """Pairs (truth index, pred index) minimising total distance, each pair within max_ft. At most four
    players a side, so trying every assignment is cheap."""
    best, best_cost = [], (0, 0.0)
    small, big, swap = (truth, pred, False) if len(truth) <= len(pred) else (pred, truth, True)
    for perm in itertools.permutations(range(len(big)), len(small)):
        pairs = [(i, j) for i, j in enumerate(perm) if np.linalg.norm(small[i] - big[j]) <= max_ft]
        cost = (-len(pairs), sum(np.linalg.norm(small[i] - big[j]) for i, j in pairs))
        if cost < best_cost or not best:
            best, best_cost = pairs, cost
    return [(j, i) for i, j in best] if swap else best


class PlayerScorer:
    """Collects map positions at labelled frames (use `hook` as run_track's frame_hook), then scores."""

    def __init__(self, labels: list[PlayerFrame], max_match_ft: float = 6.0):
        self.by_frame = {f.frame: f for f in labels}
        self.max_match_ft = max_match_ft
        self.rows: list[dict] = []

    def hook(self, frame: int, t: float, players: dict, calib) -> None:
        lab = self.by_frame.get(frame)
        if lab is None or calib is None:
            return
        truth = [calib.image_to_court([[p.u, p.v]])[0] for p in lab.points]
        slots = sorted(players)
        pred = [players[s].xy.copy() for s in slots]
        pairs = dict(match(truth, pred, self.max_match_ft))
        for i, p in enumerate(lab.points):
            j = pairs.get(i)
            self.rows.append({"frame": frame, "t": round(t, 2), "label_xy": truth[i].round(2).tolist(),
                              "hidden": p.hidden, "inside": p.inside,
                              "slot": slots[j] if j is not None else None,
                              "map_xy": pred[j].round(2).tolist() if j is not None else None})
        for j, s in enumerate(slots):
            if j not in pairs.values():
                self.rows.append({"frame": frame, "t": round(t, 2), "label_xy": None, "slot": s,
                                  "map_xy": pred[j].round(2).tolist()})

    def summary(self) -> dict:
        players = [r for r in self.rows if r["label_xy"] is not None]
        visible = [r for r in players if not r["hidden"]]
        matched = [r for r in visible if r["map_xy"] is not None]
        err = lambda rs: [float(np.linalg.norm(np.subtract(r["map_xy"], r["label_xy"]))) for r in rs]  # noqa: E731

        def stats(rs):
            e = err(rs)
            return {"n": len(e), "median_ft": round(float(np.median(e)), 2) if e else None,
                    "p90_ft": round(float(np.percentile(e, 90)), 2) if e else None}

        near = [r for r in matched if r["label_xy"][1] < court.NET_Y]
        far = [r for r in matched if r["label_xy"][1] >= court.NET_Y]
        called = [r for r in visible if r["inside"] is not None]
        calib_ok = [r for r in called if (boundary_distance_ft(r["label_xy"]) >= 0) == r["inside"]]
        map_called = [r for r in called if r["map_xy"] is not None]
        map_ok = [r for r in map_called if (boundary_distance_ft(r["map_xy"]) >= 0) == r["inside"]]
        wrong = [{"t": r["t"], "slot": r["slot"], "human": "in" if r["inside"] else "out",
                  "label_xy": r["label_xy"], "map_xy": r["map_xy"],
                  "calibration": "in" if boundary_distance_ft(r["label_xy"]) >= 0 else "out",
                  "map": None if r["map_xy"] is None else ("in" if boundary_distance_ft(r["map_xy"]) >= 0 else "out")}
                 for r in called if r not in calib_ok or (r["map_xy"] is not None and r not in map_ok)]
        return {
            "frames": len({r["frame"] for r in self.rows}),
            "players_labelled": len(players), "feet_visible": len(visible),
            "found": len(matched), "recall": round(len(matched) / max(len(visible), 1), 3),
            "extra_tracks": sum(r["label_xy"] is None for r in self.rows),
            "position_error": {"all": stats(matched), "near_half": stats(near), "far_half": stats(far)},
            "in_out": {"calls": len(called),
                       "calibration_agrees": round(len(calib_ok) / max(len(called), 1), 3),
                       "map_agrees": round(len(map_ok) / max(len(map_called), 1), 3),
                       "disagreements": wrong},
            "max_match_ft": self.max_match_ft,
        }
