"""Court calibration: image <-> court-plane homography, residuals, masks, verification views.

The camera is fixed, so H is computed once (from >= 4 clicked court keypoints) and every
ground-plane point after that (feet, bounces) maps to court feet with one 3x3 multiply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from . import court

MIN_POINTS = 4


class CalibrationError(ValueError):
    pass


def _apply(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply homography H to (N, 2) points. Returns (projected (N, 2), homogeneous w (N,))."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = hom[:, :2] / w[:, None]
    return out, w


def _has_general_quad(pts: np.ndarray) -> bool:
    """True if some 4 of the points have no 3 collinear (exact court coordinates, so exact test)."""
    def collinear(a, b, c) -> bool:
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-9

    for quad in combinations(pts, 4):
        if not any(collinear(*tri) for tri in combinations(quad, 3)):
            return True
    return False


def fit_homography(image_points: dict[str, tuple[float, float]]) -> np.ndarray:
    """Least-squares homography mapping image pixels -> court feet from named keypoints."""
    unknown = sorted(set(image_points) - set(court.KEYPOINTS))
    if unknown:
        raise CalibrationError(f"unknown keypoint names: {unknown}")
    if len(image_points) < MIN_POINTS:
        raise CalibrationError(f"need at least {MIN_POINTS} keypoints, got {len(image_points)}")

    names = list(image_points)
    src = np.array([image_points[n] for n in names], dtype=np.float64)
    dst = np.array([court.KEYPOINTS[n] for n in names], dtype=np.float64)

    # Homography needs 4 points with no 3 collinear on the court plane; check that up front
    # because OpenCV may return a degenerate matrix instead of failing.
    if not _has_general_quad(dst):
        raise CalibrationError("keypoints are (nearly) collinear; include points from at least two "
                               "court lines with no three of the four on one line")

    H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise CalibrationError("could not fit a homography; check that clicks match their keypoint names")
    return H / H[2, 2]


@dataclass
class Calibration:
    H: np.ndarray  # image px -> court ft
    image_size: tuple[int, int]  # (width, height)
    image_points: dict[str, tuple[float, float]]
    court_id: str = "court1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @classmethod
    def from_points(cls, image_points: dict[str, tuple[float, float]], image_size: tuple[int, int],
                    court_id: str = "court1") -> Calibration:
        pts = {k: (float(u), float(v)) for k, (u, v) in image_points.items()}
        return cls(H=fit_homography(pts), image_size=tuple(image_size), image_points=pts, court_id=court_id)

    @property
    def H_inv(self) -> np.ndarray:
        Hi = np.linalg.inv(self.H)
        return Hi / Hi[2, 2]

    def _ground_sign(self) -> float:
        # Sign of w for image pixels that see the ground in front of the camera; pixels above
        # the horizon get the opposite sign and must not be treated as court points.
        _, w = _apply(self.H_inv, np.array([[court.CENTER_X, court.NET_Y]]))
        return float(np.sign(w[0]))

    def rebased(self, G: np.ndarray) -> Calibration:
        """Calibration for the same court after the image moved by homography G (old px -> new px)."""
        H = self.H @ np.linalg.inv(G)
        moved, _ = _apply(G, list(self.image_points.values()))
        pts = {k: (float(u), float(v)) for k, (u, v) in zip(self.image_points, moved)}
        return Calibration(H=H / H[2, 2], image_size=self.image_size, image_points=pts,
                           court_id=self.court_id, created_at=self.created_at)

    def ft_per_px(self, uv) -> float:
        """Local scale at an image point: how many court feet one pixel spans there."""
        u, v = float(uv[0]), float(uv[1])
        p = self.image_to_court([[u, v], [u + 1, v], [u, v + 1]])
        return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2)

    def image_to_court(self, pts) -> np.ndarray:
        out, _ = _apply(self.H, pts)
        return out

    def court_to_image(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """Project court points to pixels. Returns (pixels (N, 2), visible mask (N,)).

        Points behind the camera project to meaningless pixels and are marked not visible.
        """
        out, w = _apply(self.H_inv, pts)
        return out, (w * self._ground_sign()) > 1e-9

    def residuals_ft(self) -> dict[str, float]:
        """Distance (ft) between each clicked keypoint mapped through H and its true position.

        With exactly 4 points the fit is exact and residuals are ~0; use >= 6 for a real check.
        """
        names = list(self.image_points)
        mapped = self.image_to_court([self.image_points[n] for n in names])
        truth = np.array([court.KEYPOINTS[n] for n in names])
        return dict(zip(names, np.linalg.norm(mapped - truth, axis=1).tolist()))

    def error_summary(self) -> tuple[float, float]:
        r = np.array(list(self.residuals_ft().values()))
        return float(np.sqrt(np.mean(r**2))), float(r.max())

    def leave_one_out_ft(self) -> dict[str, float]:
        """Error (ft) predicting each keypoint from a fit that excludes it.

        Unlike residuals, the tested point did not shape the fit, so this is an honest accuracy
        estimate that needs no tape measure: the painted lines are the ruler. Points whose removal
        leaves fewer than 4 usable points are skipped, so 4 clicks give an empty result.
        """
        out = {}
        for name, uv in self.image_points.items():
            rest = {k: v for k, v in self.image_points.items() if k != name}
            try:
                H = fit_homography(rest)
            except CalibrationError:
                continue
            pred, _ = _apply(H, [uv])
            out[name] = float(np.linalg.norm(pred[0] - np.array(court.KEYPOINTS[name])))
        return out

    def loo_summary(self) -> tuple[float, float] | None:
        r = np.array(list(self.leave_one_out_ft().values()))
        if r.size == 0:
            return None
        return float(np.sqrt(np.mean(r**2))), float(r.max())

    def court_mask(self, side_margin: float = 4.0, end_margin: float = 8.0) -> np.ndarray:
        """uint8 mask (255 = pixel sees the court plus margins). Used to ignore people off-court."""
        w_img, h_img = self.image_size
        uu, vv = np.meshgrid(np.arange(w_img, dtype=np.float64), np.arange(h_img, dtype=np.float64))
        xy, w = _apply(self.H, np.stack([uu.ravel(), vv.ravel()], axis=1))
        # Pixels that see the ground have the same w sign as the projected court centre.
        _, w_center = _apply(self.H, self.court_to_image([[court.CENTER_X, court.NET_Y]])[0])
        on_ground = np.sign(w) == np.sign(w_center[0])
        inside = (
            on_ground
            & (xy[:, 0] >= -side_margin) & (xy[:, 0] <= court.WIDTH_FT + side_margin)
            & (xy[:, 1] >= -end_margin) & (xy[:, 1] <= court.LENGTH_FT + end_margin)
        )
        return (inside.reshape(h_img, w_img) * 255).astype(np.uint8)

    def to_dict(self) -> dict:
        rms, worst = self.error_summary()
        loo = self.loo_summary()
        return {
            "court_id": self.court_id,
            "created_at": self.created_at,
            "image_size": list(self.image_size),
            "units": "feet",
            "origin": "near-left baseline corner (camera side); x across, y along, net at y=22",
            "H_image_to_court": self.H.tolist(),
            "image_points": {k: list(v) for k, v in self.image_points.items()},
            "rms_err_ft": rms,
            "max_err_ft": worst,
            "loo_rms_ft": loo[0] if loo else None,
            "loo_max_ft": loo[1] if loo else None,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        d = json.loads(Path(path).read_text())
        return cls(
            H=np.array(d["H_image_to_court"], dtype=np.float64),
            image_size=tuple(d["image_size"]),
            image_points={k: tuple(v) for k, v in d["image_points"].items()},
            court_id=d.get("court_id", "court1"),
            created_at=d.get("created_at", ""),
        )


# --- Verification views -------------------------------------------------------------------

LINE_COLOR = (80, 255, 120)
NET_COLOR = (255, 200, 80)
KITCHEN_COLOR = (0, 200, 255)


def _draw_court_segment(img, calib: Calibration, seg, color, thickness, samples: int = 40) -> None:
    (x0, y0), (x1, y1) = seg
    t = np.linspace(0.0, 1.0, samples)
    pts, ok = calib.court_to_image(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1))
    # Draw only consecutive visible runs, so lines that pass behind the camera don't wrap around.
    run: list[np.ndarray] = []
    for p, good in zip(pts, ok):
        if good and np.all(np.abs(p) < 1e5):
            run.append(p)
        elif run:
            cv2.polylines(img, [np.round(np.array(run)).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
            run = []
    if len(run) > 1:
        cv2.polylines(img, [np.round(np.array(run)).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_court_overlay(frame: np.ndarray, calib: Calibration) -> np.ndarray:
    """Court lines re-projected onto the camera frame, plus clicked keypoints and their errors."""
    out = frame.copy()
    tint = out.copy()
    kitchen = [(0, court.NEAR_KITCHEN_Y), (court.WIDTH_FT, court.NEAR_KITCHEN_Y),
               (court.WIDTH_FT, court.FAR_KITCHEN_Y), (0, court.FAR_KITCHEN_Y)]
    k_px, k_ok = calib.court_to_image(kitchen)
    if k_ok.all():
        cv2.fillPoly(tint, [np.round(k_px).astype(np.int32)], KITCHEN_COLOR)
        out = cv2.addWeighted(tint, 0.18, out, 0.82, 0)

    for seg in court.LINES:
        _draw_court_segment(out, calib, seg, LINE_COLOR, 2)
    _draw_court_segment(out, calib, court.NET, NET_COLOR, 2)

    residuals = calib.residuals_ft()
    for name, (u, v) in calib.image_points.items():
        p = (int(round(u)), int(round(v)))
        cv2.circle(out, p, 6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{residuals[name]:.2f}ft", (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_topdown(frame: np.ndarray, calib: Calibration, px_per_ft: float = 12.0,
                   side_margin: float = 6.0, end_margin: float = 8.0) -> np.ndarray:
    """Warp the camera frame onto the court plane (far baseline at the top) with ideal lines drawn on top."""
    W = int(round((court.WIDTH_FT + 2 * side_margin) * px_per_ft))
    Hh = int(round((court.LENGTH_FT + 2 * end_margin) * px_per_ft))
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(Hh, dtype=np.float64))
    cx = cols / px_per_ft - side_margin
    cy = (Hh - 1 - rows) / px_per_ft - end_margin
    src, ok = calib.court_to_image(np.stack([cx.ravel(), cy.ravel()], axis=1))
    src[~ok] = -1.0  # remap treats out-of-range sources as border
    map_x = src[:, 0].reshape(Hh, W).astype(np.float32)
    map_y = src[:, 1].reshape(Hh, W).astype(np.float32)
    top = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))

    def to_px(x, y):
        return int(round((x + side_margin) * px_per_ft)), int(round(Hh - 1 - (y + end_margin) * px_per_ft))

    for (a, b) in court.LINES:
        cv2.line(top, to_px(*a), to_px(*b), LINE_COLOR, 1, cv2.LINE_AA)
    cv2.line(top, to_px(*court.NET[0]), to_px(*court.NET[1]), NET_COLOR, 1, cv2.LINE_AA)
    return top


def verification_image(frame: np.ndarray, calib: Calibration) -> np.ndarray:
    """Side-by-side: camera frame with re-projected lines | top-down warp. Plus an error caption."""
    overlay = draw_court_overlay(frame, calib)
    top = render_topdown(frame, calib)
    scale = overlay.shape[0] / top.shape[0]
    top = cv2.resize(top, (int(round(top.shape[1] * scale)), overlay.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = np.hstack([overlay, top])
    rms, worst = calib.error_summary()
    n = len(calib.image_points)
    note = f"{n} keypoints | fit rms {rms:.2f} ft | max {worst:.2f} ft"
    loo = calib.loo_summary()
    if loo:
        note += f" | leave-one-out rms {loo[0]:.2f} ft, max {loo[1]:.2f} ft"
    if n == MIN_POINTS:
        note += " | exactly 4 points: errors are 0 by construction, add more to check"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(canvas, note, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def read_frame(video: str | Path, at_seconds: float = 0.0) -> np.ndarray:
    """Grab one BGR frame from a video file (or read an image file)."""
    video = str(video)
    if Path(video).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(video)
        if img is None:
            raise FileNotFoundError(video)
        return img
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"could not read a frame at {at_seconds}s from {video}")
    return frame
