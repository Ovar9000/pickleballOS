"""A full 3D camera (K, R, t) recovered from the court homography.

The homography only describes the ground plane. To reason about a ball in the air we need the
camera's rays, i.e. its focal length and pose. Assuming square pixels, no skew and the principal
point at the image centre, one homography pins down the focal length (Zhang's plane-based
calibration): the first two columns of K^-1 H must be orthogonal and of equal length. Everything
here is in court feet, with z = height above the court (up).

Lens distortion is ignored, so treat 3D quantities as estimates. `net_top_overlay` is the sanity
check: the 34-36 in net tape, projected with this camera, should sit on the real net tape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import court
from .calibration import Calibration

NET_HEIGHT_POST_FT = 36 / 12
NET_HEIGHT_CENTER_FT = 34 / 12


class CameraModelError(ValueError):
    pass


@dataclass
class Camera3D:
    K: np.ndarray
    R: np.ndarray  # columns: court x, y, up axes in camera coordinates (may be a reflection)
    t: np.ndarray

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def center(self) -> np.ndarray:
        """Camera position in court feet (x, y, height)."""
        return -np.linalg.solve(self.R, self.t)

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    def project(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        h = np.hstack([X, np.ones((len(X), 1))]) @ self.P.T
        return h[:, :2] / h[:, 2:3]

    @classmethod
    def from_calibration(cls, calib: Calibration) -> Camera3D:
        w, h = calib.image_size
        cx, cy = w / 2.0, h / 2.0
        Hc = np.linalg.inv(calib.H)  # court ft -> image px
        Hc = Hc / np.linalg.norm(Hc[:, 0])
        # Shift the principal point to the origin so K^-1 only divides x, y by f.
        T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
        G = T @ Hc
        h1, h2 = G[:, 0], G[:, 1]
        # Two equations of the form a / f^2 + b = 0: orthogonality and equal norms.
        a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
        b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
        inv_f2 = -float(a @ b) / float(a @ a)
        if not np.isfinite(inv_f2) or inv_f2 <= 0:
            raise CameraModelError("focal length is not recoverable from this view (too frontal or degenerate)")
        f = 1.0 / np.sqrt(inv_f2)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

        M = np.linalg.inv(K) @ Hc
        lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
        # Pick the sign that puts the court in front of the camera.
        centre = np.array([court.CENTER_X, court.NET_Y, 1.0])
        if (M @ centre)[2] < 0:
            lam = -lam
        r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
        R = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt  # nearest rotation
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        mirror = np.eye(3)
        if cls(K, R, t).center[2] < 0:
            # Court axes were labelled as a mirror image (left/right swapped): flip "up" so height
            # is positive. P stays a valid projection for (x, y, height) points.
            mirror = np.diag([1, 1, -1.0])
        return cls(K, R @ mirror, t)._refined(calib, mirror)

    def _refined(self, calib: Calibration, mirror: np.ndarray) -> Camera3D:
        """Polish focal length and pose against the clicked keypoints (forcing R to be a true
        rotation moves ground points a few pixels away from the homography; this pulls them back)."""
        import cv2
        from scipy.optimize import least_squares

        names = list(calib.image_points)
        X = np.array([[*court.KEYPOINTS[n], 0.0] for n in names])
        uv = np.array([calib.image_points[n] for n in names])
        cx, cy = self.K[0, 2], self.K[1, 2]
        R0 = self.R @ mirror  # proper rotation
        rvec0, _ = cv2.Rodrigues(R0)

        def build(p):
            Rr, _ = cv2.Rodrigues(p[:3])
            K = np.array([[p[6], 0, cx], [0, p[6], cy], [0, 0, 1.0]])
            return Camera3D(K, Rr @ mirror, p[3:6])

        def residual(p):
            return (build(p).project(X) - uv).ravel()

        p0 = np.concatenate([rvec0.ravel(), self.t, [self.focal_px]])
        sol = least_squares(residual, p0, method="lm" if len(uv) * 2 >= 7 else "trf")
        return build(sol.x) if sol.success else self

    def reprojection_px(self, calib: Calibration) -> float:
        names = list(calib.image_points)
        X = np.array([[*court.KEYPOINTS[n], 0.0] for n in names])
        uv = np.array([calib.image_points[n] for n in names])
        return float(np.sqrt(np.mean(np.sum((self.project(X) - uv) ** 2, axis=1))))


def net_top_points(n: int = 21) -> np.ndarray:
    """3D points along the net tape: 36 in at the sidelines, sagging to 34 in at the centre."""
    x = np.linspace(0.0, court.WIDTH_FT, n)
    sag = 1 - np.abs(x - court.CENTER_X) / court.CENTER_X
    z = NET_HEIGHT_POST_FT - (NET_HEIGHT_POST_FT - NET_HEIGHT_CENTER_FT) * sag
    return np.stack([x, np.full(n, court.NET_Y), z], axis=1)
