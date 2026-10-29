"""Detect when the "fixed" camera moves, and re-align the calibration when it can.

Every check matches ORB features between the calibration-time reference frame and the current
frame and fits a frame-to-frame homography G (reference px -> current px) with RANSAC. If the
court keypoints moved more than a few pixels, the calibration is re-based: H_new = H_ref . G^-1.
That is exact for a camera that rotated in place (re-aimed) and approximate if it was also
translated. Too few inliers (a blank frame, a cut, a fully different view) -> status "lost" and
the previous calibration is kept, so the caller can flag positions as unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Calibration


@dataclass
class CameraStatus:
    status: str  # "steady" | "rebased" | "lost"
    shift_px: float
    inliers: int
    calib: Calibration


class CameraMotionMonitor:
    def __init__(self, ref_frame: np.ndarray, calib: Calibration, scale: float = 0.5,
                 n_features: int = 1500, min_inliers: int = 40, tol_px: float = 3.0):
        self.calib_ref = calib
        self.scale = scale
        self.min_inliers = min_inliers
        self.tol_px = tol_px
        self._orb = cv2.ORB_create(n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ref_kp, self._ref_desc = self._features(ref_frame)
        self._probe = np.array(list(calib.image_points.values()), dtype=np.float64)
        self.current = CameraStatus("steady", 0.0, len(self._ref_kp), calib)
        self._thumb = self._thumbnail(ref_frame)

    @staticmethod
    def _thumbnail(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)

    def scene_changed(self, frame: np.ndarray, threshold: float = 12.0) -> bool:
        """Cheap per-frame test (a 64x36 thumbnail diff) for cuts, fades and pans.

        Players moving barely change a thumbnail; a cut or a camera move changes most of it. Use it
        to run the full ORB check immediately instead of waiting for the next periodic check.
        """
        thumb = self._thumbnail(frame)
        changed = float(np.abs(thumb - self._thumb).mean()) > threshold
        self._thumb = thumb
        return changed

    def _features(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return self._orb.detectAndCompute(small, None)

    def check(self, frame: np.ndarray) -> CameraStatus:
        kp, desc = self._features(frame)
        if desc is None or self._ref_desc is None or len(kp) < self.min_inliers:
            self.current = CameraStatus("lost", float("nan"), 0, self.current.calib)
            return self.current
        matches = self._bf.match(self._ref_desc, desc)
        if len(matches) < self.min_inliers:
            self.current = CameraStatus("lost", float("nan"), len(matches), self.current.calib)
            return self.current
        a = np.float32([self._ref_kp[m.queryIdx].pt for m in matches])
        b = np.float32([kp[m.trainIdx].pt for m in matches])
        G_small, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if G_small is None or inliers < self.min_inliers:
            self.current = CameraStatus("lost", float("nan"), inliers, self.current.calib)
            return self.current

        S = np.diag([self.scale, self.scale, 1.0])
        G = np.linalg.inv(S) @ G_small @ S  # full-resolution reference px -> current px
        moved = cv2.perspectiveTransform(self._probe.reshape(-1, 1, 2), G).reshape(-1, 2)
        shift = float(np.linalg.norm(moved - self._probe, axis=1).max())
        if shift <= self.tol_px:
            self.current = CameraStatus("steady", shift, inliers, self.calib_ref)
        else:
            self.current = CameraStatus("rebased", shift, inliers, self.calib_ref.rebased(G))
        return self.current
