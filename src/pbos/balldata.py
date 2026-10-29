"""Training data for the learned ball detector.

Labels come from two places:
- `auto`: the classical detector + tracker, keeping only frames where the detection sits within a
  few pixels of a fitted ballistic arc. A blob that agrees with physics over many frames is very
  unlikely to be a false positive, so these are precise labels, but biased towards easy frames
  (the hard ones are exactly where the classical detector fails). Good for training.
- `human`: frames reviewed in `pbos label`, including frames where the ball is not visible.
  Unbiased, so these are the only labels used to *evaluate* detectors.

Frames are cached as JPEGs (only the ones training needs), local only and git-ignored, since the
sample videos are third-party footage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .ball import BallDetector, BallTracker, detect_and_track, detection_roi
from .calibration import Calibration
from .camera_motion import CameraMotionMonitor
from .pipeline import camera_or_none


@dataclass
class BallLabel:
    frame: int  # absolute frame index in the video
    u: float
    v: float
    visible: bool = True
    source: str = "auto"  # auto | human


def save_labels(path: str | Path, video: str, fps: float, labels: list[BallLabel], meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"video": Path(video).name, "fps": fps, **(meta or {}),
                                "labels": [asdict(lb) for lb in labels]}, indent=1))


def load_labels(path: str | Path) -> tuple[dict, list[BallLabel]]:
    d = json.loads(Path(path).read_text())
    return d, [BallLabel(**lb) for lb in d["labels"]]


def autolabel(video: str, calib: Calibration, ranges: list[tuple[float, float]], max_resid_px: float = 3.0,
              progress: bool = True) -> tuple[float, list[BallLabel]]:
    """Confident ball positions from the classical tracker over the given time ranges (seconds)."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    labels: list[BallLabel] = []
    for start, end in ranges:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
        f0 = int(round(start * fps))
        n = int((end - start) * fps)
        det, monitor = BallDetector(), None
        det.set_roi(detection_roi(calib))
        camera = camera_or_none(calib)
        tracker = BallTracker(calib, camera)
        status, applied_shift = "steady", 0.0
        it = range(n)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, unit="f", desc=f"autolabel {start:.0f}-{end:.0f}s")
        for i in it:
            ok, frame = cap.read()
            if not ok:
                break
            t = (f0 + i) / fps
            if monitor is None:
                monitor = CameraMotionMonitor(frame, calib)
            elif monitor.scene_changed(frame) or i % int(fps) == 0:
                st = monitor.check(frame)
                status = st.status
                if status == "lost":
                    det.reset()
                else:
                    shift = st.shift_px if status == "rebased" else 0.0
                    if abs(shift - applied_shift) > 1.5:
                        camera = camera_or_none(st.calib)
                        tracker.set_geometry(st.calib, camera)
                        det.set_roi(detection_roi(st.calib))
                        applied_shift = shift
            if status == "lost":
                continue
            bs = detect_and_track(det, tracker, frame, t)
            if bs.uv is None or bs.xyz is None or camera is None:
                continue
            if np.linalg.norm(camera.project(bs.xyz)[0] - bs.uv) <= max_resid_px:
                labels.append(BallLabel(f0 + i, float(bs.uv[0]), float(bs.uv[1])))
    cap.release()
    return fps, labels


def cache_frames(video: str, frames: set[int], out_dir: str | Path, quality: int = 92) -> Path:
    """Decode the video once, sequentially, and write only the requested frames as JPEGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = sorted(f for f in frames if not (out_dir / f"{f:06d}.jpg").exists())
    if not todo:
        return out_dir
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, todo[0])
    idx, want = todo[0], set(todo)
    last = todo[-1]
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        idx += 1
    cap.release()
    return out_dir


def triplet_frames(labels: list[BallLabel]) -> set[int]:
    """Every frame a label needs: itself and the two before it (the model sees t-2, t-1, t)."""
    return {f - k for lb in labels for f in (lb.frame,) for k in (0, 1, 2) if f - k >= 0}
