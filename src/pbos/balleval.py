"""Frame-level ball detector evaluation against labels.

For each labelled frame the detector sees frames t-2, t-1, t (fresh state, no tracker) and its
strongest candidate is compared with the label:
- visible ball, candidate within `tol_px`  -> true positive
- visible ball, candidate elsewhere        -> false positive + false negative
- visible ball, no candidate               -> false negative
- no ball visible, any candidate           -> false positive
Use human labels for headline numbers: auto labels come from the classical detector, so scoring
the classical detector against them is circular.
"""

from __future__ import annotations

import cv2
import numpy as np

from .balldata import BallLabel


def evaluate(video: str, labels: list[BallLabel], make_detector, tol_px: float = 5.0) -> dict:
    cap = cv2.VideoCapture(video)
    tp = fp = fn = 0
    errors = []
    for lb in labels:
        det = make_detector()
        cands = []
        for f in (lb.frame - 2, lb.frame - 1, lb.frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            cands = det(img)
        best = max(cands, key=lambda c: c.area).uv if cands else None
        if lb.visible:
            if best is not None and np.linalg.norm(best - (lb.u, lb.v)) <= tol_px:
                tp += 1
                errors.append(float(np.linalg.norm(best - (lb.u, lb.v))))
            else:
                fn += 1
                fp += best is not None
        else:
            fp += best is not None
    cap.release()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"frames": len(labels), "visible": sum(lb.visible for lb in labels), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
            "median_error_px": round(float(np.median(errors)), 2) if errors else None, "tol_px": tol_px}


def evaluate_tracking(video: str, labels: list[BallLabel], calib, make_detector, tol_px: float = 8.0,
                      lead_s: float = 1.0) -> dict:
    """Score the whole ball pipeline (detector + tracker + camera-move check) against labels.

    Unlike `evaluate`, frames are played in order from `lead_s` before the first label, so the tracker
    has its usual temporal context. Its output at a labelled frame is the detection it kept, or the
    arc's prediction while coasting. Scoring is as in `evaluate`; `tol_px` is a little looser because
    a coasting prediction is not a detection. `off_court` counts output points whose ground
    projection is on another court (a track that wandered off).
    """
    from .ball import BallTracker, detect_and_track, detection_roi
    from .camera_motion import CameraMotionMonitor
    from .pipeline import camera_or_none

    by_frame = {lb.frame: lb for lb in labels}
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f0 = max(min(by_frame) - int(round(lead_s * fps)), 0)
    f1 = max(by_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    det = make_detector()
    tracker = BallTracker(calib, camera_or_none(calib))
    monitor, status, shift_done = None, "steady", 0.0
    tp = fp = fn = coast_tp = 0
    errors = []
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        t = f / fps
        if monitor is None:
            monitor = CameraMotionMonitor(img, calib)
        elif monitor.scene_changed(img) or (f - f0) % int(round(fps)) == 0:
            st = monitor.check(img)
            status = st.status
            if status == "lost":
                det.reset()
            else:
                shift = st.shift_px if status == "rebased" else 0.0
                if abs(shift - shift_done) > 1.5:
                    tracker.set_geometry(st.calib, camera_or_none(st.calib))
                    det.set_roi(detection_roi(st.calib))
                    shift_done = shift
        bs = detect_and_track(det, tracker, img, t) if status != "lost" else tracker.update(t, [])
        lb = by_frame.get(f)
        if lb is None:
            continue
        out = bs.uv if bs.uv is not None else bs.predicted_uv
        if lb.visible:
            if out is not None and np.linalg.norm(out - (lb.u, lb.v)) <= tol_px:
                tp += 1
                coast_tp += bs.uv is None
                errors.append(float(np.linalg.norm(out - (lb.u, lb.v))))
            else:
                fn += 1
                fp += out is not None
        else:
            fp += out is not None
    cap.release()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"frames": len(labels), "visible": sum(lb.visible for lb in labels), "tp": tp, "fp": fp, "fn": fn,
            "tp_coasting": coast_tp, "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
            "median_error_px": round(float(np.median(errors)), 2) if errors else None, "tol_px": tol_px}
