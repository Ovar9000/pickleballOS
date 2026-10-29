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
