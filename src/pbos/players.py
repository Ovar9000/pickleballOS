"""Player detection + pose behind a small interface, so detectors can be swapped and benchmarked.

The pipeline only needs two things from a backend: person boxes for a frame, and COCO-17 keypoints
for a chosen subset of those boxes. Pose runs only on boxes that pass court gating, so bystanders
on other courts cost a detector pass but no pose pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_WRIST, RIGHT_WRIST = 9, 10


@dataclass
class PersonDetection:
    box: np.ndarray  # (4,) x1, y1, x2, y2 in pixels
    keypoints: np.ndarray | None = None  # (17, 2) COCO order
    kp_scores: np.ndarray | None = None  # (17,)

    def box_bottom(self) -> np.ndarray:
        x1, _, x2, y2 = self.box
        return np.array([(x1 + x2) / 2.0, y2])

    def ground_point(self, kp_thr: float = 0.3) -> tuple[np.ndarray, str]:
        """Image point where the player stands, and which cue produced it.

        Ankle midpoint when both ankles are confident, else the confident ankle, else the box
        bottom. Ankles sit a few inches above the floor, so this has a small bias away from the
        camera; whether ankles beat the box bottom is measured in the evaluation, not assumed.
        """
        if self.keypoints is not None and self.kp_scores is not None:
            ok = [i for i in (LEFT_ANKLE, RIGHT_ANKLE) if self.kp_scores[i] >= kp_thr]
            if len(ok) == 2:
                return self.keypoints[ok].mean(axis=0), "ankles"
            if len(ok) == 1:
                return self.keypoints[ok[0]].copy(), "ankle"
        return self.box_bottom(), "box"


class PlayerDetector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Person boxes, (N, 4) x1, y1, x2, y2."""

    def pose(self, frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (N, 17, 2) and scores (N, 17) for the given boxes."""


class RtmlibDetector:
    """YOLOX person detector + RTMPose (rtmlib, ONNX Runtime, Apache-2.0).

    Models download on first use to ~/.cache/rtmlib. `lightweight` = YOLOX-tiny + RTMPose-s,
    `balanced` = YOLOX-m + RTMPose-m.
    """

    def __init__(self, mode: str = "lightweight", device: str = "cpu", score_thr: float = 0.5):
        from rtmlib import YOLOX, RTMPose
        from rtmlib.tools.solution.body import Body

        cfg = Body.MODE[mode]
        self.name = f"rtmlib-{mode}"
        self._det = YOLOX(cfg["det"], model_input_size=cfg["det_input_size"], score_thr=score_thr,
                          backend="onnxruntime", device=device)
        self._pose = RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"],
                             backend="onnxruntime", device=device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        boxes = self._det(frame)
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    def pose(self, frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        kps, scores = self._pose(frame, bboxes=list(boxes))
        return np.asarray(kps, dtype=np.float64), np.asarray(scores, dtype=np.float64)
