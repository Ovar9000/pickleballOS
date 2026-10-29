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


PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],  # NVIDIA, onnxruntime-gpu
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],  # any DirectX 12 GPU, onnxruntime-directml
}


def ort_session(model_path: str, device: str = "cpu", threads: int | None = None):
    """An ONNX Runtime session on the requested device. Fails loudly instead of silently using CPU."""
    import onnxruntime as ort

    if device not in PROVIDERS:
        raise ValueError(f"device must be one of {sorted(PROVIDERS)}, got {device!r}")
    if device == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # CUDA + cuDNN from the nvidia-* pip wheels, no system CUDA install needed
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    session = ort.InferenceSession(model_path, sess_options=opts, providers=PROVIDERS[device])
    if device != "cpu" and session.get_providers()[0] != PROVIDERS[device][0]:
        raise RuntimeError(f"{device} requested but ONNX Runtime only has {ort.get_available_providers()}")
    return session


def strip_nms(onnx_path: str) -> str:
    """Copy of a detector ONNX that stops right before its NonMaxSuppression node.

    rtmlib's YOLOX files end in a built-in NMS. ONNX Runtime's CUDA NMS kernel was measured at
    ~6 ms per call on an RTX 3060 laptop, more than all the convolutions together. Cutting it out
    and doing NMS on the CPU (microseconds for a few boxes) makes the GPU tier pay off.
    Outputs: boxes (1, N, 4) x1 y1 x2 y2 in model-input pixels, scores (1, classes, N).
    """
    import os

    out = onnx_path[: -len(".onnx")] + ".prenms.onnx"
    if not os.path.exists(out):
        import onnx
        import onnx.utils

        graph = onnx.load(onnx_path).graph
        nms = next(n for n in graph.node if n.op_type == "NonMaxSuppression")
        onnx.utils.extract_model(onnx_path, out, [graph.input[0].name], [nms.input[0], nms.input[1]])
    return out


class RtmlibDetector:
    """YOLOX person detector + RTMPose (rtmlib, ONNX Runtime, Apache-2.0).

    Models download on first use to ~/.cache/rtmlib. `lightweight` = YOLOX-tiny + RTMPose-s,
    `balanced` = YOLOX-m + RTMPose-m. `device` is cpu | cuda | dml; `threads` caps CPU threads
    (used to emulate smaller machines).
    """

    def __init__(self, mode: str = "lightweight", device: str = "cpu", score_thr: float = 0.3,
                 threads: int | None = None, nms_thr: float = 0.65):
        # Defaults match what rtmlib's YOLOX files do with their built-in NMS (IoU 0.65, score 0.3).
        from rtmlib import YOLOX, RTMPose
        from rtmlib.tools.solution.body import Body

        cfg = Body.MODE[mode]
        self.name = f"rtmlib-{mode}-{device}" + (f"-{threads}t" if threads else "")
        self.score_thr, self.nms_thr = score_thr, nms_thr
        self._det = YOLOX(cfg["det"], model_input_size=cfg["det_input_size"], score_thr=score_thr,
                          backend="onnxruntime", device="cpu")
        self._pose = RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"],
                             backend="onnxruntime", device="cpu")
        # rtmlib builds plain CPU sessions; rebuild them with our device and thread settings, and
        # run the detector without its built-in NMS (see strip_nms).
        self._det.session = ort_session(strip_nms(self._det.onnx_model), device, threads)
        self._pose.session = ort_session(self._pose.onnx_model, device, threads)

    def warmup(self, frame: np.ndarray, n: int = 3) -> None:
        """GPU providers tune kernels on the first runs; keep that out of timings."""
        for _ in range(n):
            self.pose(frame, self.detect(frame)[:4])

    def detect(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        img, ratio = self._det.preprocess(frame)
        boxes, scores = self._det.inference(img)
        person = scores[0, 0]  # COCO class 0
        keep = person > self.score_thr
        b, s = boxes[0][keep] / ratio, person[keep]
        if len(b) == 0:
            return np.zeros((0, 4))
        xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b]
        idx = np.asarray(cv2.dnn.NMSBoxes(xywh, s.astype(float).tolist(), self.score_thr, self.nms_thr)).reshape(-1)
        return b[idx].astype(np.float64)

    def pose(self, frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

        # One batched model call for all players (rtmlib runs one call per box), and the same
        # normalisation as rtmlib in float32 (its float64 version cost ~1.5 ms per player).
        mean = np.asarray(self._pose.mean, dtype=np.float32)
        inv_std = 1.0 / np.asarray(self._pose.std, dtype=np.float32)
        crops, meta = [], []
        for b in boxes:
            center, scale = bbox_xyxy2cs(np.asarray(b), padding=1.25)
            img, scale = top_down_affine(self._pose.model_input_size, scale, center, frame)
            crops.append(((img.astype(np.float32) - mean) * inv_std).transpose(2, 0, 1))
            meta.append((center, scale))
        session = self._pose.session
        outputs = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(crops, dtype=np.float32)})
        kps, scores = [], []
        for i, (center, scale) in enumerate(meta):
            k, s = self._pose.postprocess([o[i:i + 1] for o in outputs], center, scale)
            kps.append(k)
            scores.append(s)
        return np.concatenate(kps).astype(np.float64), np.concatenate(scores).astype(np.float64)
