import json

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from pbos.ball import BallCandidate, LearnedBallDetector
from pbos.balldata import BallLabel, load_labels, save_labels, triplet_frames
from pbos.balleval import evaluate


def brightness_model(path):
    """Stand-in 'network': heatmap = mean of frame t's three channels (inputs 6..8), so a bright dot
    in the newest frame is the peak. Exercises buffering, ROI offsets and peak finding."""
    frames = helper.make_tensor_value_info("frames", TensorProto.FLOAT, [1, 9, "h", "w"])  # legacy float input
    heat = helper.make_tensor_value_info("heatmap", TensorProto.FLOAT, [1, 1, "h", "w"])
    inits = [helper.make_tensor(n, TensorProto.INT64, [1], [v]) for n, v in
             (("s", 6), ("e", 9), ("ax", 1))]
    nodes = [helper.make_node("Slice", ["frames", "s", "e", "ax"], ["last"]),
             helper.make_node("ReduceMean", ["last"], ["heatmap"], axes=[1], keepdims=1)]
    graph = helper.make_graph(nodes, "brightness", [frames], [heat], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # the onnx package may write a newer IR than ONNX Runtime reads
    onnx.save(model, str(path))
    return path


def frame_with_dot(x, y, size=(640, 360)):
    f = np.zeros((size[1], size[0], 3), np.uint8)
    cv2.circle(f, (x, y), 2, (255, 255, 255), -1)
    return f


def test_learned_detector_buffers_three_frames_and_maps_roi_coordinates(tmp_path):
    det = LearnedBallDetector(brightness_model(tmp_path / "m.onnx"), threshold=0.5)
    det.set_roi((100, 50, 603, 350))  # width/height get rounded down to multiples of 8
    assert det.roi == (100, 50, 596, 346)
    assert det(frame_with_dot(300, 200)) == []  # needs t-2, t-1, t
    assert det(frame_with_dot(310, 200)) == []
    cands = det(frame_with_dot(320, 200))
    assert len(cands) == 1
    np.testing.assert_allclose(cands[0].uv, (320, 200), atol=1)


def test_learned_detector_relaxed_search_uses_the_last_heatmap(tmp_path):
    det = LearnedBallDetector(brightness_model(tmp_path / "m.onnx"), threshold=0.9, search_threshold=0.2)
    dim = np.zeros((360, 640, 3), np.uint8)
    cv2.circle(dim, (200, 100), 2, (110, 110, 110), -1)  # heatmap ~0.43: below 0.9, above 0.2
    for _ in range(3):
        cands = det(dim)
    assert cands == []
    near = det.search_near(np.array([205.0, 98.0]), 20)
    assert len(near) == 1 and np.allclose(near[0].uv, (200, 100), atol=1)
    assert det.search_near(np.array([500.0, 300.0]), 20) == []


def test_labels_round_trip_and_triplets(tmp_path):
    labels = [BallLabel(10, 1.0, 2.0), BallLabel(11, 3.0, 4.0, False, "human")]
    save_labels(tmp_path / "l.json", "v.mp4", 29.97, labels, {"ranges": "0-1"})
    meta, back = load_labels(tmp_path / "l.json")
    assert back == labels and meta["fps"] == 29.97
    assert triplet_frames(labels) == {8, 9, 10, 11}


def test_evaluate_scores_hits_misses_and_false_alarms(tmp_path):
    video = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for _ in range(10):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    answers = iter([(20.0, 20.0), (40.0, 10.0), None, (5.0, 5.0)])

    class Fake:
        def __call__(self, img):
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls % 3:
                return []
            a = next(answers)
            return [] if a is None else [BallCandidate(np.array(a), 10.0)]

    labels = [BallLabel(3, 21.0, 20.0), BallLabel(4, 10.0, 10.0), BallLabel(5, 30.0, 30.0),
              BallLabel(6, -1, -1, False, "human")]
    fake = Fake()
    r = evaluate(video, labels, lambda: fake, tol_px=5.0)
    # hit; wrong place (fp + fn); missed (fn); alarm on an empty frame (fp)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 2)


def test_focal_loss_prefers_a_peak_at_the_ball():
    torch = pytest.importorskip("torch")
    from pbos.ballnet import BallNet, focal_loss, gaussian_heatmap

    target = torch.from_numpy(gaussian_heatmap(32, 32, 10, 12, 2.0))[None, None]
    good = torch.full((1, 1, 32, 32), -6.0)
    good[0, 0, 12, 10] = 6.0
    bad = torch.full((1, 1, 32, 32), -6.0)
    bad[0, 0, 25, 25] = 6.0
    assert focal_loss(good, target) < focal_loss(bad, target)
    out = BallNet()(torch.zeros(1, 9, 64, 64))
    assert out.shape == (1, 1, 64, 64)
