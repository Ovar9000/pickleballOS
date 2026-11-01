import numpy as np

from pbos import court
from pbos.calibration import Calibration
from pbos.playereval import FootPoint, PlayerFrame, PlayerScorer, boundary_distance_ft, match


class _Track:
    def __init__(self, xy):
        self.xy = np.array(xy, float)


def _calib():
    # 10 px per foot, near baseline at the bottom of the image.
    pts = {n: (100 + 10 * x, 600 - 10 * y) for n, (x, y) in court.KEYPOINTS.items()}
    return Calibration.from_points(pts, (1280, 720))


def test_boundary_distance_sign():
    assert boundary_distance_ft((10, 5)) == 5
    assert boundary_distance_ft((10, 0)) == 0  # on the line counts as in
    assert np.isclose(boundary_distance_ft((10, -1.5)), -1.5)
    assert np.isclose(boundary_distance_ft((23, -4)), -5)


def test_match_pairs_nearest_and_drops_far_ones():
    truth = [np.array([2.0, 2.0]), np.array([15.0, 30.0])]
    pred = [np.array([15.5, 29.0]), np.array([2.5, 2.0]), np.array([10.0, 60.0])]
    assert sorted(match(truth, pred, 6.0)) == [(0, 1), (1, 0)]
    assert match(truth, [np.array([40.0, 40.0])], 6.0) == []


def test_scorer_reports_error_and_in_out_calls():
    calib = _calib()
    labels = [PlayerFrame(10, [FootPoint(100 + 10 * 5, 600 - 10 * 3, True),   # (5, 3) in
                               FootPoint(100 + 10 * 15, 600 + 10 * 1, False),  # (15, -1) out, behind the baseline
                               FootPoint(100 + 10 * 8, 600 - 10 * 40, None, hidden=True)])]
    sc = PlayerScorer(labels)
    # Map: first player 0.5 ft off, second drawn just inside the baseline (wrong call), one extra track.
    sc.hook(10, 0.33, {"P1": _Track((5.5, 3.0)), "P2": _Track((15.0, 0.3)), "P3": _Track((8.0, 40.0)),
                       "P4": _Track((1.0, 20.0))}, calib)
    sc.hook(11, 0.36, {}, calib)  # unlabelled frame: ignored
    s = sc.summary()
    assert s["feet_visible"] == 2 and s["found"] == 2 and s["extra_tracks"] == 1
    assert np.isclose(s["position_error"]["all"]["median_ft"], 0.9, atol=0.01)
    assert s["in_out"]["calibration_agrees"] == 1.0
    assert s["in_out"]["map_agrees"] == 0.5
    assert s["in_out"]["disagreements"][0]["map"] == "in"
