import cv2
import numpy as np
import pytest

from pbos import court
from pbos.ball import G_FT, BallCandidate, BallDetector, BallTracker, Flight, fit_ballistic
from pbos.calibration import Calibration
from pbos.camera3d import Camera3D, net_top_points

W, H = 1280, 720
FPS = 30.0


def true_camera(height=12.0, behind=14.0, f=900.0) -> Camera3D:
    """A pinhole camera behind the near baseline, looking down the court."""
    C = np.array([court.CENTER_X, -behind, height])
    target = np.array([court.CENTER_X, 26.0, 0.0])
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # world -> camera
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return Camera3D(K, R, -R @ C)


def calib_from(cam: Camera3D) -> Calibration:
    pts = {n: tuple(cam.project([[x, y, 0.0]])[0]) for n, (x, y) in court.KEYPOINTS.items()}
    return Calibration.from_points(pts, (W, H))


def test_camera_recovered_from_homography():
    truth = true_camera()
    cam = Camera3D.from_calibration(calib_from(truth))
    assert abs(cam.focal_px - 900) < 5
    np.testing.assert_allclose(cam.center, truth.center, atol=0.2)
    np.testing.assert_allclose(cam.project(net_top_points()), truth.project(net_top_points()), atol=1.0)


def simulate(cam, p0, v0, seconds, t0=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ts = t0 + np.arange(int(seconds * FPS)) / FPS
    f = Flight(t0, np.array(p0, float), np.array(v0, float), 0.0)
    uv = cam.project(np.array([f.position(t) for t in ts])) + rng.normal(0, noise, (len(ts), 2))
    return ts, uv


def test_ballistic_fit_recovers_3d_flight_from_one_view():
    cam = true_camera()
    p0, v0 = (12.0, 6.0, 3.0), (-2.0, 35.0, 9.0)  # a drive from the near baseline, ~24 mph
    ts, uv = simulate(cam, p0, v0, 0.8, noise=0.7)
    fit = fit_ballistic(ts, uv, cam.P)
    assert fit.rms_px < 1.5
    np.testing.assert_allclose(fit.p0, p0, atol=1.0)
    assert abs(fit.horizontal_mph - np.hypot(*v0[:2]) * 3600 / 5280) < 2.0


def test_tracker_splits_flights_and_places_the_bounce():
    truth = true_camera()
    calib = calib_from(truth)
    cam = Camera3D.from_calibration(calib)
    # Serve-like ball from near side, bouncing deep in the far court, then rising again.
    p0, v0 = np.array([5.0, -1.0, 3.0]), np.array([3.0, 40.0, 10.0])
    a = Flight(0.0, p0, v0, 0.0)
    t_land = (v0[2] + np.sqrt(v0[2] ** 2 + 2 * G_FT * p0[2])) / G_FT
    land = a.position(t_land)
    vb = np.array([v0[0], v0[1], -(v0[2] - G_FT * t_land) * 0.6]) * [0.8, 0.8, 1]
    b = Flight(t_land, np.array([land[0], land[1], 0.0]), vb, 0.0)
    tracker = BallTracker(calib, cam, acquire_mask=np.full((H, W), 255, np.uint8))
    events = []
    for k in range(int((t_land + 0.4) * FPS)):
        t = k / FPS
        X = a.position(t) if t < t_land else b.position(t)
        st = tracker.update(t, [BallCandidate(truth.project(X)[0], 20.0)])
        events += st.events
    bounces = [e for e in events if e.kind == "bounce"]
    assert len(bounces) == 1
    assert abs(bounces[0].t - t_land) < 2.5 / FPS
    # Bounce position from the last pre-bounce frame: within the distance travelled in a frame.
    assert np.linalg.norm(bounces[0].court_xy - land[:2]) < 2.0
    assert abs(bounces[0].speed_mph - np.hypot(*v0[:2]) * 3600 / 5280) < 2.0


def test_track_survives_a_hit_and_a_short_occlusion():
    # A dink from the far side to the near side, a 5-frame occlusion mid-flight, then a volley
    # sends it back. The tracker must keep one continuous track: coast through the gap on the arc
    # and pick the ball up again after the hit, where it leaves the predicted arc.
    truth = true_camera()
    calib = calib_from(truth)
    cam = Camera3D.from_calibration(calib)
    a = Flight(0.0, np.array([10.0, 30.0, 2.5]), np.array([0.5, -14.0, 12.0]), 0.0)
    t_hit = 0.7
    hit_at = a.position(t_hit)
    b = Flight(t_hit, hit_at, np.array([-1.0, 18.0, 6.0]), 0.0)
    tracker = BallTracker(calib, cam, acquire_mask=np.full((H, W), 255, np.uint8))
    seen = coasting = 0
    events = []
    for k in range(int(1.3 * FPS)):
        t = k / FPS
        X = a.position(t) if t < t_hit else b.position(t)
        occluded = 10 <= k < 15
        st = tracker.update(t, [] if occluded else [BallCandidate(truth.project(X)[0], 20.0)])
        events += st.events
        seen += st.uv is not None
        coasting += occluded and st.predicted_uv is not None
    assert coasting == 5  # predicted through the whole occlusion
    assert seen == int(1.3 * FPS) - 5  # every visible frame was linked, including after the hit
    hits = [e for e in events if e.kind == "hit"]
    assert len(hits) == 1 and abs(hits[0].t - t_hit) < 2.5 / FPS


def test_relaxed_search_recovers_a_faint_ball_near_the_prediction():
    det = BallDetector()
    bg = np.full((H, W, 3), (120, 90, 60), np.uint8)
    faint = (95, 118, 125)  # blurred ball: washed-out, darker yellow than the global thresholds accept
    for k in range(3):
        f = bg.copy()
        cv2.circle(f, (400 + 30 * k, 300), 3, faint, -1)
        found = det(f)
    assert found == []  # too faint for the global thresholds
    near = det.search_near(np.array([462.0, 302.0]), 20)
    assert len(near) == 1
    np.testing.assert_allclose(near[0].uv, (460, 300), atol=1.5)


def test_detector_finds_a_moving_yellow_ball_not_a_static_one():
    det = BallDetector()
    bg = np.full((H, W, 3), (120, 90, 60), np.uint8)
    yellow = (60, 230, 230)  # BGR optic yellow
    static = bg.copy()
    cv2.circle(static, (100, 100), 4, yellow, -1)  # a yellow logo that never moves
    found = []
    for k in range(4):
        f = static.copy()
        cv2.circle(f, (400 + 30 * k, 300), 4, yellow, -1)
        found = det(f)
    assert len(found) == 1
    np.testing.assert_allclose(found[0].uv, (490, 300), atol=1.0)
