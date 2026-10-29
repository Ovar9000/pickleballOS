import json

import cv2
import numpy as np
import pytest

from pbos import court
from pbos.calibration import Calibration, CalibrationError, fit_homography, render_topdown, verification_image
from pbos.cli import main

IMAGE_SIZE = (1280, 720)

# A plausible elevated behind-baseline view: near baseline wide at the bottom, far baseline narrow.
CORNER_PIXELS = {
    "near_left_corner": (200.0, 680.0),
    "near_right_corner": (1080.0, 680.0),
    "far_left_corner": (500.0, 240.0),
    "far_right_corner": (780.0, 240.0),
}


def true_court_to_image() -> np.ndarray:
    src = np.array([court.KEYPOINTS[n] for n in CORNER_PIXELS], dtype=np.float32)
    dst = np.array(list(CORNER_PIXELS.values()), dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def synthetic_points(noise_px: float = 0.0, names=None, seed: int = 0) -> dict[str, tuple[float, float]]:
    H = true_court_to_image()
    names = names or list(court.KEYPOINTS)
    pts = np.array([court.KEYPOINTS[n] for n in names], dtype=np.float64)
    img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    img += np.random.default_rng(seed).normal(0.0, noise_px, img.shape)
    return {n: tuple(p) for n, p in zip(names, img)}


def test_exact_points_recover_court_coordinates():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    rms, worst = calib.error_summary()
    assert worst < 1e-4


def test_noisy_clicks_give_small_error():
    calib = Calibration.from_points(synthetic_points(noise_px=1.0), IMAGE_SIZE)
    rms, worst = calib.error_summary()
    assert rms < 0.15
    assert worst < 0.4


def test_kitchen_points_only_is_enough():
    # Net-line cameras often lose both baselines; the six kitchen-line points must suffice.
    names = [n for n in court.KEYPOINTS if "kitchen" in n]
    calib = Calibration.from_points(synthetic_points(names=names), IMAGE_SIZE)
    far_corner_px = synthetic_points(names=["far_right_corner"])["far_right_corner"]
    np.testing.assert_allclose(calib.image_to_court([far_corner_px])[0], (20.0, 44.0), atol=1e-4)


def test_leave_one_out_is_empty_with_four_points():
    calib = Calibration.from_points(synthetic_points(names=list(CORNER_PIXELS)), IMAGE_SIZE)
    assert calib.leave_one_out_ft() == {}
    assert calib.loo_summary() is None


def test_leave_one_out_flags_a_bad_click():
    pts = synthetic_points(noise_px=0.5)
    u, v = pts["near_center_kitchen"]
    pts["near_center_kitchen"] = (u + 30.0, v)  # a sloppy click
    calib = Calibration.from_points(pts, IMAGE_SIZE)
    loo = calib.leave_one_out_ft()
    assert max(loo, key=loo.get) == "near_center_kitchen"
    # The fit absorbs part of the bad click, so its residual understates what held-out testing shows.
    assert loo["near_center_kitchen"] > calib.residuals_ft()["near_center_kitchen"]
    assert loo["near_center_kitchen"] > 1.0


def test_round_trip_image_court_image():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    pts = np.array([[3.0, 5.0], [10.0, 22.0], [17.5, 40.0]])
    img, visible = calib.court_to_image(pts)
    assert visible.all()
    np.testing.assert_allclose(calib.image_to_court(img), pts, atol=1e-6)


def test_too_few_points_rejected():
    with pytest.raises(CalibrationError):
        fit_homography(synthetic_points(names=["near_left_corner", "near_right_corner", "far_left_corner"]))


def test_collinear_points_rejected():
    # Four points, three of them on the near baseline: the homography is not determined.
    names = ["near_left_corner", "near_center_baseline", "near_right_corner", "far_left_corner"]
    with pytest.raises(CalibrationError):
        fit_homography(synthetic_points(names=names))


def test_unknown_keypoint_rejected():
    pts = synthetic_points()
    pts["mid_air"] = (1.0, 2.0)
    with pytest.raises(CalibrationError):
        fit_homography(pts)


def test_save_load_round_trip(tmp_path):
    calib = Calibration.from_points(synthetic_points(noise_px=0.5), IMAGE_SIZE, court_id="c7")
    path = tmp_path / "c7.json"
    calib.save(path)
    loaded = Calibration.load(path)
    np.testing.assert_allclose(loaded.H, calib.H)
    assert loaded.court_id == "c7"
    assert loaded.image_points.keys() == calib.image_points.keys()
    assert json.loads(path.read_text())["units"] == "feet"


def test_court_mask_includes_court_and_excludes_sky():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    mask = calib.court_mask()
    px, _ = calib.court_to_image([[10.0, 22.0]])
    u, v = px[0]
    assert mask[int(v), int(u)] == 255
    assert mask[0, 0] == 0  # top-left pixel is above the horizon / far off court


def test_topdown_warp_puts_court_centre_in_the_middle():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    # Paint the ground patch 8..12 ft x 20..24 ft red in the camera image.
    patch, _ = calib.court_to_image([[8, 20], [12, 20], [12, 24], [8, 24]])
    cv2.fillPoly(frame, [np.round(patch).astype(np.int32)], (0, 0, 255))
    px_per_ft, side, end = 10.0, 6.0, 8.0
    top = render_topdown(frame, calib, px_per_ft=px_per_ft, side_margin=side, end_margin=end)
    col = int((10 + side) * px_per_ft)
    row = int(top.shape[0] - 1 - (22 + end) * px_per_ft) + 5  # just off the drawn net line
    assert tuple(top[row, col]) == (0, 0, 255)


def test_verification_image_renders():
    calib = Calibration.from_points(synthetic_points(noise_px=0.5), IMAGE_SIZE)
    frame = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 90, np.uint8)
    img = verification_image(frame, calib)
    assert img.shape[0] == IMAGE_SIZE[1] and img.shape[1] > IMAGE_SIZE[0]


def test_cli_calibrate_from_points_file(tmp_path):
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 90, np.uint8))
    points_path = tmp_path / "points.json"
    points_path.write_text(json.dumps({k: list(v) for k, v in synthetic_points(noise_px=0.5).items()}))
    out = tmp_path / "court1.json"
    assert main(["calibrate", "--video", str(frame_path), "--points", str(points_path), "--out", str(out)]) == 0
    assert out.exists() and (tmp_path / "court1_verify.png").exists()
