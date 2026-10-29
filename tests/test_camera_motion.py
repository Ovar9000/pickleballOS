import cv2
import numpy as np

from pbos.calibration import Calibration
from pbos.camera_motion import CameraMotionMonitor
from pbos.eventlog import EventLog

from test_calibration import IMAGE_SIZE, synthetic_points


def textured_frame(seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (IMAGE_SIZE[1] // 8, IMAGE_SIZE[0] // 8, 3), dtype=np.uint8)
    return cv2.resize(noise, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)


def shifted(frame, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, IMAGE_SIZE, borderMode=cv2.BORDER_REFLECT)


def test_steady_camera_is_steady():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    ref = textured_frame()
    st = CameraMotionMonitor(ref, calib).check(ref)
    assert st.status == "steady" and st.shift_px < 1.0


def test_moved_camera_is_rebased_and_maps_correctly():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    ref = textured_frame()
    st = CameraMotionMonitor(ref, calib).check(shifted(ref, 24, -12))
    assert st.status == "rebased"
    assert 24 < st.shift_px < 30
    # The court centre has moved by (24, -12) px in the image; the new calibration must follow it.
    px, _ = calib.court_to_image([[10.0, 22.0]])
    np.testing.assert_allclose(st.calib.image_to_court(px + [24, -12])[0], (10.0, 22.0), atol=0.1)


def test_blank_frame_is_lost_and_keeps_calibration():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    mon = CameraMotionMonitor(textured_frame(), calib)
    st = mon.check(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8))
    assert st.status == "lost"
    np.testing.assert_allclose(st.calib.H, calib.H)


def test_scene_change_flags_cuts_not_small_motion():
    calib = Calibration.from_points(synthetic_points(), IMAGE_SIZE)
    ref = textured_frame()
    mon = CameraMotionMonitor(ref, calib)
    moving_player = ref.copy()
    cv2.rectangle(moving_player, (600, 300), (640, 420), (255, 255, 255), -1)
    assert not mon.scene_changed(moving_player)
    assert mon.scene_changed(textured_frame(seed=1))


def test_eventlog_round_trip(tmp_path):
    log = EventLog(tmp_path / "s.db")
    sid = log.start_session("court1", 30.0, config={"x": 1})
    log.add_position(sid, 0.1, "P1", 3.0, 4.0)
    log.add_event(sid, 0.2, "camera_rebased", frame=6, extra={"shift_px": 25.0})
    log.end_session(sid)
    log.close()
    import sqlite3

    con = sqlite3.connect(tmp_path / "s.db")
    assert con.execute("SELECT slot, court_x, estimated FROM positions").fetchall() == [("P1", 3.0, 1)]
    assert con.execute("SELECT type FROM events").fetchall() == [("camera_rebased",)]
    assert con.execute("SELECT ended_at IS NOT NULL FROM sessions").fetchone() == (1,)
