"""Phase 3 offline pipeline: video -> players on the court plane -> distance, position log, overlay video.

The loop is causal (it never looks at future frames), so the same code becomes the live worker in
Phase 8 by swapping the frame source for a camera and pacing it to wall-clock time.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import court
from .audio import OnsetDetector, load_audio
from .ball import BallDetector, BallTracker, detect_and_track, detection_roi
from .calibration import Calibration
from .camera3d import Camera3D, CameraModelError
from .camera_motion import CameraMotionMonitor
from .eventlog import EventLog
from .fusion import ContactFusion, PlayerSnapshot, RallyTracker
from .players import LEFT_WRIST, RIGHT_WRIST, PersonDetection, PlayerDetector
from .render import render_frame
from .tracking import CourtTracker, DistanceMeter, Measurement


@dataclass
class TrackConfig:
    det_every: int = 2  # run detector + pose every N frames; Kalman predicts in between
    mode: str = "lightweight"
    device: str = "cpu"  # cpu | cuda | dml
    threads: int | None = None  # cap CPU threads for inference (emulates smaller machines)
    side_margin_ft: float = 4.0  # people further outside the court than this are ignored
    end_margin_ft: float = 10.0
    camera_check_s: float = 1.0
    log_hz: float = 10.0
    kp_thr: float = 0.3
    ankle_sigma_px: float = 2.0
    box_sigma_px: float = 4.0


class VideoSink:
    """H.264 via an ffmpeg pipe (with the source audio when available); falls back to OpenCV mp4v."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], audio_src: str | None = None,
                 audio_start: float = 0.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        ffmpeg = shutil.which("ffmpeg")
        self.proc = self.writer = None
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps}", "-i", "-"]
            if audio_src:
                cmd += ["-ss", f"{audio_start}", "-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.writer.release()


def measure(dets: list[PersonDetection], calib: Calibration, cfg: TrackConfig) -> tuple[list[Measurement], Counter]:
    out, cues = [], Counter()
    for d in dets:
        uv, cue = d.ground_point(cfg.kp_thr)
        xy = calib.image_to_court([uv])[0]
        if not court.in_court(*xy, side_margin=cfg.side_margin_ft, end_margin=cfg.end_margin_ft):
            continue
        sigma_px = cfg.box_sigma_px if cue == "box" else cfg.ankle_sigma_px
        out.append(Measurement(xy, max(0.15, sigma_px * calib.ft_per_px(uv)), d.box, cue, d.keypoints, d.kp_scores))
        cues[cue] += 1
    return out, cues


def detect_players(frame: np.ndarray, detector: PlayerDetector, calib: Calibration, cfg: TrackConfig,
                   timings: dict) -> list[PersonDetection]:
    t0 = time.perf_counter()
    boxes = detector.detect(frame)
    t1 = time.perf_counter()
    # Gate on the cheap box-bottom point first, so pose only runs for people near this court.
    keep = []
    for b in boxes:
        xy = calib.image_to_court([PersonDetection(b).box_bottom()])[0]
        if court.in_court(*xy, side_margin=cfg.side_margin_ft + 2, end_margin=cfg.end_margin_ft + 2):
            keep.append(b)
    boxes = np.array(keep).reshape(-1, 4)
    kps, scores = detector.pose(frame, boxes)
    t2 = time.perf_counter()
    timings["detect"] += t1 - t0
    timings["pose"] += t2 - t1
    return [PersonDetection(b, k, s) for b, k, s in zip(boxes, kps, scores)]


def camera_or_none(calib: Calibration) -> Camera3D | None:
    try:
        return Camera3D.from_calibration(calib)
    except CameraModelError:
        return None  # ball is still tracked in 2D and bounces are still placed; no 3D arcs or speed


def run_track(video: str, calib: Calibration, start: float = 0.0, end: float | None = None,
              out_video: str | None = None, db_path: str | None = None, cfg: TrackConfig | None = None,
              detector: PlayerDetector | None = None, progress: bool = True) -> dict:
    cfg = cfg or TrackConfig()
    if detector is None:
        from .players import RtmlibDetector

        detector = RtmlibDetector(cfg.mode, cfg.device, threads=cfg.threads)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(end if end is not None else total, total)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    n_frames = int((end - start) * fps)

    log = EventLog(db_path) if db_path else None
    sid = log.start_session(calib.court_id, fps, "offline",
                            {"video": Path(video).name, "start": start, "end": end, "detector": detector.name,
                             **asdict(cfg)}) if log else None
    sink = None
    tracker = CourtTracker()
    meters: dict[str, DistanceMeter] = defaultdict(DistanceMeter)
    slot_track: dict[str, int] = {}
    monitor = None
    cam_status = "steady"
    cam_every = max(1, int(round(cfg.camera_check_s * fps)))
    log_every = max(1, int(round(fps / cfg.log_hz)))
    timings: dict[str, float] = defaultdict(float)
    cues: Counter = Counter()
    camera_events = []
    active_calib = calib
    camera3d = camera_or_none(calib)
    ball_det = BallDetector()
    ball_det.set_roi(detection_roi(calib))
    ball_tr = BallTracker(calib, camera3d)
    ball_shift = 0.0  # camera shift the ball geometry was last rebuilt for
    ball_seen = -1e9
    bounces: list[tuple[float, np.ndarray]] = []
    ball_stats: Counter = Counter()
    try:
        onsets = OnsetDetector().detect(load_audio(video, start, end), start)
    except Exception:  # no audio track or no ffmpeg: contacts then come from vision alone
        onsets = []
    next_onset = 0
    fusion = ContactFusion(require_audio=bool(onsets))
    rallies = RallyTracker()
    contact_stats: Counter = Counter()

    frames = range(n_frames)
    if progress:
        from tqdm import tqdm

        frames = tqdm(frames, unit="f", desc="tracking")
    wall0 = time.perf_counter()
    done = 0
    for i in frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        timings["decode"] += time.perf_counter() - t0
        if not ok:
            break
        t = start + i / fps
        if sink is None and out_video:
            sink = VideoSink(out_video, fps, (frame.shape[1], frame.shape[0]), video, start)

        t0 = time.perf_counter()
        if monitor is None:
            monitor = CameraMotionMonitor(frame, calib)
        elif monitor.scene_changed(frame) or i % cam_every == 0:
            st = monitor.check(frame)
            if st.status != cam_status:
                camera_events.append({"t": round(t, 2), "status": st.status, "shift_px": round(st.shift_px, 1),
                                      "inliers": st.inliers})
                if log:
                    log.add_event(sid, t, f"camera_{st.status}", frame=i, source="vision",
                                  extra={"shift_px": st.shift_px, "inliers": st.inliers})
            cam_status = st.status
            active_calib = st.calib
            if st.status == "lost":
                ball_det.reset()
            else:
                shift = st.shift_px if st.status == "rebased" else 0.0
                if abs(shift - ball_shift) > 1.5:  # rebuilding the 3D camera + masks costs ~50 ms
                    camera3d = camera_or_none(active_calib)
                    ball_tr.set_geometry(active_calib, camera3d)
                    ball_det.set_roi(detection_roi(active_calib))
                    ball_shift = shift
        timings["camera"] += time.perf_counter() - t0

        measurements = None
        if i % cfg.det_every == 0 and cam_status != "lost":
            dets = detect_players(frame, detector, active_calib, cfg, timings)
            t0 = time.perf_counter()
            measurements, c = measure(dets, active_calib, cfg)
            cues.update(c)
            timings["track"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        tracker.step(t, measurements)
        players = tracker.players()
        for slot, tr in players.items():
            if slot_track.get(slot) != tr.id:
                meters[slot].restart()  # a different track now holds this slot: no distance across the gap
                slot_track[slot] = tr.id
            if tr.last_update == t:  # measured this frame; predictions alone never add distance
                meters[slot].add(tr.xy, t)
        if log and i % log_every == 0:
            for slot, tr in players.items():
                log.add_position(sid, t, slot, float(tr.xy[0]), float(tr.xy[1]))
        timings["track"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        bs = detect_and_track(ball_det, ball_tr, frame, t) if cam_status != "lost" else ball_tr.update(t, [])
        if bs.uv is not None:
            ball_seen = t
            ball_stats["frames_detected"] += 1
        if bs.uv is not None or bs.predicted_uv is not None:
            ball_stats["frames_tracked"] += 1  # detected, or coasting on a fitted arc
        if bs.xyz is not None:
            ball_stats["frames_3d"] += 1
        for e in bs.events:  # raw direction changes, kept for evaluation; fusion decides what they were
            ball_stats["direction_changes"] += 1
            if log:
                log.add_event(sid, e.t, "ball_break", court_xy=e.court_xy, img_uv=e.uv, source="vision",
                              extra={"vision_kind": e.kind, "speed_mph_est": e.speed_mph})
        timings["ball"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        while next_onset < len(onsets) and onsets[next_onset].t <= t:
            fusion.add_onsets([onsets[next_onset]])
            next_onset += 1
        snaps = {}
        for slot, tr in players.items():
            if tr.box is None:
                continue
            wrists = np.zeros((0, 2))
            if tr.keypoints is not None and tr.kp_scores is not None:
                ok_w = [k for k in (LEFT_WRIST, RIGHT_WRIST) if tr.kp_scores[k] >= cfg.kp_thr]
                wrists = tr.keypoints[ok_w]
            snaps[slot] = PlayerSnapshot(tr.box, wrists, tr.xy.copy())
        fusion.add_frame(t, bs.uv if bs.uv is not None else bs.predicted_uv, snaps, in_flight=bs.xyz is not None)
        fusion.add_vision(bs.events)
        for c in fusion.resolve(t):
            contact_stats[f"{c.kind}_{c.source}"] += 1
            closed = rallies.on_contact(c)
            if closed and log:
                log.add_rally(sid, closed.start, closed.end, closed.hits, closed.bounces)
            if c.kind == "bounce" and c.court_xy is not None:
                bounces = [b for b in bounces if t - b[0] < 3.0] + [(c.t, c.court_xy)]
            if log:
                log.add_event(sid, c.t, "contact" if c.kind == "hit" else "bounce", slot=c.slot,
                              court_xy=c.court_xy, img_uv=c.uv, source=c.source,
                              extra={"speed_mph_est": c.speed_mph})
        closed = rallies.step(t)
        if closed and log:
            log.add_rally(sid, closed.start, closed.end, closed.hits, closed.bounces)
        timings["fusion"] += time.perf_counter() - t0

        if sink:
            t0 = time.perf_counter()
            fresh = t - ball_seen < 0.1
            ball_view = {
                "uv": bs.uv, "predicted_uv": bs.predicted_uv, "trail": bs.trail if t - ball_seen < 0.3 else [],
                # Ground track from the 3D arc while the ball is seen or briefly coasting on its arc.
                "xy": bs.xyz[:2] if bs.xyz is not None and (fresh or bs.predicted_uv is not None)
                and -0.5 < bs.xyz[2] < 30 else None,
                "mph": bs.last_mph,
                "bounces": [(xy, 1 - (t - tb) / 3.0) for tb, xy in bounces if 0 <= t - tb < 3.0],
            }
            if rallies.current is not None:
                rally_view = {"hits": rallies.current.hits, "live": True}
            elif rallies.rallies:
                rally_view = {"hits": rallies.rallies[-1].hits, "live": False}
            else:
                rally_view = None
            view = render_frame(frame, active_calib, {s: {"xy": tr.xy, "box": tr.box} for s, tr in players.items()},
                                {s: m.total_ft for s, m in meters.items()}, cam_status, ball_view, rally_view)
            timings["render"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            sink.write(view)
            timings["encode"] += time.perf_counter() - t0
        done += 1

    wall = time.perf_counter() - wall0
    cap.release()
    if sink:
        sink.close()
    if log:
        log.end_session(sid)
        log.close()

    per_frame_ms = {k: round(1000 * v / max(done, 1), 2) for k, v in timings.items()}
    analysis_ms = sum(v for k, v in per_frame_ms.items() if k not in ("render", "encode"))
    return {
        "video": Path(video).name, "start": start, "end": round(start + done / fps, 2), "frames": done, "fps": fps,
        "detector": detector.name, "det_every": cfg.det_every,
        "distance_ft_est": {s: round(m.total_ft, 1) for s, m in sorted(meters.items())},
        "distance_rejected_jumps": {s: m.rejected_jumps for s, m in sorted(meters.items())},
        "ground_point_cues": dict(cues),
        "ball": {**dict(ball_stats),
                 "detected_share": round(ball_stats["frames_detected"] / max(done, 1), 3),
                 "tracked_share": round(ball_stats["frames_tracked"] / max(done, 1), 3),
                 "fitted_3d_share": round(ball_stats["frames_3d"] / max(done, 1), 3),
                 "camera_focal_px": round(camera3d.focal_px, 1) if camera3d else None,
                 "camera_position_ft": [round(v, 1) for v in camera3d.center] if camera3d else None},
        "contacts": {**dict(contact_stats), "audio_onsets": len(onsets)},
        "rallies": [{"start": round(r.start, 2), "end": round(r.end, 2), "hits": r.hits, "bounces": r.bounces}
                    for r in rallies.rallies + ([rallies.current] if rallies.current else [])],
        "camera_events": camera_events,
        "ms_per_frame": per_frame_ms,
        "analysis_fps": round(1000 / analysis_ms, 1) if analysis_ms else None,
        "overall_fps": round(done / wall, 1) if wall else None,
        "session_id": sid,
    }
