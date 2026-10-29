"""Command-line entry point: `pbos <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import court
from .calibration import Calibration, CalibrationError, read_frame, verification_image


def _load_points(path: str) -> dict[str, tuple[float, float]]:
    d = json.loads(Path(path).read_text())
    d = d.get("image_points", d)  # accept a bare {name: [u, v]} file or a saved calibration
    return {k: (float(v[0]), float(v[1])) for k, v in d.items()}


def _write_image(path: str | Path, img) -> None:
    # cv2.imwrite returns False instead of raising when the folder is missing or the path is bad.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise ValueError(f"could not write image to {path}")


def _report(calib: Calibration) -> None:
    loo = calib.leave_one_out_ft()
    print(f"{'keypoint':<24}{'fit err (ft)':>14}{'held-out (ft)':>15}")
    for name, err in calib.residuals_ft().items():
        held = f"{loo[name]:.3f}" if name in loo else "-"
        print(f"{name:<24}{err:>14.3f}{held:>15}")
    rms, worst = calib.error_summary()
    print(f"\n{len(calib.image_points)} points  fit rms {rms:.3f} ft  max {worst:.3f} ft")
    summary = calib.loo_summary()
    if summary:
        print(f"leave-one-out rms {summary[0]:.3f} ft  max {summary[1]:.3f} ft  (accuracy estimate, no ruler needed)")
    else:
        print("note: with exactly 4 points the fit is exact and can't be checked; click 6+ for a real check")


def cmd_calibrate(args) -> int:
    frame = read_frame(args.video, args.at)
    if args.points:
        calib = Calibration.from_points(_load_points(args.points), (frame.shape[1], frame.shape[0]), args.court_id)
    else:
        from . import calibrate_ui  # imports HighGUI only when the window is needed

        calib = calibrate_ui.run(frame, args.court_id)
        if calib is None:
            print("calibration cancelled")
            return 1
    out = Path(args.out)
    calib.save(out)
    verify_out = Path(args.verify_out) if args.verify_out else out.with_name(out.stem + "_verify.png")
    _write_image(verify_out, verification_image(frame, calib))
    _report(calib)
    print(f"saved {out}\nsaved {verify_out}")
    return 0


def cmd_verify(args) -> int:
    calib = Calibration.load(args.calib)
    frame = read_frame(args.video, args.at)
    if (frame.shape[1], frame.shape[0]) != tuple(calib.image_size):
        print(f"warning: frame size {frame.shape[1]}x{frame.shape[0]} != calibrated {calib.image_size}")
    out = args.out or str(Path(args.calib).with_name(Path(args.calib).stem + f"_verify_{args.at:g}s.png"))
    _write_image(out, verification_image(frame, calib))
    _report(calib)
    print(f"saved {out}")
    return 0


def cmd_track(args) -> int:
    from .pipeline import TrackConfig, run_track

    calib = Calibration.load(args.calib)
    stem = Path(args.video).stem
    out_video = None if args.no_video else (args.out or f"out/{stem}_track.mp4")
    db = args.db or f"out/{stem}.db"
    cfg = TrackConfig(det_every=args.det_every, mode=args.mode, device=args.device, threads=args.threads,
                      ball_model=args.ball_model, ball_threshold=args.ball_threshold)
    summary = run_track(args.video, calib, args.start, args.end, out_video, db, cfg)
    summary_path = Path(db).with_name(f"{stem}_track_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved {db}" + (f"\nsaved {out_video}" if out_video else "") + f"\nsaved {summary_path}")
    return 0


# Compute tiers measurable on one machine: (ONNX Runtime device, CPU thread cap).
TIERS = {"cuda": ("cuda", None), "dml": ("dml", None), "cpu": ("cpu", None), "cpu4": ("cpu", 4), "cpu2": ("cpu", 2)}


def cmd_bench(args) -> int:
    from .pipeline import TrackConfig, run_track
    from .players import RtmlibDetector

    calib = Calibration.load(args.calib)
    frame = read_frame(args.video, args.start)
    default_cv_threads = cv2.getNumThreads()
    rows = []
    for tier in args.tiers.split(","):
        device, threads = TIERS[tier]
        try:
            detector = RtmlibDetector(args.mode, device, threads=threads)
        except Exception as e:  # provider not installed, no GPU, ...
            print(f"{tier}: unavailable ({e})")
            continue
        cv2.setNumThreads(threads or default_cv_threads)  # OpenCV work (ball, render) gets the same cap
        detector.warmup(frame)
        cfg = TrackConfig(det_every=args.det_every, mode=args.mode, device=device, threads=threads,
                          ball_model=args.ball_model)
        s = run_track(args.video, calib, args.start, args.start + args.seconds, None, None, cfg,
                      detector=detector, progress=False)
        ms = s["ms_per_frame"]
        rows.append({"tier": tier, "detector": s["detector"], "det_every": args.det_every,
                     "detect_ms": ms.get("detect"), "pose_ms": ms.get("pose"), "ball_ms": ms.get("ball"),
                     "analysis_fps": s["analysis_fps"], "video_fps": round(s["fps"], 2),
                     "realtime": bool(s["analysis_fps"] and s["analysis_fps"] >= s["fps"])})
    cv2.setNumThreads(default_cv_threads)

    print(f"\n| tier | detect ms | pose ms | ball ms | analysis fps | real time ({rows[0]['video_fps'] if rows else '?'} fps) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['tier']} | {r['detect_ms']} | {r['pose_ms']} | {r['ball_ms']} | {r['analysis_fps']} | "
              f"{'yes' if r['realtime'] else 'no'} |")
    out = Path(args.out or f"out/bench_{Path(args.video).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"video": Path(args.video).name, "start": args.start, "seconds": args.seconds,
                               "rows": rows}, indent=2))
    print(f"saved {out}")
    return 0


def _parse_ranges(text: str) -> list[tuple[float, float]]:
    return [tuple(float(x) for x in part.split("-")) for part in text.split(",") if part]


def _ball_detector_factory(kind: str, calib: Calibration, model: str | None, device: str):
    """A function returning a reset detector, reusing one instance (ONNX sessions are slow to build)."""
    from .ball import BallDetector, LearnedBallDetector, detection_roi

    det = LearnedBallDetector(model, device) if kind == "learned" else BallDetector()
    roi = detection_roi(calib)

    def make():
        det.set_roi(roi)
        return det
    return make


def cmd_ball_train(args) -> int:
    from .balldata import autolabel, cache_frames, load_labels, save_labels, triplet_frames

    stem = Path(args.video).stem
    labels_path = Path(args.labels or f"out/ball_dataset/{stem}_auto.json")
    if not labels_path.exists():
        calib = Calibration.load(args.calib)
        fps, labels = autolabel(args.video, calib, _parse_ranges(args.ranges))
        save_labels(labels_path, args.video, fps, labels, {"ranges": args.ranges})
    meta, labels = load_labels(labels_path)
    fps = meta["fps"]
    (v0, v1), = _parse_ranges(args.val)
    gap = 2.0  # seconds kept out on both sides of the validation window: neighbouring frames look alike
    val = [lb for lb in labels if v0 <= lb.frame / fps < v1][:: args.stride]
    train = [lb for lb in labels if not (v0 - gap <= lb.frame / fps < v1 + gap)][:: args.stride]
    print(f"{len(labels)} auto labels -> train {len(train)}, val {len(val)} (every {args.stride}th)")
    frame_dir = cache_frames(args.video, triplet_frames(train + val), f"out/ball_dataset/frames_{stem}")
    from .ballnet import train as train_net

    report = train_net(train, val, frame_dir, args.out, epochs=args.epochs, batch=args.batch, workers=args.workers,
                       crop=args.crop)
    report.update({"labels": str(labels_path), "val_window_s": [v0, v1]})
    Path(args.out, "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("best_val_hit_rate", "train_labels", "val_labels", "device")}))
    print(f"saved {Path(args.out, 'ballnet.onnx')}")
    return 0


def cmd_label(args) -> int:
    from . import label_ui

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    (r0, r1), = _parse_ranges(args.range)
    frames = label_ui.sample_frames(fps, r0, r1, args.n, args.seed)
    make = None
    if args.calib:
        kind = "learned" if args.ball_model else "classical"
        make = _ball_detector_factory(kind, Calibration.load(args.calib), args.ball_model, args.device)
    n = label_ui.run(args.video, args.out, frames, fps, make)
    print(f"{n} labels in {args.out}")
    return 0


def cmd_ball_eval(args) -> int:
    from .balldata import load_labels
    from .balleval import evaluate

    meta, labels = load_labels(args.labels)
    if args.range:
        (r0, r1), = _parse_ranges(args.range)
        labels = [lb for lb in labels if r0 <= lb.frame / meta["fps"] < r1]
    if args.limit:
        labels = labels[:: max(1, len(labels) // args.limit)][: args.limit]
    calib = Calibration.load(args.calib)
    results = {"labels": args.labels, "source": sorted({lb.source for lb in labels})}
    results["classical"] = evaluate(args.video, labels, _ball_detector_factory("classical", calib, None, "cpu"))
    if args.ball_model:
        results["learned"] = evaluate(args.video, labels,
                                      _ball_detector_factory("learned", calib, args.ball_model, args.device))
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


def cmd_keypoints(_args) -> int:
    for name, (x, y) in court.KEYPOINTS.items():
        print(f"{name:<24} x={x:>5.1f} ft  y={y:>5.1f} ft")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbos", description="pickleballOS tools")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="click court keypoints and save the homography")
    c.add_argument("--video", required=True, help="video file (or image) from the fixed camera")
    c.add_argument("--at", type=float, default=0.0, help="timestamp in seconds of the frame to use")
    c.add_argument("--court-id", default="court1")
    c.add_argument("--out", default="calib/court1.json")
    c.add_argument("--points", help="skip the UI: JSON {keypoint_name: [u, v]} or an existing calibration")
    c.add_argument("--verify-out", help="where to write the verification PNG (default: next to --out)")
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("verify", help="re-check a saved calibration on another frame (did the camera move?)")
    v.add_argument("--calib", required=True)
    v.add_argument("--video", required=True)
    v.add_argument("--at", type=float, default=0.0)
    v.add_argument("--out")
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser("track", help="track players, log court positions, render the overlay video")
    t.add_argument("--video", required=True)
    t.add_argument("--calib", required=True, help="calibration JSON (valid at --start)")
    t.add_argument("--start", type=float, default=0.0, help="seconds")
    t.add_argument("--end", type=float, help="seconds (default: end of video)")
    t.add_argument("--out", help="overlay video path (default: out/<video>_track.mp4)")
    t.add_argument("--db", help="SQLite event log (default: out/<video>.db)")
    t.add_argument("--no-video", action="store_true", help="skip rendering; positions and stats only")
    t.add_argument("--det-every", type=int, default=2, help="run detector + pose every N frames")
    t.add_argument("--mode", default="lightweight", choices=["lightweight", "balanced", "performance"])
    t.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"],
                   help="cuda needs onnxruntime-gpu, dml needs onnxruntime-directml")
    t.add_argument("--threads", type=int, help="cap CPU inference threads")
    t.add_argument("--ball-model", help="learned ball detector ONNX (from pbos ball-train); default classical")
    t.add_argument("--ball-threshold", type=float, default=0.7, help="learned detector peak threshold")
    t.set_defaults(func=cmd_track)

    b = sub.add_parser("bench", help="measure analysis speed per compute tier (no video output)")
    b.add_argument("--video", required=True)
    b.add_argument("--calib", required=True)
    b.add_argument("--start", type=float, default=0.0)
    b.add_argument("--seconds", type=float, default=20.0)
    b.add_argument("--tiers", default="cuda,cpu,cpu4,cpu2", help=f"comma list from {sorted(TIERS)}")
    b.add_argument("--det-every", type=int, default=2)
    b.add_argument("--mode", default="lightweight", choices=["lightweight", "balanced", "performance"])
    b.add_argument("--ball-model", help="benchmark with the learned ball detector")
    b.add_argument("--out")
    b.set_defaults(func=cmd_bench)

    bt = sub.add_parser("ball-train", help="auto-label the ball with the classical tracker and train the learned detector")
    bt.add_argument("--video", required=True)
    bt.add_argument("--calib", required=True)
    bt.add_argument("--ranges", default="20-460,480-699", help="seconds to auto-label, e.g. 20-460,480-699")
    bt.add_argument("--val", default="380-460", help="validation window in seconds (kept out of training)")
    bt.add_argument("--labels", help="existing labels JSON (skips auto-labelling)")
    bt.add_argument("--stride", type=int, default=1, help="use every Nth label")
    bt.add_argument("--epochs", type=int, default=12)
    bt.add_argument("--batch", type=int, default=32)
    bt.add_argument("--workers", type=int, default=4)
    bt.add_argument("--crop", type=int, default=256, help="training crop size in px (multiple of 8)")
    bt.add_argument("--out", default="models/ballnet")
    bt.set_defaults(func=cmd_ball_train)

    lb = sub.add_parser("label", help="click the ball (or mark it hidden) on random frames: ground truth")
    lb.add_argument("--video", required=True)
    lb.add_argument("--out", required=True, help="labels JSON (created or resumed)")
    lb.add_argument("--range", default="380-460", help="seconds to sample from, e.g. 380-460")
    lb.add_argument("--n", type=int, default=200)
    lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--calib", help="show a detector suggestion ring (needs the calibration)")
    lb.add_argument("--ball-model", help="suggest with the learned detector instead of the classical one")
    lb.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"])
    lb.set_defaults(func=cmd_label)

    be = sub.add_parser("ball-eval", help="precision/recall of ball detectors against labels")
    be.add_argument("--video", required=True)
    be.add_argument("--calib", required=True)
    be.add_argument("--labels", required=True)
    be.add_argument("--ball-model")
    be.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"])
    be.add_argument("--limit", type=int, help="evaluate an evenly spaced subset of this size")
    be.add_argument("--range", help="only labels in this time window, e.g. 380-460 (the held-out window)")
    be.add_argument("--out")
    be.set_defaults(func=cmd_ball_eval)

    k = sub.add_parser("keypoints", help="list the court keypoint names and their court coordinates")
    k.set_defaults(func=cmd_keypoints)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CalibrationError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
