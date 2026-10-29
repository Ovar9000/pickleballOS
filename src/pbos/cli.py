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
    print(f"{'keypoint':<24}{'error (ft)':>12}")
    for name, err in calib.residuals_ft().items():
        print(f"{name:<24}{err:>12.3f}")
    rms, worst = calib.error_summary()
    print(f"\n{len(calib.image_points)} points  rms {rms:.3f} ft  max {worst:.3f} ft")
    if len(calib.image_points) == 4:
        print("note: with exactly 4 points the errors are 0 by construction; click 6+ to get a real check")


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
