# pickleballOS

Low-cost, privacy-first pickleball analytics and ranking from one fixed camera. Thesis project.
Architecture, decisions, risks and evaluation plan: **[DESIGN.md](DESIGN.md)**.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pip install --no-deps rtmlib  # its opencv-contrib dep clashes with opencv-python
.venv/Scripts/python -m pytest
```

Sample videos are not in the repo (third-party footage, and raw video never goes in git). Put them
in the project root or `samples/`.

## Phase 2 — court calibration

The camera is fixed, so the court is calibrated once by clicking painted line intersections.
Any **4 or more** of the 12 keypoints work; click **6+** so the tool can report a real error.

```bash
pbos keypoints                                             # list keypoint names and court coordinates
pbos calibrate --video dsrgtsdg.mp4 --at 200 --out calib/court1.json
```

In the window: click the keypoint highlighted in the mini court (camera side is at the bottom),
`s` skip a point you can't see, `u` undo, `Enter` fit and review, `q` quit. The loupe in the corner
magnifies the area under the cursor. The review screen shows the court re-projected on the frame
(with per-point error in feet) next to a top-down warp; `Enter` saves.

Outputs: `calib/court1.json` (homography image px → court ft, clicked points, rms/max error) and
`calib/court1_verify.png`.

Check that the camera hasn't moved, on any later frame:

```bash
pbos verify --calib calib/court1.json --video dsrgtsdg.mp4 --at 600
```

Scripted / reproducible calibration (no window): `--points file.json` with `{"keypoint_name": [u, v]}`.
**Accuracy without a tape measure.** The report shows two errors per keypoint. *Fit error* is how
well the fitted court matches the clicks, which flatters the fit. *Held-out* error re-fits without that
point and predicts it, using the painted lines as the ruler; that one is the honest accuracy estimate.
Click as many of the 12 points as you can see: held-out error shrinks and becomes more stable with
more points.

`calib/samples/dsrgtsdg_points.json` is a 5-point example for the corner camera in `dsrgtsdg.mp4`
(fit 0.09 ft, held-out ~0.6 ft, because only 5 points on one half are visible; the re-projected net,
which was not clicked, lands on the real net). It is valid for
~20–460 s of that video only: the camera is re-aimed at ~460 s, which `pbos verify --at 600` shows.

**Naming for side/corner cameras:** "near" is the half the camera looks into most directly, "left"
is the sideline closest to the camera. Any consistent choice works; it only fixes which way the court
map is drawn.

## Phase 3 — player tracking, distance, court map

```bash
pbos track --video dsrgtsdg.mp4 --calib calib/samples/dsrgtsdg.json --start 420 --end 500
```

What it does per frame: YOLOX person detection (every `--det-every` frames) → drop people outside
the court → RTMPose ankles for the rest → ankle midpoint through the homography → Kalman tracker in
court feet → P1–P4 slots → distance covered. A camera-move check (a cheap per-frame scene-change test,
plus an ORB check each second) re-aligns the calibration when the camera is re-aimed and pauses
tracking during cut-aways. Models download to `~/.cache/rtmlib` on first run.

Outputs in `out/` (git-ignored): `<video>_track.mp4` (overlay with source audio), `<video>.db`
(SQLite: session, 10 Hz positions, camera events) and `<video>_track_summary.json` (distances,
camera events, ms per stage, FPS).

Measured on the sample (420–500 s, CPU only, Ryzen 7 5800H, 1280×720, detector every 2nd frame):
**34.5 fps analysis** (decode 2.7, camera 1.9, detect 10.7, pose 13.0, track 0.6 ms/frame), 26 fps
including overlay render + H.264 encode. It caught the 10 s cut-away at 430 s and the camera
re-aim at 475 s.

Known limitation of the sample: its calibration only has near-half points, so far-court positions
are off by several feet (far players show up beyond the far baseline). Re-click it with points on
both halves, e.g. `pbos calibrate --video dsrgtsdg.mp4 --at 492 --out calib/dsrgtsdg_492.json`, and
run tracking from 480 s.

## Phase 4 (part 1) — ball, bounces, speed

Runs inside `pbos track`. The ball is found with classical vision (moving + optic-yellow + ball-sized
blobs, no training data). The court homography can't place a ball that is in the air, so the
tool recovers the full 3D camera from the calibration and fits each flight with a gravity arc.
That gives the ball's ground track on the court map and its horizontal speed. Bounces are placed
exactly through the homography at touchdown and logged with the hits in the `events` table.

On the sample (420–500 s, including dead time and a 10 s cut-away): ball found in 53% of frames,
3D arc in 41%, 16 bounces and 24 hits logged. The recovered camera is about 5 ft high, 7 ft outside
the near sideline, with a ~660 px focal length (≈ 88° wide lens); the projected net tape lands on the
real one. Dink speeds read 7–20 mph horizontal. CPU cost: 6 ms/frame; the whole pipeline analyses
at 25.5 fps on CPU. Speeds are estimates (drag is ignored) and haven't been checked against ground
truth yet.

## Status

| Phase | | |
|---|---|---|
| 1 | DESIGN.md | done (v0.2, decisions folded in) |
| 2 | Calibration tool + leave-one-out accuracy | done |
| 3 | Player tracking → positions → distance, court map, camera-move check | done |
| 4 | Ball tracking, 3D arcs, bounces, speed | done · audio contacts next |

License: [MIT](LICENSE).
