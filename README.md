# pickleballOS

Low-cost, privacy-first pickleball analytics and ranking from one fixed camera. Thesis project.
Architecture, decisions, risks and evaluation plan: **[DESIGN.md](DESIGN.md)**.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows; use .venv/bin/python on Linux/macOS
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
`calib/samples/dsrgtsdg_points.json` is a 5-point example for the corner camera in `dsrgtsdg.mp4`
(rms 0.09 ft; the re-projected net, which was not clicked, lands on the real net). It is valid for
~20–460 s of that video only: the camera is re-aimed at ~460 s, which `pbos verify --at 600` shows.

**Naming for side/corner cameras:** "near" is the half the camera looks into most directly, "left"
is the sideline closest to the camera. Any consistent choice works; it only fixes which way the court
map is drawn.

## Status

| Phase | | |
|---|---|---|
| 1 | DESIGN.md | done (draft for review) |
| 2 | Calibration tool | done |
| 3+ | Player tracking onward | waiting for design review |
