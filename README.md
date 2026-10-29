# pickleballOS

Low-cost, privacy-first pickleball analytics and ranking from one fixed camera. Thesis project.
Architecture, decisions, risks and evaluation plan: **[DESIGN.md](DESIGN.md)**.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,cpu]"  # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pip install --no-deps rtmlib  # its opencv-contrib dep clashes with opencv-python
.venv/Scripts/python -m pytest
```

**NVIDIA GPU (optional):** use `.[dev,gpu]` instead of `.[dev,cpu]` (uninstall `onnxruntime` first; the
two builds overwrite each other). CUDA and cuDNN come as pip wheels, so only the NVIDIA driver is
needed. Then add `--device cuda` to `pbos track`. The CPU path stays the reference; the GPU is an
optional speed tier.

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

**Temporal tracking.** The fitted arc predicts where the ball goes next. The track coasts through
short occlusions (drawn as a thin, dim ring), picks the ball up again after hits, and runs a
relaxed second detection near the prediction for motion-blurred frames. On a 10 s rally that took
the track from 20 fragments to one continuous track (detections 228 → 252 of 299 frames). Over the
whole 80 s sample, ball detection went from 53% to 65% of frames, and 76% of frames are tracked
(the rest is mostly dead time and the cut-away).
## Phase 4 (part 2) — audio contacts, fusion, rallies

Also inside `pbos track`. The soundtrack is scanned for paddle pops: sharp 1.5–6 kHz bursts at
least 12 dB above the past second's background. Pops are fused with the ball track's direction
changes and the players' wrists from pose:

- **hit**: a pop plus a direction change at a wrist, or a pop while a ball in flight reaches a wrist
  (the contact itself hidden from the camera). Credited to that player.
- **bounce**: a low, falling direction change away from the players; placed exactly on the court.
- dropped: pops from other courts, a dead ball rolling or being caught.

Hits group into rallies (gaps under 3 s). The overlay gets a **RALLY TOTAL · OBSERVED CONTACTS**
card, and the log gets `contact` / `bounce` events (with their source: `audio+vision` or `audio`),
raw `ball_break` events for evaluation, and a `rallies` table.

On the sample (420–500 s): 241 audio onsets → 60 hits and 10 bounces in 5 rallies. Without fusion,
vision alone had logged 85 "hits". One 17 s dink rally was checked frame by frame and is real. Exact
hit precision and recall still need labelled events.

## Learned ball detector (experimental)

A small U-Net (0.48M parameters, our own code; the idea comes from TrackNet) looks at frames t-2,
t-1 and t at full resolution and outputs a heatmap of where the ball is at t. Its peaks feed the
same tracker, so the arcs, coasting and fusion all still apply.

```bash
# needs PyTorch: CUDA build for GPU training, or the CPU build (slower)
pbos ball-train --video dsrgtsdg.mp4 --calib calib/samples/dsrgtsdg.json            # -> models/ballnet/ballnet.onnx
pbos track --video dsrgtsdg.mp4 --calib calib/samples/dsrgtsdg.json --start 380 --end 460 \
    --device cuda --ball-model models/ballnet/ballnet.onnx
```

`ball-train` first **auto-labels** the video with the classical tracker. It keeps only detections
within 3 px of a fitted 3D arc, which gives 5,474 precise labels on the sample. It holds out
380–460 s for validation, caches the frames it needs (`out/ball_dataset/`, about 1.9 GB, git-ignored)
and trains. The weights are written to `models/`, also git-ignored because they derive from
third-party footage.

**Results so far** (held-out window 380–460 s, threshold 0.7):

| | Classical | Learned, CPU-trained (128 px crops, 10 epochs, ~30 min) | Learned, GPU-trained (256 px crops, 12 epochs, ~10 min) |
|---|---|---|---|
| Precision / recall vs held-out auto-labels | 0.86 / 0.82 | 0.85 / 0.84 | **0.87** / 0.74 |
| Median error | (label source) | 0.69 px | **0.34 px** |
| Ball detected (pipeline) | 69.5% of frames | **78.9%** | 63.3% |
| On a fitted 3D arc (speed, court map) | 44.5% | 36.2% | **55.2%** |
| Detector cost, RTX 3060 | 5 ms | 33 ms | 31 ms (CPU: ~310 ms) |

The GPU-trained model gives the most usable output: more of its detections land on physically
consistent arcs, so more frames get a ball speed and a court-map position. It finds the ball in
fewer frames, though. Lowering its threshold to 0.5 barely changes that (66.8% detected, 54.8% on
arcs). Both models learned from auto-labels, which only contain frames the classical tracker
already handled. The hard frames (blur, occlusion, slow balls) are what's missing, and the
auto-label precision/recall rows are biased for the same reason.

**Training environment.** Training doesn't need the project's ONNX Runtime environment, just
PyTorch. If PyTorch with CUDA is already installed in another Python, reuse it instead of
downloading it again:

```bash
py -3.13 -m venv --system-site-packages .venv-gpu-train   # sees the system torch
.venv-gpu-train/Scripts/python -m pip install opencv-python onnx
.venv-gpu-train/Scripts/python -m pip install --no-deps -e .
.venv-gpu-train/Scripts/pbos ball-train --video dsrgtsdg.mp4 --calib calib/samples/dsrgtsdg.json --out models/ballnet_gpu
```

Next steps, in order of impact:
1. **Human labels on hard frames.** `pbos label --video dsrgtsdg.mp4 --out out/ball_dataset/dsrgtsdg_gold.json --calib calib/samples/dsrgtsdg.json`
   shows random frames from the held-out window with a suggestion ring. Click the ball, press
   Enter to accept the ring, or `n` if it isn't visible. About 200–300 frames gives an unbiased
   test set (`pbos ball-eval --labels ...gold.json`). More, from the training range, make the next
   model better than its teacher.
2. FP16 inference to bring the ~31 ms down.
3. Use both detectors: classical candidates for recall and learned ones for precision, letting the
   tracker's arc check pick between them.

The auto-label F1 for the classical detector is below 1.0 because the frame-level test runs it
without the tracker (fresh state, strongest blob). It is still a biased comparison; only human
labels settle it.

## Compute tiers

```bash
pbos bench --video dsrgtsdg.mp4 --calib calib/samples/dsrgtsdg.json --start 420 --seconds 20
```

Runs the analysis (no video output) on each available tier: `cuda`, `cpu` (all threads), `cpu4`, `cpu2`
(thread-capped to stand in for smaller machines). Prints a table and saves `out/bench_<video>.json`.

| Tier | Analysis fps (30 fps video) |
|---|---|
| CUDA, RTX 3060 Laptop | **40.1** (real time) |
| CPU, Ryzen 7 5800H, all threads | 25.3 |
| CPU, 4 threads | 23.4 |
| CPU, 2 threads | 18.6 |

Two fixes made the GPU pay off. The detector's built-in NMS is cut out and done on the CPU (it
took 6 ms on the GPU). Pose runs as one batched call with float32 preprocessing.

## Status

| Phase | | |
|---|---|---|
| 1 | DESIGN.md | done (v0.2, decisions folded in) |
| 2 | Calibration tool + leave-one-out accuracy | done |
| 3 | Player tracking → positions → distance, court map, camera-move check | done |
| 4 | Ball tracking, 3D arcs, bounces, speed, audio contacts, rallies | done (classical ball detector) |
| 4b | Learned ball detector: auto-labels, training, ONNX, labelling and eval tools | first model trained; needs human labels to beat the classical detector |

License: [MIT](LICENSE).
