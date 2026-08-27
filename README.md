# scan2print

Shoot a phone video of an object sitting on a printed ChArUco board. Get back a
watertight STL **in true millimetres** and gcode for a Flashforge Adventurer 5M Pro.

The point of this project is not "3D scanning". It is **scale**. Photogrammetry
gives you a shape at an arbitrary size; the printed board gives you a metric ruler
in every frame. scan2print recovers the camera track twice — once from Apple's
Object Capture (arbitrary units) and once from the ChArUco board (millimetres) —
and fits one against the other. The agreement between those two independently
derived tracks is the quality signal, and it is reported in millimetres, not as a
vibe.

```
video.mov
  -> frames      ffmpeg extract, blur reject, even temporal coverage
  -> objcap      Apple Object Capture: mesh (usdz) + camera poses (arbitrary scale)
  -> charuco     ChArUco detect + solvePnP: camera poses in board millimetres
  -> scale       robust similarity fit RECON -> BOARD  ==> mm per recon unit
  -> mesh        scale, drop the board plane, repair, orient, decimate -> STL (mm)
  -> verify      report + REVIEW GATE  (nothing prints until a human confirms)
  -> compensate  per-axis filament shrinkage pre-scale
  -> slice       Orca-Flashforge CLI -> gcode
```

---

## Install / run

Everything is already installed in this repo's venv. Nothing else is needed.

```bash
cd ~/scan2print
uv sync                      # installs the `scan2print` console script
.venv/bin/scan2print doctor  # or: .venv/bin/python -m scan2print.cli doctor
```

`doctor` checks the venv, every python package, and every external binary
(ffmpeg, Blender, Orca-Flashforge, the Object Capture wrapper, usdcat, the printer
preset). If it prints "All external tools and packages present", you are ready.

Both invocations are equivalent and both are verified:

```bash
.venv/bin/scan2print <command>
.venv/bin/python -m scan2print.cli <command>
```

If the Object Capture wrapper is missing, build it once:

```bash
cd tools/objcap && swift build -c release
```

---

## Quick start

```bash
# 1. environment check
scan2print doctor

# 2. print the target  (a Letter-safe 24 mm variant already exists in targets/)
scan2print target --square-mm 24 --marker-mm 18 --paper letter
#    -> targets/charuco_7x5_24mm_300dpi.pdf   PRINT AT 100%, NOT "fit to page"
#    -> measure the printed 100 mm bar. If it is not 100.00 mm, the page was scaled.

# 3. calibrate your phone camera (once per phone/resolution/zoom)
scan2print calibrate ~/board_photos --fix-k3
#    -> profiles/intrinsics.json, picked up automatically from then on

# 4. scan
scan2print run ~/Desktop/knife.mov --square-mm 23.96 --caliper-mm 152.4
#    stops at the review gate, shows you the numbers, waits for confirmation

# 5. printer shrinkage calibration (once per filament)
scan2print cube                      # prints instructions, writes out/calibration_cube.stl
scan2print cube --measure 29.88 29.86 30.05 --material PLA
```

Prove the whole chain works without touching a camera:

```bash
scan2print synth        # renders a scene with exact ground truth and measures the error
```

---

## The capture recipe

This matters more than any setting in the program.

1. **Tape the board down flat.** A curled corner is a metrology error you cannot
   see afterwards. Do not move the board or the object mid-shoot.
2. **Put the object near the middle of the board**, not covering more of it than
   necessary. Frames where the board is hidden contribute no metric pose.
3. **Lock focus and exposure** before you start moving (tap-and-hold on iPhone).
   Autofocus hunting changes the effective focal length mid-scan.
4. **Orbit slowly.** ~30–60 s for a full circuit. Motion blur is the number one
   cause of dropped frames, and blur is systematic — it kills whole angular
   sectors, not random frames.
5. **Vary your height.** Do at least two or three laps at different elevations
   (low ~20°, mid ~45°, high ~70°). A constant-height circle is a *degenerate*
   camera track: the solver detects it and downgrades the result every time.
6. **Keep the whole board in frame**, all the way round.
7. **Even, diffuse light.** No hard shadow crossing the board, no glare on it.
8. **Matte-spray anything shiny or dark.** Specular highlights move with the
   camera; photogrammetry reads them as texture that will not triangulate.
9. **The bottom face is not recoverable.** Object Capture cuts the model off where
   the object touches the support surface. See LIMITATIONS.

Then: **measure one real dimension of the object with calipers** and pass it as
`--caliper-mm`. That is the only check that can catch a mis-scaled printed board.

### The one trap everyone hits

A consumer printer will silently apply "fit to page" and shrink your target by a
few tenths of a percent. Every millimetre this program reports is directly
proportional to `square_mm`, so a board printed at 99.9% makes every dimension
0.1% wrong — **and the fit residual still reports "good"**, because the camera
tracks still agree perfectly with each other. Measure a printed square with
calipers and pass the real number:

```bash
scan2print run scan.mov --square-mm 23.94
```

The generated PDF carries a 100 mm horizontal and 50 mm vertical caliper bar for
exactly this reason. (At 300 dpi the renderer also quantises the square to whole
pixels — 24 mm becomes 23.9607 mm — which is recorded in the `.json` sidecar and
is used automatically if you pass `--board-json <sidecar>`.)

---

## The review gate

`run` stops after the verification report and will not slice anything until you
confirm. This is deliberate and it is not skippable by accident:

| verdict | what happens |
|---|---|
| `good` | shows the numbers, asks `y/N` |
| `marginal` | same, plus a warning to run the caliper check first |
| `reject` | **refuses outright.** `--force` is required even to be asked, and then you must type `PRINT ANYWAY` in full |

`--yes` skips the interactive prompt for automation; it still cannot approve a
rejected fit without `--force`. With no TTY and no `--yes`, the run stops with
exit code 3 and the uncompensated STL still on disk.

Exit codes: `0` ok, `1` command-specific failure, `2` user-facing error,
`3` stopped at the review gate.

---

## Accuracy

All numbers below were measured on this machine, not estimated.

**Synthetic end-to-end, with real Object Capture** (`scan2print synth`: Blender
renders a scene where every dimension is known, then the real pipeline runs on the
rendered pixels — board detection, Object Capture, scale fit, mesh conditioning):

| quantity | measured |
|---|---|
| recovered scale vs ground truth | **−0.0076 %** (30 frames), **−0.0063 %** (60 frames) |
| scale-fit residual | 0.32–0.45 mm RMS over 30–60 cameras |
| ChArUco camera-centre error | mean 0.046 mm, max 0.71 mm at ~380 mm standoff |
| reprojection RMS | 0.069 px |
| wall time | 33 s total (21 s render, 9 s Object Capture, 3 s the rest) |

Two independent `scan2print synth` runs of the same scene recovered the scale to
−0.00760 % and −0.00761 % — while the *mesh's* long axis came out 48.42 mm on one
run and 50.97 mm on the other. That is the whole story in one line: **the scale
solve is repeatable to parts in 10⁵; Object Capture's surface is repeatable to
about ±1.5 mm.** Which of those two you are limited by depends entirely on your
subject.

**Scale arithmetic in isolation** (`--recon ideal`, an error-free reconstruction
track, so the only error sources are board detection and the similarity fit):
recovered 49.9954 × 29.9970 × 19.9971 mm against a true 50 × 30 × 20 mm object —
worst axis **0.015 %**.

**Full `run` on a video, including slicing** (the synthetic scene re-encoded as
H.264, so the frame-extraction path is exercised too):

| | long axis | mid axis | height |
|---|---|---|---|
| truth | 50.00 | 30.00 | 20.00 |
| with a correct calibration | 50.05 (+0.10 %) | 28.55 (−4.8 %) | 20.06 (+0.3 %) |
| uncalibrated (Object Capture's own intrinsics) | 50.36 (+0.72 %) | 28.93 (−3.6 %) | 19.91 (−0.4 %) |

Read that table carefully, because it is the honest summary of this tool:

* **Scale is excellent** — sub-0.1 % when calibrated, i.e. well under a 0.4 mm
  nozzle's resolution on a 100 mm part.
* **Calibrating your camera is worth ~0.6 %.** Object Capture's self-recovered
  intrinsics are a usable fallback, not a substitute.
* **The mid axis is short by ~1.4 mm, and that is not a scale error.** It is
  Object Capture's own surface reconstruction plus the geometry lost under the
  `--z-tol` ground cut. Re-shooting will reduce it; it will not remove it.

Measured `--z-tol` behaviour on that scan (mm above the board where the ground is
cut away): `1.0` → 47.2 mm (clean object, bottom slice lost), `0.5` → 53.8 mm,
`0.25` → 79.2 mm, `0.1` → 202.7 mm (the board plane itself is now part of the
"object"). 1.0 mm is the default because the failure at the low end is far worse
than the failure at the high end.

Slicing the resulting part produced real gcode: 100 layers, 18 m 25 s, 7.60 g PLA.

**None of this has been validated on a real phone video of a real object yet.**
Synthetic renders have no sensor noise, no defocus, no motion blur, no rolling
shutter, zero lens distortion and a perfectly flat, perfectly printed board. They
prove the geometry and the unit bookkeeping are right. They do not bound
real-world accuracy. The only honest validation is a printed part measured with
calipers.

---

## LIMITATIONS

Read this before you trust a number.

* **The contact face is missing.** Object Capture truncates the model where the
  object meets the board — it never sees that surface. Measured on a synthetic
  ellipsoid: the base was cut 24.6 mm above true. `run` cuts the ground plane
  1 mm up and then drops the cut back to z = 0 so the *height* stays correct and
  the solid stays closed, but the actual bottom geometry is a flat lie. **For a
  sheath or holster formed around the replica, that is exactly the face you care
  about.** Shoot the object in two orientations, or reconstruct the base by hand.
* **Thin, shiny, dark or featureless objects reconstruct badly.** A bare knife
  blade is close to the worst case: specular, low-texture, and thinner than the
  reconstruction's own noise floor. Matte spray (or dry shampoo / talc) is not
  optional for these. Object Capture will *refuse* genuinely low-texture input —
  verified: 24 near-featureless frames reached 89 % alignment and then failed.
* **Object Capture is not deterministic.** Identical-looking input produced
  meshes whose longest axis differed by 2.5 mm between two runs (48.42 vs
  50.97 mm on a 50 mm object) while the recovered scale was identical to five
  decimal places. If a dimension matters, scan twice and compare.
* **Sub-millimetre edges will not survive.** Fine detail is smoothed by the
  reconstruction, then again by decimation, then again by a 0.4 mm nozzle.
  Anything under ~1 mm should be treated as decorative, not dimensional.
* **A mis-scaled printed board is invisible to every internal check.** The fit
  residual cannot see it. Only the caliper cross-check can.
* **The blur filter is relative, not absolute.** It drops the blurriest ~25 % of
  candidates. If the *entire* video is motion-blurred, 75 % of it still passes.
* **The default 7×5 @ 30 mm board (210 × 150 mm) does not fit Letter or A4 at
  100 %.** Use the 24 mm variant in `targets/`, or the print dialog will "fit to
  page" and quietly corrupt every measurement.
* **Do not calibrate from your scan orbit.** Measured: calibrating from 14 frames
  of an orbit gave fx = 1840 against a true 1867 (−1.4 %) and cy off by 79 px.
  Calibration needs close, strongly tilted, varied views of the board — a dozen
  deliberate photos, not a slice of the scan.
* **HDR footage is clipped, not tone-mapped.** 10-bit HLG iPhone video is
  converted to 8-bit RGB; shooting into a window may blow out the board.
* **Disk.** Frames are lossless PNG: roughly 1 GB per 120 frames at 4K, ~4× that
  transiently before candidates are deleted (`--keep-intermediates` keeps them).
* **Object Capture writes only `.usdz`.** Every other format the SDK is asked for
  is rejected; conversion to STL happens downstream through Blender.
* **Shrinkage is NOT compensated until you measure it.** Until the gauge cube is
  printed and measured, nothing is pre-scaled at all, and `run` says so every
  time. Your part will be small by whatever this printer's real shrinkage is
  (typically 0.2–0.5 %). `--assume-shrink` substitutes an unmeasured literature
  figure; that is a guess, not a measurement, and is off by default deliberately.
* **NOTHING HERE HAS BEEN VALIDATED ON A REAL PHONE VIDEO.** Every accuracy
  number in this README comes from Blender renders or an H.264 re-encode of one:
  no sensor noise, no defocus, no motion blur, no rolling shutter, no real lens
  distortion. These are best-case figures and real-world accuracy is *unmeasured*.
  The first real scan is an experiment, not a production run.
* **~1 % dimensional error can still pass every automated check.** Measured: a
  1 % focal-length error yields +0.98 % on the part at 0.23 px mean reprojection
  error, under the gate. The automated witnesses bound gross failure, not fine
  accuracy. The caliper cross-check is what turns "plausible" into "verified" —
  which is why `run` refuses to produce G-code without one.
* **A wrong `--square-mm` is mathematically unbounded and anti-correlated with
  every automated gate.** Telling the tool the squares are 24 mm when they are
  30 mm gives a 21 % error while the fit residual *improves*. No internal check
  can ever detect this, because every dimension is defined relative to that
  number. The caliper check catches it (verified: blocks at −21.2 %); nothing
  else can.

---

## Output layout

```
out/<video-stem>/
  frames/               frame_00001.png ...   the frames actually used
  frames_manifest.json  source, timestamp, sharpness and salvage flag per frame
  recon/                model.usdz, poses.json   (Object Capture)
  <stem>_scan.stl       conditioned mesh in TRUE millimetres  <- measure this
  <stem>_print.stl      the same, pre-scaled for filament shrinkage
  gcode/                sliced gcode
  run.json              every parameter and every result of the run
```

`run.json` records the board spec, the frame selection report, the Object Capture
summary, the intrinsics actually used, the full `ScaleResult` (scale, residual,
inlier fraction, notes), the mesh statistics, the caliper check, the review-gate
decision and its reason, the slicer estimates, and per-stage timings. A demo run
is reproducible and arguable from that file alone.

---

## Command reference

| command | what it does |
|---|---|
| `run <video>` | the full pipeline, with the review gate |
| `target` | print-ready ChArUco PDF/PNG + JSON sidecar |
| `calibrate <dir>` | camera intrinsics from board photographs |
| `synth` | render a ground-truth scene and measure the pipeline's error |
| `doctor` | environment, packages, tools, project assets |
| `cube` | shrinkage gauge cube; `--measure X Y Z` records the calipers |

Useful `run` flags: `--square-mm` (measured printed square — use it),
`--caliper-mm` (cross-check a real dimension), `--frames N`, `--detail`,
`--intrinsics PATH`, `--dry-run` (stop before slicing), `--reuse` (keep existing
frames/reconstruction — fast iteration), `--keep-intermediates`, `--z-tol`,
`--material`, `--layer-mm`, `--yes`, `--force`.

Useful `synth` flags: `--frames`, `--object-mm X Y Z`, `--recon ideal|objcap|auto`,
`--board-scale 0.999` (models a printer that mis-scaled the target — the error
mode the residual cannot see), `--reuse`, `--tol-pct`.

---

## Layout

```
scan2print/
  types.py       BoardSpec, Intrinsics, FramePose, ReconPose, ScaleResult
  config.py      tool paths, printer facts, tuning constants, check_tools()
  charuco.py     target rendering, detection, calibration, metric poses
  frames.py      video probe, frame extraction, blur/coverage selection
  scale.py       robust RECON -> BOARD similarity (this is the whole ballgame)
  mesh.py        USD/STL IO, scaling, ground removal, repair, orientation
  compensate.py  filament shrinkage calibration and pre-scaling
  slicer.py      Orca-Flashforge CLI driver and gcode stats
  synth.py       Blender ground-truth scene generator
  verify.py      the verification report and the caliper gate
  cli.py         the command line
tools/objcap/    Swift wrapper around Apple's PhotogrammetrySession
tests/           end-to-end synthetic accuracy test
targets/         generated printable boards
profiles/        camera intrinsics, printer shrinkage calibration
```

Every module runs its own self-test:

```bash
.venv/bin/python -m scan2print.scale     # and charuco / mesh / verify / compensate
.venv/bin/python tests/test_pipeline.py  # full synthetic accuracy test
```
