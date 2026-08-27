"""Machine-specific paths and tuning constants.

Every path here was verified present on this machine during setup; if the
pipeline is moved to another Mac these are the only values that should need
changing.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- external tools -------------------------------------------------------
ORCA_APP = Path("/Applications/Orca-Flashforge.app")
ORCA_BIN = ORCA_APP / "Contents/MacOS/Orca-Flashforge"
ORCA_DATA = Path.home() / "Library/Application Support/Orca-Flashforge"
ORCA_SYSTEM = ORCA_DATA / "system/Flashforge"

BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")
FFMPEG_BIN = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE_BIN = Path("/opt/homebrew/bin/ffprobe")

OBJCAP_BIN = ROOT / "tools/objcap/.build/release/objcap"

# --- printer --------------------------------------------------------------
# Values read from the Orca machine profile for this printer.
PRINTER_MODEL = "Flashforge Adventurer 5M Pro"
PRINTER_NOZZLE = "0.4"
MACHINE_PRESET = ORCA_SYSTEM / f"machine/{PRINTER_MODEL} {PRINTER_NOZZLE} Nozzle.json"
BED_MM = (220.0, 220.0)
BED_HEIGHT_MM = 220.0

# Flashforge stock-firmware LAN API, consumed by printer.py. Serial + check
# code come off the printer's own screen under Settings -> Network once LAN
# mode is enabled. Absent these, printer.py falls back to writing the gcode
# to disk for manual transfer -- never a hard failure.
FF_HTTP_PORT = 8898
FF_TCP_PORT = 8899
FF_SERIAL = os.environ.get("FF_SERIAL", "")
FF_CHECKCODE = os.environ.get("FF_CHECKCODE", "")
FF_HOST = os.environ.get("FF_HOST", "")

# --- pipeline tuning ------------------------------------------------------
# Frame extraction: Object Capture wants many well-spaced sharp views. Too
# few frames and geometry is missing; too many and it just costs time.
TARGET_FRAMES = 120
MIN_FRAMES = 40
MAX_FRAMES = 250
BLUR_REJECT_PERCENTILE = 25.0   # drop the blurriest quarter of candidates

# --- quality gates --------------------------------------------------------
# Scale-solve residual between the two independently-derived camera tracks.
# Absolute millimetre thresholds are kept as a coarse backstop, but they are
# NOT the primary gate: an identical residual means very different accuracy at
# a 60 mm standoff than at a 400 mm one (measured: the same 1.41 mm residual
# spanned a 6.3x range of real scale error). The relative gate below is the
# one that actually tracks accuracy.
SCALE_RMS_GOOD_MM = 1.5
SCALE_RMS_MARGINAL_MM = 4.0
SCALE_MIN_CAMERAS = 12
SCALE_MIN_INLIER_FRAC = 0.6

# Primary gate: residual as a fraction of the camera track's spatial spread.
SCALE_RMS_REL_GOOD = 0.004        # 0.4% of track spread
SCALE_RMS_REL_MARGINAL = 0.012    # 1.2%

# Independent pairwise-distance cross-check (scale.py). This compares median
# inter-camera distance ratios against the fitted scale and is genuinely
# independent of the similarity residual, which is precisely what makes it
# worth GATING on rather than merely noting.
PAIRWISE_DISAGREE_WARN = 0.02     # 2% -> marginal
PAIRWISE_DISAGREE_REJECT = 0.10   # 10% -> reject

# ChArUco pose acceptance, per frame.
MIN_CHARUCO_CORNERS = 8
MAX_REPROJ_RMS_PX = 2.0

# Run-level reprojection gate. A wrong camera model is invisible to the
# similarity residual but moves mean reprojection error by an order of
# magnitude (measured: 0.09 px -> 1.51 px for a 10% focal error, which cost
# ~5% on the part). This is the signal that catches bad intrinsics.
RUN_REPROJ_GOOD_PX = 0.30
RUN_REPROJ_MARGINAL_PX = 0.80
# Fraction of frames that must yield a board pose before we distrust the run.
MIN_FRAME_SOLVE_FRAC = 0.75

# Caliper cross-check: the ONLY check that can detect a mis-measured board
# square, which is otherwise completely invisible (a wrong square_mm produces
# a proportionally wrong part with a *better* residual). Treated as blocking.
CALIPER_WARN_PCT = 0.5
CALIPER_FAIL_PCT = 1.0
# Refuse an auto-matched caliper axis when two model dimensions are within
# this fraction of each other -- an ambiguous match silently certified a
# 2x-oversize model as 0.00% error.
CALIPER_AMBIGUOUS_FRAC = 0.15

# --- materials ------------------------------------------------------------
# Veg-tan leather weights in mm, for buck offsetting. 1 oz ~= 0.4 mm.
LEATHER_OZ_TO_MM = 0.4

# Filament shrinkage, overridden by profiles/printer_cal.json once the gauge
# cube has actually been printed and measured. These literature values are
# NOT applied unless the user opts in with --assume-shrink: silently applying
# an unmeasured guess to a metrology deliverable is exactly the sort of
# confident fiction this tool exists to avoid. Uncalibrated == compensate
# nothing, and say so.
DEFAULT_SHRINK = {"PLA": 0.003, "PETG": 0.005, "ABS": 0.008}
APPLY_UNCALIBRATED_SHRINK = False

OUT_DIR = ROOT / "out"
TARGETS_DIR = ROOT / "targets"
PROFILES_DIR = ROOT / "profiles"


def check_tools() -> dict[str, bool]:
    """Report which external dependencies are actually present."""
    return {
        "orca": ORCA_BIN.exists(),
        "blender": BLENDER_BIN.exists(),
        "ffmpeg": FFMPEG_BIN.exists(),
        "ffprobe": FFPROBE_BIN.exists(),
        "objcap": OBJCAP_BIN.exists(),
        "machine_preset": MACHINE_PRESET.exists(),
    }
