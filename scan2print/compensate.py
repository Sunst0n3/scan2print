"""Printer shrinkage compensation.

Thermoplastics contract as they cool. On a well-tuned Adventurer 5M Pro the
error is roughly 0.2-0.5% -- on a 60 mm knife blank that is 0.12-0.30 mm,
which is the same order as the entire scan-to-print error budget the rest of
this pipeline fights for. Ignoring it silently throws that accuracy away.

The fix is a per-axis pre-scale: multiply the model by nominal/measured so the
*cooled* part lands on the number the scan reported. X and Y shrink together
(in-plane contraction of each layer); Z is usually different because layer
squish, not thermal contraction, dominates it. Hence three factors, not one.

Workflow:
    1. compensate.write_default_cal()                  -- starter profile
    2. compensate.make_calibration_cube('cube.stl')    -- print this
    3. measure the printed cube with calipers
    4. compensate.update_cal_from_measurement('PLA', (29.91, 29.93, 30.04))
    5. compensate.apply(mesh, 'PLA')                   -- every print after that

UNTIL STEP 4 HAPPENS, NOTHING IS APPLIED. config.APPLY_UNCALIBRATED_SHRINK is
False, so an uncalibrated material gets factors of exactly 1.0 and the mesh
comes out of apply() byte-for-byte the same size it went in. This module used
to quietly pre-scale by config.DEFAULT_SHRINK -- a literature guess for "PLA
in general", not a measurement of this printer with this filament -- which
means every dimension downstream carried a 0.3% fudge that no one measured and
no gate could see. In a tool whose whole claim is "the number is real", an
unmeasured correction is worse than no correction: at least an uncompensated
part is wrong in a direction the user can measure and calibrate away.

The estimate is still available, but only on explicit request:
apply(..., assume_shrink=True). describe() and the metadata returned by
apply() say which of the three states a run is in -- measured, estimated, or
nothing applied -- and why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import trimesh

from . import config

CAL_VERSION = 1
CAL_FILENAME = "printer_cal.json"

# Key apply() stamps into mesh.metadata. Also what apply() looks for to refuse
# a second pass: compensation is not idempotent, and applying it twice
# compounds the pre-scale (0.301% -> 0.603% measured) with nothing in the
# output to show it happened.
MARKER = "shrink_compensation"

# A pre-scale outside this band is almost certainly a typo or a units mistake,
# not thermal shrinkage. Refuse rather than quietly print a part 30% wrong.
MIN_FACTOR = 0.90
MAX_FACTOR = 1.10

_README = [
    "scan2print printer shrinkage calibration.",
    "",
    "Each material holds three PRE-SCALE factors. The mesh is multiplied by",
    "these before slicing so the cooled part lands on the nominal dimension.",
    "factor = nominal_mm / measured_mm, so a part that prints small gets a",
    "factor above 1.0.",
    "",
    "TO CALIBRATE (do this once per filament type -- it is what makes the",
    "millimetre claim real instead of theoretical):",
    "  1. python -m scan2print.compensate --cube out/cal_cube.stl",
    "  2. Slice and print it with the material you actually use.",
    "  3. Let it cool completely (30+ min). Measure X, Y and Z across the",
    "     flat faces with calipers, away from the marked corner notch and",
    "     away from any elephant-foot on the first layer.",
    "  4. python -m scan2print.compensate --measure PLA 29.91 29.93 30.04",
    "",
    "Until 'calibrated' is true for a material, scan2print applies NO",
    "compensation at all -- the factors stay 1.0 and the part is printed",
    "exactly as scanned. The literature estimates in config.DEFAULT_SHRINK",
    "are NOT used unless you ask for them explicitly (--assume-shrink),",
    "because an unmeasured correction is indistinguishable from a bug.",
]


# --------------------------------------------------------------------------
# calibration profile I/O
# --------------------------------------------------------------------------
def cal_path(path: str | Path | None = None) -> Path:
    """Resolve the calibration profile location."""
    return Path(path) if path is not None else config.PROFILES_DIR / CAL_FILENAME


def _blank_material(material: str) -> dict[str, Any]:
    return {
        "x": 1.0,
        "y": 1.0,
        "z": 1.0,
        "calibrated": False,
        "nominal_mm": None,
        "measured_mm": None,
        "note": f"uncalibrated -- NO compensation is applied for {material!r}; "
                f"print and measure the gauge cube to change that",
    }


def default_cal() -> dict[str, Any]:
    """The in-memory starter profile: 1.0 factors, nothing calibrated yet."""
    return {
        "_README": list(_README),
        "version": CAL_VERSION,
        "nominal_cube_mm": 30.0,
        "materials": {m: _blank_material(m) for m in sorted(config.DEFAULT_SHRINK)},
    }


def write_default_cal(path: str | Path | None = None, overwrite: bool = False) -> Path:
    """Create the starter calibration profile on disk.

    Refuses to clobber real measurements unless overwrite=True.
    """
    p = cal_path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"{p} already exists; pass overwrite=True to discard existing calibration"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(default_cal(), indent=2) + "\n")
    return p


def save_cal(cal: dict[str, Any], path: str | Path | None = None) -> Path:
    p = cal_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2) + "\n")
    return p


def load_cal(path: str | Path | None = None) -> dict[str, Any]:
    """Load the calibration profile, or the 1.0-factor starter if absent.

    A missing file is not an error -- it just means nothing has been measured
    yet and axis_factors() will fall back to config.DEFAULT_SHRINK. A file
    that exists but is malformed IS an error; silently ignoring a corrupt
    calibration is exactly the failure mode this module exists to prevent.
    """
    p = cal_path(path)
    if not p.exists():
        return default_cal()
    try:
        cal = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"calibration profile {p} is not valid JSON: {e}") from e
    if not isinstance(cal, dict) or "materials" not in cal:
        raise ValueError(f"calibration profile {p} has no 'materials' section")
    if not isinstance(cal["materials"], dict):
        raise ValueError(f"calibration profile {p}: 'materials' must be an object")
    return cal


# --------------------------------------------------------------------------
# factors
# --------------------------------------------------------------------------
def _shrink_to_factor(shrink: float) -> float:
    """Fractional shrinkage -> pre-scale factor.

    A part that shrinks by s ends up at (1-s) of nominal, so it must be grown
    by 1/(1-s) beforehand.
    """
    if not -0.5 < shrink < 0.5:
        raise ValueError(f"implausible shrinkage fraction {shrink!r}")
    return 1.0 / (1.0 - shrink)


def _is_calibrated(entry: dict[str, Any]) -> bool:
    if entry.get("calibrated"):
        return True
    # Tolerate hand-edited profiles that set factors but forget the flag.
    return any(abs(float(entry.get(a, 1.0)) - 1.0) > 1e-9 for a in "xyz")


def compensation_plan(
    cal: dict[str, Any] | None = None,
    material: str = "PLA",
    assume_shrink: bool | None = None,
) -> dict[str, Any]:
    """Decide what compensation to apply for `material`, and why.

    This is the single place the three-state decision is made, so the console
    output, the run record and the mesh metadata cannot disagree about it:

      "measured"   -- a gauge cube was printed and measured for this material
                      on this printer. The only state that earns real factors.
      "none"       -- uncalibrated and config.APPLY_UNCALIBRATED_SHRINK is
                      False (the default). Factors are exactly 1.0 and the
                      geometry is left alone.
      "estimate"   -- uncalibrated but the caller explicitly asked for the
                      config.DEFAULT_SHRINK literature value anyway.

    `assume_shrink=None` means "use the config default", which is False.
    Returns a plain dict so callers can drop it straight into JSON.
    """
    cal = load_cal() if cal is None else cal
    if assume_shrink is None:
        assume_shrink = bool(config.APPLY_UNCALIBRATED_SHRINK)
    entry = cal.get("materials", {}).get(material)
    calibrated = entry is not None and _is_calibrated(entry)

    if calibrated:
        f = np.array([float(entry["x"]), float(entry["y"]), float(entry["z"])], dtype=np.float64)
        source = "measured"
        reason = (
            f"measured on a {entry.get('nominal_mm')} mm gauge cube -> "
            f"{entry.get('measured_mm')} mm"
        )
    elif not assume_shrink:
        # The important branch. Do nothing, and be unmistakable about it.
        f = np.ones(3, dtype=np.float64)
        source = "none"
        reason = (
            f"NO COMPENSATION APPLIED: {material!r} has never been calibrated on "
            f"this printer and config.APPLY_UNCALIBRATED_SHRINK is False. The "
            f"printed part will be small by whatever this printer's real "
            f"shrinkage is (typically 0.2-0.5%), and that error is NOT corrected "
            f"here. Run 'scan2print cube', print it, measure it and feed the "
            f"numbers back to make this real. Passing assume_shrink=True instead "
            f"substitutes an unmeasured literature guess, which is not a "
            f"measurement and should not be reported as one."
        )
        if material not in config.DEFAULT_SHRINK:
            reason += f" (There is also no literature figure for {material!r}.)"
    else:
        if material not in config.DEFAULT_SHRINK:
            known = sorted(set(config.DEFAULT_SHRINK) | set(cal.get("materials", {})))
            raise ValueError(
                f"no shrinkage data for material {material!r}. Known materials: {known}. "
                f"Add it to {cal_path()} or to config.DEFAULT_SHRINK."
            )
        f = np.full(3, _shrink_to_factor(float(config.DEFAULT_SHRINK[material])), dtype=np.float64)
        source = "estimate"
        reason = (
            f"UNMEASURED ESTIMATE, applied because assume_shrink=True: "
            f"config.DEFAULT_SHRINK[{material!r}]={config.DEFAULT_SHRINK[material]} is a "
            f"literature figure for the material in general, not a measurement of "
            f"this printer. Do not quote the resulting dimensions as verified."
        )

    bad = [(a, float(v)) for a, v in zip("xyz", f) if not (MIN_FACTOR <= v <= MAX_FACTOR)]
    if bad:
        raise ValueError(
            f"compensation factors for {material!r} out of the plausible band "
            f"[{MIN_FACTOR}, {MAX_FACTOR}]: {bad}. Check {cal_path()} -- a factor "
            f"this large is a data-entry error, not thermal shrinkage."
        )

    return {
        "material": material,
        "factors": {"x": float(f[0]), "y": float(f[1]), "z": float(f[2])},
        "applied": source != "none",
        "calibrated": bool(calibrated),
        "source": source,
        "reason": reason,
    }


def axis_factors(
    cal: dict[str, Any] | None,
    material: str = "PLA",
    assume_shrink: bool | None = None,
) -> np.ndarray:
    """Per-axis pre-scale factors (x, y, z) for `material`.

    A thin view onto compensation_plan(); returns exactly [1, 1, 1] whenever
    the plan is not to compensate, so a caller that ignores the plan still
    cannot accidentally apply an unmeasured correction.
    """
    p = compensation_plan(cal, material, assume_shrink)
    return np.array([p["factors"][a] for a in "xyz"], dtype=np.float64)


def factors_from_measurement(
    nominal_mm: float, measured_mm: Sequence[float]
) -> dict[str, float]:
    """Turn caliper readings of a printed gauge cube into pre-scale factors."""
    m = np.asarray(list(measured_mm), dtype=np.float64)
    if m.shape != (3,):
        raise ValueError(f"measured_mm must be three numbers (x, y, z), got {measured_mm!r}")
    if nominal_mm <= 0:
        raise ValueError(f"nominal_mm must be positive, got {nominal_mm}")
    if np.any(m <= 0):
        raise ValueError(f"measured dimensions must be positive, got {m.tolist()}")
    f = nominal_mm / m
    bad = [(a, float(v)) for a, v in zip("xyz", f) if not (MIN_FACTOR <= v <= MAX_FACTOR)]
    if bad:
        raise ValueError(
            f"measurement implies compensation factors {bad} outside "
            f"[{MIN_FACTOR}, {MAX_FACTOR}]. Did you measure the right cube, in mm? "
            f"nominal={nominal_mm} measured={m.tolist()}"
        )
    return {"x": float(f[0]), "y": float(f[1]), "z": float(f[2])}


def update_cal_from_measurement(
    material: str,
    measured_mm: Sequence[float],
    nominal_mm: float | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Record caliper readings for `material` and persist the profile."""
    cal = load_cal(path)
    nominal = float(nominal_mm if nominal_mm is not None else cal.get("nominal_cube_mm", 30.0))
    f = factors_from_measurement(nominal, measured_mm)
    entry = cal.setdefault("materials", {}).setdefault(material, _blank_material(material))
    entry.update(f)
    entry["calibrated"] = True
    entry["nominal_mm"] = nominal
    entry["measured_mm"] = [float(v) for v in measured_mm]
    entry["note"] = (
        f"measured on a {nominal:g} mm gauge cube; "
        f"error before compensation "
        f"{', '.join(f'{a}{(v - nominal):+.3f}' for a, v in zip('XYZ', measured_mm))} mm"
    )
    save_cal(cal, path)
    return cal


# --------------------------------------------------------------------------
# the actual compensation
# --------------------------------------------------------------------------
def already_compensated(mesh: trimesh.Trimesh) -> dict[str, Any] | None:
    """The compensation record a previous apply() left on `mesh`, if any.

    Only a record that actually scaled the geometry counts. A "nothing was
    applied" marker leaves the mesh dimensionally untouched, so a later pass
    that does compensate it is legitimate rather than a double application.
    """
    prior = (getattr(mesh, "metadata", None) or {}).get(MARKER)
    if isinstance(prior, dict) and prior.get("applied"):
        return prior
    return None


def apply(
    mesh: trimesh.Trimesh,
    material: str = "PLA",
    cal: dict[str, Any] | None = None,
    assume_shrink: bool | None = None,
) -> trimesh.Trimesh:
    """Pre-scale `mesh` per axis so the cooled print lands on nominal.

    Anchored at the bounding box's XY centre and its minimum Z, so the part
    stays centred and stays sitting on the bed -- scaling about the mesh
    origin would slide it off the plate for any model not already centred.
    Returns a copy; the input is untouched.

    With no calibration for `material` this is a deliberate no-op: the factors
    are 1.0, the geometry is unchanged, and the returned metadata says so.
    See compensation_plan().

    Refuses a mesh that a previous apply() already scaled. The transform is
    not idempotent -- a second pass multiplies the pre-scale again, which is
    invisible in the output geometry and lands a part twice as far out as the
    error it was correcting.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"apply() needs a trimesh.Trimesh, got {type(mesh).__name__}")
    if mesh.is_empty:
        raise ValueError("cannot compensate an empty mesh")

    prior = already_compensated(mesh)
    if prior is not None:
        raise ValueError(
            f"this mesh has already been shrink-compensated "
            f"({prior.get('material')}, factors {prior.get('factors')}, "
            f"source {prior.get('source')!r}); applying compensation again would "
            f"compound the pre-scale. Compensate the as-scanned mesh once, not "
            f"the output of a previous apply()."
        )

    plan = compensation_plan(cal, material, assume_shrink)
    f = np.array([plan["factors"][a] for a in "xyz"], dtype=np.float64)
    lo, hi = mesh.bounds
    anchor = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]], dtype=np.float64)

    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = f
    M[:3, 3] = anchor - f * anchor

    out = mesh.copy()
    out.apply_transform(M)
    out.metadata = dict(out.metadata or {})
    out.metadata[MARKER] = dict(plan)
    return out


def describe(
    cal: dict[str, Any] | None = None,
    material: str = "PLA",
    assume_shrink: bool | None = None,
) -> str:
    """One-line human summary, loud when nothing is being corrected."""
    plan = compensation_plan(cal, material, assume_shrink)
    f = np.array([plan["factors"][a] for a in "xyz"], dtype=np.float64)
    pct = (f - 1.0) * 100.0
    head = (
        f"{material}: pre-scale x{f[0]:.5f} y{f[1]:.5f} z{f[2]:.5f} "
        f"({pct[0]:+.3f}% {pct[1]:+.3f}% {pct[2]:+.3f}%)"
    )
    if plan["source"] == "none":
        # No factors to report is the whole point; lead with that, not with
        # a row of 1.00000s that reads like a successful correction.
        return f"{head}  [NOT COMPENSATED -- {plan['reason']}]"
    return f"{head}  [{plan['reason']}]"


# --------------------------------------------------------------------------
# the gauge cube
# --------------------------------------------------------------------------
def make_calibration_cube(out_stl: str | Path, size_mm: float = 30.0) -> Path:
    """Emit the gauge cube the user prints and measures.

    Exactly `size_mm` on every axis, sitting on Z=0 with its XY centre at the
    origin. A small square notch is cut from one vertical edge to mark the +X
    +Y corner, so the printed part can be oriented unambiguously on the bench
    -- measure across the three *unnotched* faces. If the boolean engine is
    unavailable the notch is skipped; the cube's overall extents are identical
    either way, which is all the measurement needs.
    """
    if size_mm <= 0:
        raise ValueError(f"size_mm must be positive, got {size_mm}")
    out = Path(out_stl)
    out.parent.mkdir(parents=True, exist_ok=True)

    cube = trimesh.creation.box(extents=(size_mm, size_mm, size_mm))
    cube.apply_translation([0.0, 0.0, size_mm / 2.0])

    notch = size_mm * 0.1
    # Twice the notch size, centred on the corner, so exactly `notch` mm of it
    # lands inside the cube.
    marker = trimesh.creation.box(extents=(notch * 2.0, notch * 2.0, size_mm * 3.0))
    marker.apply_translation([size_mm / 2.0, size_mm / 2.0, size_mm / 2.0])
    try:
        cut = cube.difference(marker)
        # Only accept the boolean if it preserved the gauge dimensions.
        if isinstance(cut, trimesh.Trimesh) and not cut.is_empty:
            ext = cut.extents
            if np.allclose(ext, size_mm, atol=1e-6):
                cube = cut
    except Exception:
        pass  # plain cube is a perfectly valid gauge; the notch is a convenience

    cube.export(out)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="scan2print.compensate", description=__doc__)
    ap.add_argument("--init", action="store_true", help="write the starter calibration profile")
    ap.add_argument("--overwrite", action="store_true", help="allow --init to clobber")
    ap.add_argument("--cube", metavar="OUT.stl", help="emit the gauge cube STL")
    ap.add_argument("--size", type=float, default=30.0, help="gauge cube size in mm")
    ap.add_argument(
        "--measure",
        nargs=4,
        metavar=("MATERIAL", "X", "Y", "Z"),
        help="record caliper readings of the printed gauge cube",
    )
    ap.add_argument("--show", metavar="MATERIAL", help="print the factors in force")
    ap.add_argument(
        "--assume-shrink",
        action="store_true",
        help="use the UNMEASURED config.DEFAULT_SHRINK estimate for an "
             "uncalibrated material instead of applying nothing",
    )
    a = ap.parse_args(argv)

    did = False
    if a.init:
        print(f"wrote {write_default_cal(overwrite=a.overwrite)}")
        did = True
    if a.cube:
        p = make_calibration_cube(a.cube, a.size)
        m = trimesh.load(p, force="mesh")
        print(f"wrote {p}  extents={np.round(m.extents, 6).tolist()} mm")
        did = True
    if a.measure:
        mat, *vals = a.measure
        update_cal_from_measurement(mat, [float(v) for v in vals])
        print(describe(material=mat, assume_shrink=a.assume_shrink))
        did = True
    if a.show:
        print(describe(material=a.show, assume_shrink=a.assume_shrink))
        did = True
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
