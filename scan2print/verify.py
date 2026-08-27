"""Human-facing scale verification -- the review gate before anything prints.

Everything upstream of this module produces numbers. This module's job is to
put those numbers in front of a person in a form that makes a bad scan
obviously bad, because the failure this project actually has to prevent is a
*confidently wrong millimetre*: a reconstruction that looks perfect, slices
cleanly, prints, and comes out 3% small -- which for a leather sheath formed
around the replica means the whole piece is scrap.

Three gates live here:

  assess()       -- the honest verdict. The fit residual is NOT an accuracy
                    measurement: it only says the two camera tracks agree with
                    each other. assess() folds in the genuinely independent
                    witnesses (reprojection error, frame solve rate, the
                    pairwise-distance cross-check, intrinsics provenance) and
                    downgrades the solver's own grade accordingly, because
                    every one of those can move several percent of real
                    dimension while the residual improves.

  report()       -- the automatic gate. Everything the pipeline measured about
                    itself: fit residual, camera count, inlier fraction, final
                    bounding box, bed fit, every independent witness, and every
                    diagnostic note the solver emitted, colour-coded, with
                    reshoot advice when the verdict is not "good".

  caliper_gate() -- the human gate, and the only one that can catch a
                    systematically mis-scaled *board* (a printer that output the
                    target at 99.9% produces a beautifully low fit residual and
                    a 0.1% wrong part -- the residual cannot see it). The user
                    measures one real dimension with calipers AND says which
                    dimension it was; we report the discrepancy as a percentage.
                    It refuses to guess which dimension was measured: picking
                    the model extent nearest the reading certifies whatever the
                    model already says, and reported 0.00% error on a model that
                    was uniformly twice its true size.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config
from .types import BoardSpec, Intrinsics, ScaleResult

__all__ = [
    "QUALITY_STYLE",
    "advice_for",
    "assess",
    "relative_residual",
    "pairwise_disagreement",
    "report",
    "caliper_gate",
    "resolve_caliper_axis",
    "CALIPER_AXES",
    "format_caliper_result",
]

# Verdict -> (rich style, one-word gloss)
QUALITY_STYLE: dict[str, tuple[str, str]] = {
    "good": ("bold green", "ACCEPT"),
    "marginal": ("bold yellow", "REVIEW"),
    "reject": ("bold red", "DO NOT PRINT"),
}

# A caliper cross-check disagreeing by more than this is treated as a failure
# rather than noise. 1% of a 100 mm object is 1 mm -- far outside anything the
# fit residual should allow, so it means a real systematic error (usually the
# printed board's square size). Thresholds live in config; these names are kept
# because they are part of caliper_gate()'s signature.
CALIPER_WARN_PCT = config.CALIPER_WARN_PCT
CALIPER_FAIL_PCT = config.CALIPER_FAIL_PCT

# good < marginal < reject; used to take the worst of several witnesses.
_QUALITY_RANK = {"good": 0, "marginal": 1, "reject": 2}
_STATUS_STYLE = {"good": "green", "marginal": "yellow", "reject": "red",
                 "unknown": "dim"}

_GENERIC_RESHOOT = [
    "Orbit slower. Motion blur is the single biggest cause of dropped frames.",
    "Vary your HEIGHT as you orbit -- a constant-height circle is a degenerate "
    "camera track and the solver will flag it every time.",
    "Keep the whole board in frame in every shot. Partial board = no metric pose "
    "for that frame.",
    "Lock focus and exposure (tap-and-hold on iPhone) before you start moving.",
    "Even, diffuse light. No hard shadows crossing the board, no glare on it.",
    "Matte-spray or dust shiny/dark objects. Specular highlights move with the "
    "camera and photogrammetry treats them as texture that will not triangulate.",
]


def _fmt_mm(v: float | None, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _fmt_xyz(v: Sequence[float] | None, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    return " x ".join(f"{float(x):.{nd}f}" for x in v)


def advice_for(res: ScaleResult, mesh_stats: dict[str, Any] | None = None) -> list[str]:
    """Specific, actionable reasons this scan is not trustworthy.

    Driven off the actual measured numbers, not off the verdict string, so the
    advice tells the user what to change rather than restating the grade.
    """
    tips: list[str] = []

    if res.rms_mm > config.SCALE_RMS_MARGINAL_MM:
        tips.append(
            f"Fit residual {res.rms_mm:.2f} mm is above the reject gate "
            f"({config.SCALE_RMS_MARGINAL_MM:.1f} mm). The ChArUco camera track and the "
            "reconstruction camera track disagree by more than the board can explain: "
            "usually motion blur, a partially-visible board, or an out-of-date "
            "camera calibration. Recalibrate, then reshoot slower."
        )
    elif res.rms_mm > config.SCALE_RMS_GOOD_MM:
        tips.append(
            f"Fit residual {res.rms_mm:.2f} mm is above the 'good' gate "
            f"({config.SCALE_RMS_GOOD_MM:.1f} mm) but below reject. The scale is "
            "probably usable; a slower orbit with the board fully visible should "
            "tighten it."
        )

    if res.n_cameras < config.SCALE_MIN_CAMERAS:
        tips.append(
            f"Only {res.n_cameras} cameras support this fit (want >= "
            f"{config.SCALE_MIN_CAMERAS}). Shoot for longer, and keep the board fully "
            "in frame -- a frame that loses the board contributes nothing metric."
        )

    if res.inlier_frac < config.SCALE_MIN_INLIER_FRAC:
        tips.append(
            f"Only {res.inlier_frac * 100:.0f}% of shared frames agreed with the fit "
            f"(want >= {config.SCALE_MIN_INLIER_FRAC * 100:.0f}%). A large minority of "
            "frames produced poses inconsistent with the rest -- board glare, a bent or "
            "curled target, or the board moving mid-shoot. Tape the board flat and do "
            "not touch it during the orbit."
        )

    joined = " ".join(res.notes).lower()
    if "planar" in joined or "flat" in joined:
        tips.append(
            "The camera track is nearly FLAT. Orbit at two or three different heights "
            "(e.g. one lap low, one lap at 45 degrees, one lap high). A single-height "
            "circle cannot pin the scale down in Z."
        )
    if "collinear" in joined or "straight-line" in joined or "linear" in joined:
        tips.append(
            "The camera track is nearly a STRAIGHT LINE. Walk all the way around the "
            "object; a back-and-forth sweep across one face is degenerate."
        )
    if "collapsed" in joined:
        tips.append(
            "The robust fit collapsed to the minimum sample -- almost nothing in this "
            "scan agreed with anything else. Treat every number on this page as "
            "meaningless and reshoot from scratch."
        )

    if mesh_stats:
        if mesh_stats.get("watertight") is False:
            tips.append(
                "The conditioned mesh is NOT watertight, so its volume is undefined and "
                "the slicer may produce odd internal walls. Usually means the "
                "reconstruction has holes: featureless, shiny or very thin regions. "
                "Matte spray and a denser orbit around the problem area."
            )
        if mesh_stats.get("fits_bed") is False:
            bed = mesh_stats.get("bed_mm")
            tips.append(
                f"Object does not fit the {_fmt_xyz(bed, 0)} mm bed. It must be split, "
                "or printed at reduced scale (which defeats the point of this tool)."
            )

    tips.extend(_GENERIC_RESHOOT)
    return tips


# ==========================================================================
# independent accuracy witnesses
# ==========================================================================
#
# The similarity residual is a SELF-consistency measure: it says the ChArUco
# camera track and the reconstruction camera track agree with each other. It
# says nothing about whether either is metrically right, and it is measurably
# ANTI-correlated with the two error modes that actually cost millimetres --
# a mis-measured board square and wrong intrinsics both improve it. Everything
# below is a signal the residual cannot produce, which is exactly why these
# and not the residual are what gets gated.

# Note text emitted by scale.solve_scale(); parsed rather than plumbed because
# ScaleResult carries no numeric field for either quantity.
_SPREAD_RE = re.compile(r"track spread\s+([-\d.]+)\s*/\s*([-\d.]+)\s*/\s*([-\d.]+)\s*mm")
_REL_RE = re.compile(
    r"relative residual\s+([\d.]+)\s*%"
    r"|=\s*([\d.]+)\s*%\s*of\s+(?:the\s+[\d.]+\s*mm\s+)?(?:camera-)?track spread")
_CROSSCHECK_RE = re.compile(r"(?:cross-check|pairwise)[^%]*?([\d.]+)\s*%")
_NO_CROSSCHECK = ("cross-check unavailable", "no cross-check available")


def relative_residual(res: ScaleResult) -> float | None:
    """Fit residual as a fraction of the camera track's spatial spread.

    The absolute residual means different things at different standoffs; this
    is the form config.SCALE_RMS_REL_* gates on. Returns None when the solver
    did not report a track spread.
    """
    rel = getattr(res, "rms_rel", None)
    if rel is not None:
        return float(rel)
    for n in res.notes:
        m = _REL_RE.search(n)
        if m:
            return float(m.group(1) or m.group(2)) / 100.0
        m = _SPREAD_RE.search(n)
        if m:
            spread = [abs(float(v)) for v in m.groups()]
            norm = float(sum(v * v for v in spread)) ** 0.5
            if norm > 0.0:
                return float(res.rms_mm) / norm
    return None


def pairwise_disagreement(res: ScaleResult) -> float | None:
    """Fractional disagreement between the pairwise-distance cross-check and
    the fitted scale, as reported by the solver. None when it said nothing."""
    val = getattr(res, "pairwise_rel", None)
    if val is not None:
        return float(val)
    for n in res.notes:
        if "cross-check" in n.lower() or "pairwise" in n.lower():
            m = _CROSSCHECK_RE.search(n)
            if m:
                return float(m.group(1)) / 100.0
    return None


def _worst(*qualities: str) -> str:
    return max(qualities, key=lambda q: _QUALITY_RANK.get(q, 2))


def _w(name: str, value: str, status: str, gate: str, catches: str,
       detail: str = "") -> dict[str, Any]:
    return {"name": name, "value": value, "status": status, "gate": gate,
            "catches": catches, "detail": detail}


def assess(
    scale_res: ScaleResult,
    *,
    reproj_mean_px: float | None = None,
    reproj_max_px: float | None = None,
    frames_solved: int | None = None,
    frames_total: int | None = None,
    intrinsics: Intrinsics | None = None,
    caliper: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fold the independent witnesses into one honest quality verdict.

    Returns {"quality", "base_quality", "downgrades", "witnesses", ...}.
    `quality` is never better than scale_res.quality and is what the review
    gate and the report must key on. The caliper result is carried through and
    displayed, but it is NOT folded into `quality`: it is a separate, blocking
    verdict of its own, because it is the only one that can see a mis-measured
    board and the only one that costs a human anything to produce.
    """
    base = (scale_res.quality or "reject").lower()
    quality = base
    downgrades: list[str] = []
    witnesses: list[dict[str, Any]] = []

    def demote(to: str, why: str) -> None:
        """Record why, and never let the verdict improve."""
        nonlocal quality
        if to == "good":
            return
        quality = _worst(quality, to)
        downgrades.append(why)

    # -- 1. relative fit residual ------------------------------------------
    rel = relative_residual(scale_res)
    if rel is None:
        witnesses.append(_w(
            "relative fit residual", "n/a", "unknown",
            f"good <= {config.SCALE_RMS_REL_GOOD * 100:.1f}%",
            "a residual too big for THIS camera track",
            "solver reported no track spread"))
    else:
        st = ("good" if rel <= config.SCALE_RMS_REL_GOOD
              else "marginal" if rel <= config.SCALE_RMS_REL_MARGINAL else "reject")
        witnesses.append(_w(
            "relative fit residual", f"{rel * 100:.3f}% of track spread", st,
            f"good <= {config.SCALE_RMS_REL_GOOD * 100:.1f}%, "
            f"reject > {config.SCALE_RMS_REL_MARGINAL * 100:.1f}%",
            "a residual too big for THIS camera track",
            "an absolute mm gate is 6x looser at arm's length than up close: the "
            "same 1.41 mm residual spanned a 6.3x range of real scale error"))
        if st != "good":
            demote(st, f"relative residual {rel * 100:.3f}% (gate "
                       f"{config.SCALE_RMS_REL_GOOD * 100:.1f}%/"
                       f"{config.SCALE_RMS_REL_MARGINAL * 100:.1f}%)")

    # -- 2. run-level reprojection error -----------------------------------
    if reproj_mean_px is None:
        witnesses.append(_w(
            "reprojection RMS (mean)", "n/a", "unknown",
            f"good <= {config.RUN_REPROJ_GOOD_PX:g} px",
            "wrong camera intrinsics -- invisible to the residual"))
    else:
        st = ("good" if reproj_mean_px <= config.RUN_REPROJ_GOOD_PX
              else "marginal" if reproj_mean_px <= config.RUN_REPROJ_MARGINAL_PX
              else "reject")
        witnesses.append(_w(
            "reprojection RMS (mean)",
            f"{reproj_mean_px:.4f} px" + (f"  (max {reproj_max_px:.4f})"
                                          if reproj_max_px is not None else ""),
            st,
            f"good <= {config.RUN_REPROJ_GOOD_PX:g} px, "
            f"reject > {config.RUN_REPROJ_MARGINAL_PX:g} px",
            "wrong camera intrinsics -- invisible to the residual",
            "measured: a 10% focal error moved this 0.09 -> 1.51 px and cost ~10% of "
            "scale while the fit residual still said 'good'"))
        if st != "good":
            demote(st, f"mean reprojection {reproj_mean_px:.3f} px (gate "
                       f"{config.RUN_REPROJ_GOOD_PX:g}/{config.RUN_REPROJ_MARGINAL_PX:g} px) "
                       "-- the board does not project where this camera model says")

    # -- 3. frames that actually solved ------------------------------------
    if frames_solved is None or not frames_total:
        witnesses.append(_w("frames solved", "n/a", "unknown",
                            f"min {config.MIN_FRAME_SOLVE_FRAC * 100:.0f}%",
                            "a camera model wrong enough to lose board poses"))
    else:
        frac = frames_solved / float(frames_total)
        st = ("good" if frac >= config.MIN_FRAME_SOLVE_FRAC
              else "marginal" if frac >= config.MIN_FRAME_SOLVE_FRAC * 0.66 else "reject")
        witnesses.append(_w(
            "frames solved", f"{frames_solved}/{frames_total}  ({frac * 100:.0f}%)", st,
            f"min {config.MIN_FRAME_SOLVE_FRAC * 100:.0f}%",
            "a camera model wrong enough to lose board poses",
            "measured: the same 10% focal error dropped 8 of 40 frames"))
        if st != "good":
            demote(st, f"only {frac * 100:.0f}% of frames yielded a board pose "
                       f"(min {config.MIN_FRAME_SOLVE_FRAC * 100:.0f}%)")

    # -- 4. pairwise-distance cross-check ----------------------------------
    pw = pairwise_disagreement(scale_res)
    joined_notes = " ".join(scale_res.notes).lower()
    if pw is None and any(s in joined_notes for s in _NO_CROSSCHECK):
        witnesses.append(_w(
            "pairwise cross-check", "UNAVAILABLE", "marginal",
            f"warn > {config.PAIRWISE_DISAGREE_WARN * 100:.0f}%, "
            f"reject > {config.PAIRWISE_DISAGREE_REJECT * 100:.0f}%",
            "recon/board tracks that are not the same motion",
            "the solver could not evaluate it, so the fitted scale has no independent "
            "confirmation at all"))
        demote("marginal", "the pairwise distance cross-check could not be evaluated, so "
                           "the fitted scale has no independent confirmation")
    elif pw is None:
        witnesses.append(_w(
            "pairwise cross-check", "quiet", "unknown",
            f"warn > {config.PAIRWISE_DISAGREE_WARN * 100:.0f}%, "
            f"reject > {config.PAIRWISE_DISAGREE_REJECT * 100:.0f}%",
            "recon/board tracks that are not the same motion",
            "solver published no disagreement figure"))
    else:
        st = ("good" if pw <= config.PAIRWISE_DISAGREE_WARN
              else "marginal" if pw <= config.PAIRWISE_DISAGREE_REJECT else "reject")
        witnesses.append(_w(
            "pairwise cross-check", f"{pw * 100:.2f}% from the fitted scale", st,
            f"warn > {config.PAIRWISE_DISAGREE_WARN * 100:.0f}%, "
            f"reject > {config.PAIRWISE_DISAGREE_REJECT * 100:.0f}%",
            "recon/board tracks that are not the same motion",
            "independent of the similarity residual: it compares median inter-camera "
            "distances, so it does not share the residual's blind spots"))
        if st != "good":
            demote(st, f"pairwise distance cross-check disagrees with the fitted "
                       f"scale by {pw * 100:.1f}%")

    # -- 5. intrinsics provenance ------------------------------------------
    if intrinsics is None:
        witnesses.append(_w("intrinsics", "unknown provenance", "marginal",
                            "must be charuco-calibration",
                            "an unmeasured camera model: its error is 1:1 in mm"))
        demote("marginal", "camera intrinsics provenance is unknown")
    else:
        src = getattr(intrinsics, "source", "unknown")
        cal = bool(getattr(intrinsics, "is_calibrated", False))
        st = "good" if cal else "marginal"
        witnesses.append(_w(
            "intrinsics", f"{src}" + (f", calibration RMS {intrinsics.rms_px:.4f} px"
                                      if intrinsics.rms_px else ", not measured"),
            st, "must be charuco-calibration",
            "an unmeasured camera model: its error is 1:1 in mm",
            "" if cal else
            "Object Capture's self-estimate measured 0.33% off on ground-truth "
            "renders and is recorded as rms_px 0.0, which reads like a flawless "
            "calibration and is not one"))
        if not cal:
            demote("marginal",
                   f"intrinsics are not a measured calibration (source={src!r}); "
                   "quality is capped at 'marginal' -- run `scan2print calibrate`")

    # -- 6. caliper: reported, gated separately ----------------------------
    cal_verdict = (caliper or {}).get("verdict", "no-measurement")
    certifies = bool((caliper or {}).get("certifies_scale", False))
    cal_status = {"ok": "good", "warn": "marginal", "fail": "reject"}.get(
        cal_verdict, "unknown")
    if cal_verdict == "ok" and not certifies:
        cal_status = "unknown"
    witnesses.append(_w(
        "caliper cross-check", (caliper or {}).get("summary", cal_verdict), cal_status,
        f"warn > {config.CALIPER_WARN_PCT:g}%, fail > {config.CALIPER_FAIL_PCT:g}%",
        "A MIS-MEASURED BOARD -- nothing else here can see it",
        "the only comparison against the physical world; blocking, and reported "
        "separately from the scale verdict rather than folded into it"))

    return {
        "quality": quality,
        "base_quality": base,
        "downgrades": downgrades,
        "witnesses": witnesses,
        "relative_residual": rel,
        "pairwise_disagreement": pw,
        "reproj_mean_px": reproj_mean_px,
        "reproj_max_px": reproj_max_px,
        "frames_solved": frames_solved,
        "frames_total": frames_total,
        "intrinsics_source": None if intrinsics is None else getattr(
            intrinsics, "source", "unknown"),
        "intrinsics_calibrated": None if intrinsics is None else bool(
            getattr(intrinsics, "is_calibrated", False)),
        "caliper_verdict": cal_verdict,
        "caliper_certifies_scale": certifies,
    }


def report(
    scale_res: ScaleResult,
    mesh_stats: dict[str, Any] | None = None,
    spec: BoardSpec | None = None,
    *,
    console: Console | None = None,
    extras: dict[str, Any] | None = None,
    show_advice: bool | None = None,
    witnesses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the verification report. Returns a JSON-able summary of it.

    `witnesses` is an assess() result. Pass it: without it this prints the
    solver's own self-consistency grade, which is the exact failure this
    module exists to prevent. When it is absent the witnesses that can be
    derived from `scale_res` alone are still shown, and the rest read "n/a".

    `show_advice` defaults to "only when the verdict is not good".
    """
    con = console or Console()
    if witnesses is None:
        witnesses = assess(scale_res)
    quality = (witnesses.get("quality") or scale_res.quality or "reject").lower()
    base_quality = (witnesses.get("base_quality") or scale_res.quality or "reject").lower()
    style, verdict_word = QUALITY_STYLE.get(quality, ("bold red", "UNKNOWN VERDICT"))
    if show_advice is None:
        show_advice = quality != "good"

    scale_tbl = Table(box=ROUNDED, show_header=False, title="scale solve",
                      title_style="bold", expand=False, pad_edge=False)
    scale_tbl.add_column("k", style="cyan", no_wrap=True)
    scale_tbl.add_column("v")
    scale_tbl.add_row("verdict", Text(f"{quality.upper()}  ({verdict_word})", style=style))
    if base_quality != quality:
        scale_tbl.add_row(
            "",
            Text(f"downgraded from '{base_quality}' by the independent witnesses below",
                 style="yellow"))
    scale_tbl.add_row("scale", f"{scale_res.scale:.6g} mm per RECON unit")
    rms_style = (
        "green" if scale_res.rms_mm <= config.SCALE_RMS_GOOD_MM
        else "yellow" if scale_res.rms_mm <= config.SCALE_RMS_MARGINAL_MM else "red"
    )
    scale_tbl.add_row(
        "fit residual",
        Text(f"{scale_res.rms_mm:.3f} mm RMS", style=rms_style)
        + Text(f"   (good <= {config.SCALE_RMS_GOOD_MM}, reject > "
               f"{config.SCALE_RMS_MARGINAL_MM})", style="dim"),
    )
    cam_style = "green" if scale_res.n_cameras >= config.SCALE_MIN_CAMERAS else "yellow"
    scale_tbl.add_row(
        "cameras in fit",
        Text(f"{scale_res.n_cameras}", style=cam_style)
        + Text(f"   (min {config.SCALE_MIN_CAMERAS})", style="dim"),
    )
    inl_style = "green" if scale_res.inlier_frac >= config.SCALE_MIN_INLIER_FRAC else "yellow"
    scale_tbl.add_row(
        "inlier fraction",
        Text(f"{scale_res.inlier_frac * 100:.1f}%", style=inl_style)
        + Text(f"   (min {config.SCALE_MIN_INLIER_FRAC * 100:.0f}%)", style="dim"),
    )
    if spec is not None:
        scale_tbl.add_row(
            "board",
            f"{spec.squares_x}x{spec.squares_y} @ {spec.square_mm:g} mm squares "
            f"({spec.width_mm:g} x {spec.height_mm:g} mm)",
        )
        scale_tbl.add_row(
            "",
            Text("every reported millimetre is proportional to square_mm -- "
                 "measure the PRINTED square with calipers", style="dim italic"),
        )
    for k, v in (extras or {}).items():
        scale_tbl.add_row(str(k), str(v))

    blocks: list[Any] = [scale_tbl]

    # The witnesses table is the point of this report. The fit residual above
    # is self-consistency only; everything here is a signal it cannot produce.
    wit_tbl = Table(box=ROUNDED, show_header=True,
                    title="independent accuracy witnesses",
                    title_style="bold", expand=False, pad_edge=False)
    wit_tbl.add_column("check", style="cyan", no_wrap=True)
    wit_tbl.add_column("measured  (gate)")
    wit_tbl.add_column("verdict", no_wrap=True)
    wit_tbl.add_column("what only this can catch", ratio=2)
    for wrec in witnesses.get("witnesses", []):
        wstyle = _STATUS_STYLE.get(wrec["status"], "dim")
        catches = Text(wrec["catches"], style="bold" if wrec["name"].startswith("caliper")
                       else "")
        # The long-form explanation only earns its lines when the check fires.
        if wrec.get("detail") and wrec["status"] != "good":
            catches = catches + Text("\n" + wrec["detail"], style="dim italic")
        wit_tbl.add_row(wrec["name"],
                        Text(str(wrec["value"])) + Text(f"\n({wrec['gate']})", style="dim"),
                        Text(wrec["status"].upper(), style=wstyle), catches)
    blocks.append(wit_tbl)
    if witnesses.get("downgrades"):
        dg = Table(box=ROUNDED, show_header=False, title="why the verdict was downgraded",
                   title_style="bold yellow", expand=False, pad_edge=False)
        dg.add_column("-", style="dim", no_wrap=True)
        dg.add_column("reason")
        for d in witnesses["downgrades"]:
            dg.add_row("!", Text(str(d), style="yellow"))
        blocks.append(dg)

    if mesh_stats:
        mesh_tbl = Table(box=ROUNDED, show_header=False, title="conditioned mesh",
                         title_style="bold", expand=False, pad_edge=False)
        mesh_tbl.add_column("k", style="cyan", no_wrap=True)
        mesh_tbl.add_column("v")
        mesh_tbl.add_row("bounding box (as oriented for print)",
                         f"{_fmt_xyz(mesh_stats.get('bbox_mm'))} mm")
        if mesh_stats.get("obb_mm"):
            mesh_tbl.add_row(
                "true size (oriented box)",
                Text(f"{_fmt_xyz(mesh_stats['obb_mm'])} mm", style="bold")
                + Text("   <- quote this one", style="dim"),
            )
        if mesh_stats.get("volume_cm3") is not None:
            mesh_tbl.add_row("volume", f"{mesh_stats['volume_cm3']:.2f} cm3")
        wt = mesh_stats.get("watertight")
        mesh_tbl.add_row("watertight",
                         Text("yes" if wt else "NO", style="green" if wt else "red"))
        mesh_tbl.add_row("triangles",
                         f"{mesh_stats.get('faces', 0):,} faces / "
                         f"{mesh_stats.get('vertices', 0):,} verts")
        fits = mesh_stats.get("fits_bed")
        margin = mesh_stats.get("bed_margin_mm")
        mesh_tbl.add_row(
            "fits bed",
            Text("yes" if fits else "NO", style="green" if fits else "red")
            + Text(f"   bed {_fmt_xyz(mesh_stats.get('bed_mm'), 0)} mm, margin "
                   f"{_fmt_xyz(margin)} mm", style="dim"),
        )
        blocks.append(mesh_tbl)

    if scale_res.notes:
        note_tbl = Table(box=ROUNDED, show_header=False, title="solver notes",
                         title_style="bold", expand=False, pad_edge=False)
        note_tbl.add_column("i", style="dim", no_wrap=True)
        note_tbl.add_column("note")
        for i, n in enumerate(scale_res.notes):
            low = n.lower()
            nstyle = ("red" if any(w in low for w in ("reject", "collapsed", "fail"))
                      else "yellow" if any(w in low for w in
                                           ("warn", "marginal", "planar", "collinear",
                                            "shortfall", "rejected"))
                      else "")
            note_tbl.add_row(str(i), Text(n, style=nstyle))
        blocks.append(note_tbl)

    tips = advice_for(scale_res, mesh_stats)
    if show_advice:
        tip_tbl = Table(box=ROUNDED, show_header=False, title="how to improve this scan",
                        title_style="bold", expand=False, pad_edge=False)
        tip_tbl.add_column("-", style="dim", no_wrap=True)
        tip_tbl.add_column("advice")
        for t in tips:
            tip_tbl.add_row("*", t)
        blocks.append(tip_tbl)

    con.print(Panel(Group(*blocks),
                    title=Text(f"SCALE VERIFICATION -- {quality.upper()}", style=style),
                    border_style=style.split()[-1], expand=False))

    if quality == "reject":
        con.print(Text(
            "REJECTED: this scale is not trustworthy. Nothing will be sliced. "
            "Reshoot -- see the advice above.", style="bold red"))
    elif quality == "marginal":
        con.print(Text(
            "MARGINAL: usable but unproven. Run the caliper cross-check before you "
            "commit filament to it.", style="bold yellow"))
    if witnesses.get("caliper_verdict") in (None, "no-measurement", "no-axis", "ambiguous"):
        con.print(Text(
            "NOTE: no usable caliper reading. Every number above is derived from "
            f"square_mm; if the printed board is not the size the pipeline was told, "
            "all of it is wrong by the same ratio and nothing above can see it.",
            style="yellow"))

    return {
        "quality": quality,
        "base_quality": base_quality,
        "verdict": verdict_word,
        "scale_mm_per_recon_unit": float(scale_res.scale),
        "rms_mm": float(scale_res.rms_mm),
        "rms_rel": witnesses.get("relative_residual"),
        "n_cameras": int(scale_res.n_cameras),
        "inlier_frac": float(scale_res.inlier_frac),
        "notes": list(scale_res.notes),
        "advice": tips,
        "mesh": mesh_stats,
        "board": spec.to_json() if spec is not None else None,
        "witnesses": witnesses.get("witnesses", []),
        "downgrades": witnesses.get("downgrades", []),
        "reproj_mean_px": witnesses.get("reproj_mean_px"),
        "reproj_max_px": witnesses.get("reproj_max_px"),
        "frames_solved": witnesses.get("frames_solved"),
        "frames_total": witnesses.get("frames_total"),
        "pairwise_disagreement": witnesses.get("pairwise_disagreement"),
        "intrinsics_source": witnesses.get("intrinsics_source"),
        "intrinsics_calibrated": witnesses.get("intrinsics_calibrated"),
        "caliper_verdict": witnesses.get("caliper_verdict"),
        "caliper_certifies_scale": witnesses.get("caliper_certifies_scale"),
        "gates": {
            "rms_good_mm": config.SCALE_RMS_GOOD_MM,
            "rms_marginal_mm": config.SCALE_RMS_MARGINAL_MM,
            "rms_rel_good": config.SCALE_RMS_REL_GOOD,
            "rms_rel_marginal": config.SCALE_RMS_REL_MARGINAL,
            "min_cameras": config.SCALE_MIN_CAMERAS,
            "min_inlier_frac": config.SCALE_MIN_INLIER_FRAC,
            "run_reproj_good_px": config.RUN_REPROJ_GOOD_PX,
            "run_reproj_marginal_px": config.RUN_REPROJ_MARGINAL_PX,
            "min_frame_solve_frac": config.MIN_FRAME_SOLVE_FRAC,
            "pairwise_warn": config.PAIRWISE_DISAGREE_WARN,
            "pairwise_reject": config.PAIRWISE_DISAGREE_REJECT,
            "caliper_warn_pct": config.CALIPER_WARN_PCT,
            "caliper_fail_pct": config.CALIPER_FAIL_PCT,
        },
    }


def _dims_of(mesh_or_stats: Any) -> tuple[list[float], list[float]]:
    """(oriented extents desc, axis-aligned extents) in mm, from a mesh or stats."""
    if isinstance(mesh_or_stats, dict):
        st = mesh_or_stats
    else:
        from . import mesh as mesh_mod  # local: trimesh import is not cheap

        st = mesh_mod.stats(mesh_or_stats)
    aabb = [float(v) for v in (st.get("bbox_mm") or [])]
    obb = [float(v) for v in (st.get("obb_mm") or aabb)]
    if not aabb:
        raise ValueError("no bounding box available on this mesh/stats object")
    return sorted(obb, reverse=True), aabb


# axis token -> (source, index, human description, source-of-truth kind).
# "rank" entries are the ones a human can actually answer without knowing how
# the pipeline oriented anything: which of the three sizes did you measure?
CALIPER_AXES: dict[Any, tuple[str, int, str, str]] = {
    "longest": ("obb", 0, "the LONGEST dimension", "rank"),
    "long": ("obb", 0, "the LONGEST dimension", "rank"),
    "middle": ("obb", 1, "the MIDDLE dimension", "rank"),
    "mid": ("obb", 1, "the MIDDLE dimension", "rank"),
    "shortest": ("obb", 2, "the SHORTEST dimension", "rank"),
    "short": ("obb", 2, "the SHORTEST dimension", "rank"),
    "obb0": ("obb", 0, "the LONGEST dimension", "rank"),
    "obb1": ("obb", 1, "the MIDDLE dimension", "rank"),
    "obb2": ("obb", 2, "the SHORTEST dimension", "rank"),
    "x": ("aabb", 0, "the X extent, as the model is oriented for print", "explicit-axis"),
    "y": ("aabb", 1, "the Y extent, as the model is oriented for print", "explicit-axis"),
    "z": ("aabb", 2, "the Z extent (height off the bed)", "explicit-axis"),
    0: ("aabb", 0, "the X extent, as the model is oriented for print", "explicit-axis"),
    1: ("aabb", 1, "the Y extent, as the model is oriented for print", "explicit-axis"),
    2: ("aabb", 2, "the Z extent (height off the bed)", "explicit-axis"),
    "nearest": ("nearest", -1, "whichever model extent is nearest the reading", "nearest"),
    "auto": ("nearest", -1, "whichever model extent is nearest the reading", "nearest"),
}


def resolve_caliper_axis(axis: str | int | None) -> tuple[str, int, str, str] | None:
    """Normalise an --caliper-axis token. None means 'not stated'."""
    if axis is None:
        return None
    key = axis.strip().lower() if isinstance(axis, str) else axis
    if key not in CALIPER_AXES:
        raise ValueError(
            f"unknown caliper axis {axis!r}. Say which dimension you measured: "
            "longest / middle / shortest (recommended -- no need to know how the "
            "pipeline oriented the part), or x / y / z for a specific print axis, "
            "or 'nearest' to opt into the unsafe nearest-extent match."
        )
    return CALIPER_AXES[key]


def caliper_gate(
    mesh: Any,
    expected_mm: float | None = None,
    *,
    measured_mm: float | None = None,
    axis: str | int | None = None,
    warn_pct: float = CALIPER_WARN_PCT,
    fail_pct: float = CALIPER_FAIL_PCT,
    ambiguous_frac: float = config.CALIPER_AMBIGUOUS_FRAC,
) -> dict[str, Any]:
    """Cross-check ONE real caliper measurement against the model.

    `mesh` is a trimesh.Trimesh or a `mesh.stats()` dict. `expected_mm` (alias
    `measured_mm`) is what the calipers actually read on the physical object.
    `axis` says WHICH dimension that reading is: "longest"/"middle"/"shortest"
    (rank -- what a human can answer), "x"/"y"/"z" (print axes), or "nearest"
    to opt into matching whichever model extent is closest.

    `axis` is REQUIRED. This function used to default to the nearest extent,
    which is not a cross-check at all: it re-targets the measurement onto
    whatever the model already says, so a model uniformly twice its true size
    reported verdict "ok" at 0.0000% error, and a real -33.3% error was
    reported as 11.17%. The nearest path survives only as an explicit opt-in,
    it hard-refuses when two extents are within `ambiguous_frac` of each other,
    it never certifies the scale, and it never emits a square_mm correction --
    an auto-matched dimension cannot support one.

    Returns a dict; caller decides what to do with it. `verdict` is
    "ok" | "warn" | "fail" | "no-measurement" | "no-axis" | "ambiguous".
    `certifies_scale` is True only when the user actually stated which physical
    dimension they measured.
    """
    obb, aabb = _dims_of(mesh)
    reading = expected_mm if expected_mm is not None else measured_mm
    base = {"model_obb_mm": obb, "model_bbox_mm": aabb,
            "warn_pct": warn_pct, "fail_pct": fail_pct,
            "certifies_scale": False, "square_mm_correction": None}
    if reading is None:
        return dict(base, verdict="no-measurement", summary="not measured",
                    message=(
                        "No caliper reading supplied. The fit residual CANNOT detect a "
                        "mis-scaled printed board -- a board printed at 99.9% yields a "
                        "perfect-looking residual and a 0.1% wrong part. Measure one real "
                        "dimension and re-run with --caliper-mm <mm> --caliper-axis "
                        "longest|middle|shortest."))

    reading = float(reading)
    if reading <= 0:
        raise ValueError(f"caliper reading must be positive, got {reading}")

    resolved = resolve_caliper_axis(axis)
    if resolved is None:
        return dict(base, verdict="no-axis", measured_mm=reading,
                    summary="reading given, dimension not stated",
                    message=(
                        f"A caliper reading of {reading:g} mm was supplied but not WHICH "
                        "dimension it is. Refusing to guess: choosing the model extent "
                        "nearest the reading makes the model agree with itself by "
                        "construction -- it certified a uniformly 2x-oversize model as "
                        "0.00% error. Re-run adding --caliper-axis longest (or middle / "
                        f"shortest / x / y / z). Model extents, longest first: "
                        f"{', '.join(f'{d:.3f}' for d in obb)} mm."))

    src, idx, label, kind = resolved
    if kind == "nearest":
        order = sorted(range(len(obb)), key=lambda i: abs(obb[i] - reading))
        nearest, second = obb[order[0]], obb[order[1]] if len(obb) > 1 else None
        if second is not None and abs(second - nearest) <= ambiguous_frac * max(nearest, 1e-9):
            return dict(base, verdict="ambiguous", measured_mm=reading,
                        summary=f"ambiguous auto-match near {reading:g} mm",
                        message=(
                            f"Refusing to auto-match: model extents {nearest:.3f} mm and "
                            f"{second:.3f} mm are within {ambiguous_frac * 100:.0f}% of each "
                            f"other, so the reading {reading:g} mm cannot be attributed to "
                            "one of them. Say which you measured with --caliper-axis "
                            "longest|middle|shortest."))
        model_mm = nearest
        rank = order[0]
        which = (f"obb[{rank}] = {('longest', 'middle', 'shortest')[min(rank, 2)]} "
                 "(AUTO-MATCHED, not stated by you)")
        label = f"the {('longest', 'middle', 'shortest')[min(rank, 2)]} dimension"
        certifies = False
    else:
        seq = aabb if src == "aabb" else obb
        if idx >= len(seq):
            raise ValueError(f"axis {axis!r} not available on this mesh")
        model_mm = seq[idx]
        which = f"{src}[{idx}] = {label}"
        certifies = True

    delta = model_mm - reading
    pct = 100.0 * delta / reading
    apct = abs(pct)
    verdict = "ok" if apct <= warn_pct else "warn" if apct <= fail_pct else "fail"

    implied_square_mm = None
    if certifies and apct > 0 and model_mm > 0:
        # The whole metric chain is linear in the board's square size, so a
        # consistent discrepancy is CONSISTENT WITH the printed square being
        # this much off -- it is a hypothesis to test with the calipers on the
        # board itself, not a correction to apply. It is only meaningful at all
        # when the user said which dimension they measured; an auto-matched
        # extent would turn any model error into a confident board correction.
        implied_square_mm = reading / model_mm
    return {
        "verdict": verdict,
        "measured_mm": reading,
        "model_mm": round(model_mm, 4),
        "which_dimension": which,
        "assumed_dimension": label,
        "axis_source": kind,
        "certifies_scale": certifies,
        "delta_mm": round(delta, 4),
        "error_pct": round(pct, 4),
        "warn_pct": warn_pct,
        "fail_pct": fail_pct,
        "model_obb_mm": obb,
        "model_bbox_mm": aabb,
        "summary": (f"{pct:+.3f}% vs {label}"
                    + ("" if certifies else "  (auto-matched: proves nothing)")),
        # HYPOTHESIS ONLY: multiplying BoardSpec.square_mm by this would absorb
        # the discrepancy IF the board is the cause. Verify with calipers first.
        "square_mm_correction": None if implied_square_mm is None else round(implied_square_mm, 6),
    }


def format_caliper_result(
    res: dict[str, Any], spec: BoardSpec | None = None, *, console: Console | None = None
) -> None:
    """Print a caliper_gate() result. Loud when it disagrees."""
    con = console or Console()
    verdict = res["verdict"]
    if verdict in ("no-measurement", "no-axis", "ambiguous"):
        titles = {"no-measurement": "caliper cross-check: SKIPPED",
                  "no-axis": "caliper cross-check: REFUSED (which dimension?)",
                  "ambiguous": "caliper cross-check: REFUSED (ambiguous)"}
        colour = "yellow" if verdict == "no-measurement" else "red"
        body = Text(res["message"], style=colour)
        if res.get("model_obb_mm"):
            body = body + Text(
                "\n\nmodel dimensions, longest first: "
                + " x ".join(f"{d:.3f}" for d in res["model_obb_mm"]) + " mm",
                style="dim")
        con.print(Panel(body, title=titles[verdict], border_style=colour, expand=False))
        return

    style = {"ok": "bold green", "warn": "bold yellow", "fail": "bold red"}[verdict]
    tbl = Table(box=ROUNDED, show_header=False, expand=False, pad_edge=False)
    tbl.add_column("k", style="cyan", no_wrap=True)
    tbl.add_column("v")
    tbl.add_row("measured (calipers)", f"{res['measured_mm']:.3f} mm")
    # ALWAYS state which physical dimension the reading was compared against:
    # a cross-check that quietly re-targets itself is not a cross-check.
    tbl.add_row("compared against",
                Text(f"{res.get('assumed_dimension', res['which_dimension'])}",
                     style="bold" if res.get("certifies_scale") else "bold yellow")
                + Text(f"   [{res['which_dimension']}]", style="dim"))
    tbl.add_row("model", f"{res['model_mm']:.3f} mm")
    tbl.add_row("discrepancy",
                Text(f"{res['delta_mm']:+.3f} mm  ({res['error_pct']:+.3f}%)", style=style))
    if not res.get("certifies_scale"):
        tbl.add_row("value as evidence", Text(
            "NONE. This dimension was auto-matched to the reading, so agreement is "
            "guaranteed by construction and cannot certify the scale. Re-run with "
            "--caliper-axis longest|middle|shortest to make this a real check.",
            style="bold yellow"))
    if verdict != "ok" and res.get("square_mm_correction"):
        corr = res["square_mm_correction"]
        line = f"IF the board is the cause: square_mm x {corr:.6f}"
        if spec is not None:
            line += f"  ->  {spec.square_mm * corr:.4f} mm (currently {spec.square_mm:g})"
        tbl.add_row("hypothesis to test", Text(line, style="yellow"))
        tbl.add_row("", Text(
            "This is a hypothesis, not an instruction. A whole-model proportional error "
            "is CONSISTENT WITH a mis-measured board square, but wrong intrinsics and a "
            "mesh truncated by the ground cut produce the same signature. Verify first: "
            "measure the PRINTED square with the calipers and compare it to square_mm, "
            "and check the reprojection witness above. Only re-run with --square-mm if "
            "the printed square really is that size -- typing this factor in without "
            "measuring just moves the error somewhere you cannot see it.",
            style="dim"))
    con.print(Panel(tbl, title=Text(f"caliper cross-check: {verdict.upper()}", style=style),
                    border_style=style.split()[-1], expand=False))


def _selftest() -> int:
    """Exercise both gates on synthetic inputs. `python -m scan2print.verify`."""
    con = Console()
    spec = BoardSpec()
    good = ScaleResult(
        scale=0.501578, rms_mm=0.114, n_cameras=48,
        R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 0],
        inlier_frac=1.0, quality="good",
        notes=["scale=0.501578 rms=0.114mm inliers=48/48 max_res=0.227mm "
               "spread=620.0mm det(R)=+1.000000"],
    )
    stats = {
        "bbox_mm": [50.0, 30.0, 20.0], "obb_mm": [50.0, 30.0, 20.0],
        "volume_cm3": 30.0, "watertight": True, "faces": 41234, "vertices": 20619,
        "fits_bed": True, "bed_mm": [220.0, 220.0, 220.0],
        "bed_margin_mm": [170.0, 190.0, 200.0],
    }
    con.rule("[bold]case 1: good")
    report(good, stats, spec, console=con)

    bad = ScaleResult(
        scale=0.4, rms_mm=6.2, n_cameras=7,
        R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 0],
        inlier_frac=0.31, quality="reject",
        notes=["scale=0.400000 rms=6.200mm inliers=7/23 max_res=41.0mm",
               "camera track is nearly PLANAR (minor/major = 0.011)",
               "only 7 cameras support the fit, below SCALE_MIN_CAMERAS=12"],
    )
    bad_stats = dict(stats, watertight=False, fits_bed=False,
                     bbox_mm=[260.0, 30.0, 20.0], obb_mm=[260.0, 30.0, 20.0],
                     bed_margin_mm=[-40.0, 190.0, 200.0], volume_cm3=None)
    con.rule("[bold]case 2: reject")
    report(bad, bad_stats, spec, console=con)

    con.rule("[bold]case 3: caliper gate")
    for reading in (50.02, 50.4, 52.0, None):
        r = caliper_gate(stats, reading, axis="longest")
        format_caliper_result(r, spec, console=con)
    r = caliper_gate(stats, 30.1, axis="y")
    format_caliper_result(r, spec, console=con)
    con.rule("[bold]case 4: caliper gate refuses to guess")
    format_caliper_result(caliper_gate(stats, 50.0), spec, console=con)
    oversize = {"obb_mm": [100.0, 50.0, 25.0], "bbox_mm": [100.0, 50.0, 25.0]}
    format_caliper_result(caliper_gate(oversize, 50.0, axis="longest"), spec, console=con)
    format_caliper_result(caliper_gate(oversize, 50.0, axis="nearest"), spec, console=con)

    # Assertions: the numbers, not just the rendering.
    assert caliper_gate(stats, 50.0, axis="longest")["verdict"] == "ok"
    assert caliper_gate(stats, 52.0, axis="longest")["verdict"] == "fail"
    assert abs(caliper_gate(stats, 52.0, axis="longest")["error_pct"] - (-3.846)) < 0.01
    assert caliper_gate(stats, None)["verdict"] == "no-measurement"
    # The defect this gate was rebuilt for: a model at 2x its true size, with a
    # 50 mm reading that happens to equal one of its extents. Auto-matching
    # called that "ok, 0.0000%".
    assert caliper_gate(oversize, 50.0)["verdict"] == "no-axis"
    assert caliper_gate(oversize, 50.0)["certifies_scale"] is False
    r2x = caliper_gate(oversize, 50.0, axis="longest")
    assert r2x["verdict"] == "fail" and abs(r2x["error_pct"] - 100.0) < 1e-6
    # Even the opt-in nearest path must never emit a board correction...
    rn = caliper_gate(oversize, 50.0, axis="nearest")
    assert rn["square_mm_correction"] is None and rn["certifies_scale"] is False
    # ...and must refuse outright when two extents are too close to tell apart.
    close = {"obb_mm": [30.0, 28.0, 10.0], "bbox_mm": [30.0, 28.0, 10.0]}
    assert caliper_gate(close, 29.0, axis="nearest")["verdict"] == "ambiguous"
    assert advice_for(bad, bad_stats)[0].startswith("Fit residual")

    # Witnesses must be able to overrule a 'good' solver grade.
    a_ok = assess(good, reproj_mean_px=0.09, reproj_max_px=0.4, frames_solved=40,
                  frames_total=40,
                  intrinsics=Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, rms_px=0.21,
                                        source="charuco-calibration"))
    assert a_ok["quality"] == "good", a_ok["downgrades"]
    a_bad = assess(good, reproj_mean_px=1.506, reproj_max_px=1.95, frames_solved=32,
                   frames_total=40,
                   intrinsics=Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0,
                                         source="objectcapture"))
    assert a_bad["quality"] == "reject", a_bad
    con.rule("[bold]case 5: a 'good' fit with bad intrinsics")
    report(good, stats, spec, console=con, witnesses=a_bad)
    con.print("[bold green]verify.py selftest OK[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
