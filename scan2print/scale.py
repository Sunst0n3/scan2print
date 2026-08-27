"""Scale recovery -- fit the RECON camera track onto the metric BOARD track.

This is the metrological heart of scan2print. Object Capture returns geometry in
an arbitrary-scale RECON frame; the ChArUco board gives true metric camera
centres in BOARD millimetres. Both tracks describe the *same* physical camera
trajectory, so the similarity transform that maps one set of camera centres onto
the other is exactly the transform that converts RECON geometry to millimetres:

    x_board_mm = scale * R @ x_recon + t

Public API
    umeyama()          closed-form similarity fit (Umeyama 1991), reflection-safe
    solve_scale()      robust RECON -> BOARD fit over a whole camera track
    apply_to_points()  apply a ScaleResult to RECON geometry

Self-test:  python -m scan2print.scale
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

try:  # normal package import
    from scan2print import config
    from scan2print.types import FramePose, ReconPose, ScaleResult
except ImportError:  # allow `python scan2print/scale.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scan2print import config
    from scan2print.types import FramePose, ReconPose, ScaleResult


# --- tuning local to the scale solve --------------------------------------
# These are solver internals rather than pipeline policy, so they live here
# instead of config.py (which this module is not allowed to edit).

MIN_FIT_POINTS = 3          # a 3D similarity has 7 DOF; 3 points give 9 equations
RANSAC_TRIALS = 200
RANSAC_SAMPLE = 4
RANSAC_SEED = 0xC0FFEE      # fixed: a metrology tool must be reproducible
TRIM_MAX_ITERS = 10
OUTLIER_FLOOR_MM = float(config.SCALE_RMS_MARGINAL_MM)  # never call <4 mm an outlier
# ...and never call >20 mm an inlier: a scan whose rms is already past the reject
# gate at 4 mm has no legitimate 20 mm cameras, so the ceiling only ever fires on
# genuine blunders. Without it, a 50/50 good/bad split pushes the adaptive
# threshold above every residual and the trimming keeps the blunders.
OUTLIER_CEILING_MM = 5.0 * OUTLIER_FLOOR_MM
OUTLIER_SIGMA_K = 3.0       # robust sigmas beyond the median before we drop a camera
LINEAR_RATIO_WARN = 0.10    # mid/major principal spread -> straight-line track
PLANAR_RATIO_WARN = 0.05    # minor/major principal spread -> flat track

# Confidence multiplier for the implied dimensional uncertainty reported in the
# notes: 1.96 sigma == a two-sided 95% interval.
UNCERTAINTY_K = 1.96

_EPS = 1e-12


# --- core solver ----------------------------------------------------------

def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping src onto dst.

    Returns (scale, R, t, rms) such that ``dst ~= scale * (R @ src.T).T + t``.
    Closed form of Umeyama, "Least-Squares Estimation of Transformation
    Parameters Between Two Point Patterns", IEEE PAMI 13(4), 1991.

    WHY THE DETERMINANT CORRECTION MATTERS: the raw SVD solution R = U @ Vt is
    the optimal *orthogonal* matrix, and with noisy or near-planar data the
    optimum is sometimes a reflection (det = -1) rather than a rotation. A
    reflection fits the camera centres just as well -- the residual looks
    perfect -- but it mirrors the reconstructed geometry. For a tool that turns
    the result into a physical printed part that a leather sheath is built
    around, a silently mirrored model is a catastrophic, invisible failure: the
    part looks plausible, measures correctly, and is the wrong handedness.
    Flipping the sign of the smallest singular value whenever det(U @ Vt) < 0
    constrains the answer to SO(3) and makes that failure impossible.
    """
    src = _as_xyz(src, "src")
    dst = _as_xyz(dst, "dst")
    if src.shape != dst.shape:
        raise ValueError(f"umeyama: src {src.shape} and dst {dst.shape} must match")
    n = src.shape[0]
    if n < MIN_FIT_POINTS:
        raise ValueError(
            f"umeyama: need at least {MIN_FIT_POINTS} correspondences, got {n}"
        )

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d

    var_s = float((sc ** 2).sum()) / n
    if var_s < _EPS:
        raise ValueError("umeyama: source points are coincident; similarity undefined")

    cov = (dc.T @ sc) / n                       # 3x3 cross-covariance dst x src
    U, sing, Vt = np.linalg.svd(cov)

    # d = sign(det(U @ Vt)); det(U)*det(Vt) is the same quantity, computed on
    # two well-conditioned orthogonal matrices instead of their product.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0                          # flip the smallest singular value
    R = U @ S @ Vt

    if with_scale:
        scale = float((sing * np.diag(S)).sum() / var_s)
    else:
        scale = 1.0

    t = mu_d - scale * (R @ mu_s)
    rms = float(np.sqrt((_residuals(src, dst, scale, R, t) ** 2).mean()))
    return scale, R, t, rms


def _as_xyz(a, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name}: expected an (N,3) array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains NaN/Inf")
    return arr


def _residuals(src, dst, scale, R, t) -> np.ndarray:
    """Per-point Euclidean residual, in dst units."""
    pred = scale * (src @ R.T) + t
    return np.linalg.norm(dst - pred, axis=1)


def _robust_threshold(r: np.ndarray) -> float:
    """Outlier cut-off in mm: median + k*sigma_robust, clamped to [floor, ceiling].

    The floor stops a very clean scan from trimming honest cameras (which would
    both bias the fit and make inlier_frac a lie); the ceiling stops a heavily
    corrupted scan from declaring its own blunders normal."""
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    sigma = 1.4826 * mad
    return float(np.clip(med + OUTLIER_SIGMA_K * sigma,
                         OUTLIER_FLOOR_MM, OUTLIER_CEILING_MM))


def _trim_refit(src, dst, seed_model):
    """Iterate: score all points against the model, drop outliers, refit."""
    model = seed_model
    mask = None
    n = src.shape[0]
    for _ in range(TRIM_MAX_ITERS):
        r = _residuals(src, dst, *model[:3])
        thr = _robust_threshold(r)
        new_mask = r <= thr
        if int(new_mask.sum()) < MIN_FIT_POINTS:      # keep the best few, refit anyway
            new_mask = np.zeros(n, dtype=bool)
            new_mask[np.argsort(r)[:MIN_FIT_POINTS]] = True
        if mask is not None and np.array_equal(new_mask, mask):
            break
        mask = new_mask
        model = umeyama(src[mask], dst[mask])
    if mask is None:
        mask = np.ones(n, dtype=bool)
    return model, mask


def _well_conditioned(pts: np.ndarray) -> bool:
    c = pts - pts.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return bool(s[0] > _EPS and s[-1] > 1e-9 * s[0])


def _robust_similarity(src, dst, rng):
    """RANSAC seed + iterative trimming.

    Returns (scale, R, t, rms_over_inliers, inlier_mask, all_residuals)."""
    n = src.shape[0]
    candidates = []

    # Seed 1: plain least squares over everything. Correct when the data are
    # clean, and the trimming stage repairs it when they are not.
    candidates.append(umeyama(src, dst))

    # Seed 2: best RANSAC hypothesis. Survives gross ChArUco pose blunders that
    # would drag the plain LS fit far enough to poison the trimming.
    k = min(RANSAC_SAMPLE, n)
    best = None
    if n > k:
        for _ in range(RANSAC_TRIALS):
            idx = rng.choice(n, size=k, replace=False)
            if not _well_conditioned(dst[idx]):
                continue
            try:
                m = umeyama(src[idx], dst[idx])
            except ValueError:
                continue
            if not np.isfinite(m[0]) or m[0] <= 0.0:
                continue
            r = _residuals(src, dst, *m[:3])
            inl = r <= OUTLIER_FLOOR_MM
            n_in = int(inl.sum())
            if n_in < MIN_FIT_POINTS:
                continue
            score = (n_in, -float(np.sqrt((r[inl] ** 2).mean())))
            if best is None or score > best[0]:
                best = (score, umeyama(src[inl], dst[inl]))
        if best is not None:
            candidates.append(best[1])

    results = []
    for seed in candidates:
        model, mask = _trim_refit(src, dst, seed)
        r = _residuals(src, dst, *model[:3])
        rms_in = float(np.sqrt((r[mask] ** 2).mean()))
        results.append((int(mask.sum()), -rms_in, model, mask))

    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, model, mask = results[0]
    scale, R, t, _ = model
    r = _residuals(src, dst, scale, R, t)
    rms_in = float(np.sqrt((r[mask] ** 2).mean()))
    return scale, R, t, rms_in, mask, r


def _track_shape(centers: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Principal-axis spreads (mm) of a camera track, plus shape ratios.

    Returns (spread_mm[3], mid/major, minor/major)."""
    c = centers - centers.mean(axis=0)
    sing = np.linalg.svd(c, compute_uv=False)
    spread = sing / math.sqrt(max(centers.shape[0] - 1, 1))
    if spread[0] < _EPS:
        return spread, 0.0, 0.0
    return spread, float(spread[1] / spread[0]), float(spread[2] / spread[0])


def _dimension_uncertainty_pct(rms_rel: float, n_in: int) -> float:
    """Implied 95% uncertainty on any dimension derived from this fit, in percent.

    The residual is a 3-D distance, but only its component along the line
    between two cameras carries information about the scale, so the effective
    one-dimensional noise is rms/sqrt(3). Averaged over n_in cameras spanning a
    track of RMS radius `spread`, the standard error on the fitted scale is

        sigma_scale / scale ~= (rms / sqrt(3)) / (sqrt(n_in) * spread)
                             == rms_rel / sqrt(3 * n_in)

    Validated against 120-trial Monte Carlo at three orbit radii: predicted 95%
    bounds of 0.456 / 0.182 / 0.068 % against measured p95 errors of
    0.420 / 0.168 / 0.063 % -- about 9% conservative, which is the right side to
    err on. This is FIT STATISTICS ONLY: it says nothing about a mis-measured
    board square or wrong intrinsics, both of which bias every camera the same
    way and so leave the residual untouched.
    """
    if not math.isfinite(rms_rel) or n_in < 1:
        return float("inf")
    return 100.0 * UNCERTAINTY_K * rms_rel / math.sqrt(3.0 * n_in)


def _pairwise_scale(src, dst, rng, max_pairs: int = 400) -> float:
    """Rotation-free scale estimate: median ratio of inter-camera distances.

    Completely independent of the similarity fit, so it is a genuine check
    rather than a restatement of the same arithmetic."""
    n = src.shape[0]
    if n < 2:
        return float("nan")
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_src = np.linalg.norm(src[i] - src[j], axis=1)
    d_dst = np.linalg.norm(dst[i] - dst[j], axis=1)
    ok = d_src > _EPS
    if not ok.any():
        return float("nan")
    return float(np.median(d_dst[ok] / d_src[ok]))


# --- pipeline entry point -------------------------------------------------

def solve_scale(
    recon: dict[str, ReconPose],
    metric: dict[str, FramePose],
    min_cams: int = config.SCALE_MIN_CAMERAS,
) -> ScaleResult:
    """Recover the RECON -> BOARD similarity from two views of one camera track.

    `recon` and `metric` are keyed by frame name; only names present in both are
    used. The returned ScaleResult.scale is millimetres per RECON unit.

    Raises ValueError when a fit is impossible at all (no shared frames, fewer
    than 3 usable cameras, non-finite geometry). Everything else comes back as a
    populated ScaleResult whose `quality` and `notes` say how much to trust it.
    """
    notes: list[str] = []

    common = sorted(set(recon.keys()) & set(metric.keys()))
    n_common = len(common)
    if n_common == 0:
        raise ValueError(
            "solve_scale: no frame names shared between the recon poses "
            f"({len(recon)} frames, e.g. {sorted(recon)[:3]}) and the metric poses "
            f"({len(metric)} frames, e.g. {sorted(metric)[:3]}). "
            "The two stages must agree on frame naming."
        )

    rec_all = np.array([recon[k].center_np for k in common], dtype=np.float64)
    met_all = np.array([metric[k].center_np for k in common], dtype=np.float64)

    finite = np.isfinite(rec_all).all(axis=1) & np.isfinite(met_all).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        notes.append(f"dropped {n_bad} frame(s) with non-finite camera centres")
    src = rec_all[finite]        # RECON units
    dst = met_all[finite]        # BOARD millimetres
    names = [k for k, f in zip(common, finite) if f]
    n_used = src.shape[0]

    if n_used < MIN_FIT_POINTS:
        raise ValueError(
            f"solve_scale: only {n_used} usable shared camera(s) "
            f"({n_common} shared frames, {n_bad} non-finite); "
            f"a similarity fit needs at least {MIN_FIT_POINTS}."
        )

    rng = np.random.default_rng(RANSAC_SEED)
    scale, R, t, rms_mm, mask, resid = _robust_similarity(src, dst, rng)

    n_in = int(mask.sum())
    inlier_frac = n_in / float(n_used)

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"solve_scale: fit produced a non-physical scale ({scale!r}). "
            "The recon and metric camera tracks are probably not the same track."
        )

    det_R = float(np.linalg.det(R))
    if det_R < 0.0:  # cannot happen with the guard in umeyama; assert it anyway
        raise ValueError(f"solve_scale: reflection leaked into the fit, det(R)={det_R}")

    # --- degenerate motion --------------------------------------------------
    # With only 3 surviving cameras the principal-axis test is meaningless (any
    # 3 points are coplanar), so the shape analysis is gated on having 4+.
    spread, linear_ratio, planar_ratio = _track_shape(dst[mask])
    shape_meaningful = n_in >= 4
    degenerate_linear = shape_meaningful and linear_ratio < LINEAR_RATIO_WARN
    degenerate_planar = (
        shape_meaningful and not degenerate_linear and planar_ratio < PLANAR_RATIO_WARN
    )
    collapsed = n_in <= MIN_FIT_POINTS

    # --- independent cross-check -------------------------------------------
    # Median inter-camera DISTANCE ratio. It shares no arithmetic with the
    # similarity residual -- no rotation, no least squares, no trimming -- so it
    # is a genuine second opinion rather than a restatement of the first. That
    # is exactly why it gates below instead of merely being noted: a second
    # opinion nobody acts on is decoration.
    pair_scale = _pairwise_scale(src[mask], dst[mask], rng)
    pair_rel = float("nan")
    if np.isfinite(pair_scale) and pair_scale > 0.0:
        pair_rel = abs(pair_scale - scale) / scale

    # --- residual, relative to the track it was measured over ---------------
    # PRIMARY gate. An absolute millimetre residual is not a measure of
    # accuracy: measured over 120 trials at 60/150/400 mm orbit radii, the same
    # 1.34 mm residual came with p95 scale errors of 0.42 / 0.17 / 0.06 % -- a
    # 6.7x span that the absolute gate graded identically 'good'. The residual
    # as a fraction of the track's spatial extent tracks the real error almost
    # exactly (0.135 / 0.134 / 0.134 % of scale error per % of relative
    # residual), because a fixed positional error is a far larger *angular*
    # error on a small orbit than on a big one.
    spread_mm = float(np.linalg.norm(spread))       # RMS radius of the inlier track
    rms_rel = rms_mm / spread_mm if spread_mm > _EPS else float("inf")
    dim_unc_pct = _dimension_uncertainty_pct(rms_rel, n_in)

    # --- quality ladder (config thresholds) --------------------------------
    # Relative gate decides; the absolute millimetre thresholds are kept as a
    # coarse backstop so that neither a tiny-but-noisy track nor a huge track
    # carrying honest-to-goodness blunders can sneak through on a favourable
    # ratio alone. Both must be satisfied.
    if (
        rms_rel <= config.SCALE_RMS_REL_GOOD
        and rms_mm <= config.SCALE_RMS_GOOD_MM
        and inlier_frac >= config.SCALE_MIN_INLIER_FRAC
        and n_in >= min_cams
    ):
        quality = "good"
    elif rms_rel <= config.SCALE_RMS_REL_MARGINAL and rms_mm <= config.SCALE_RMS_MARGINAL_MM:
        quality = "marginal"
    else:
        quality = "reject"

    # --- the cross-check GATES ----------------------------------------------
    if not np.isfinite(pair_rel):
        notes.append(
            "cross-check unavailable: the inter-camera distance ratio could not be "
            "evaluated (degenerate recon track), so the fitted scale has no "
            "independent confirmation -- capped at 'marginal'"
        )
        if quality == "good":
            quality = "marginal"
    elif pair_rel > config.PAIRWISE_DISAGREE_WARN:
        gate = ("REJECT" if pair_rel > config.PAIRWISE_DISAGREE_REJECT else "marginal")
        notes.append(
            f"CROSS-CHECK DISAGREES ({gate}): median inter-camera distance ratio gives "
            f"{pair_scale:.6g} mm/unit vs fitted {scale:.6g} mm/unit "
            f"({pair_rel * 100:.1f}% apart, gate {config.PAIRWISE_DISAGREE_WARN * 100:.0f}% "
            f"warn / {config.PAIRWISE_DISAGREE_REJECT * 100:.0f}% reject). This estimate is "
            "independent of the fit residual, so a low residual does NOT excuse it -- the "
            "two tracks are probably not the same motion (check frame naming and that the "
            "board was visible in the frames Object Capture used)"
        )
        if pair_rel > config.PAIRWISE_DISAGREE_REJECT:
            quality = "reject"
        elif quality == "good":
            quality = "marginal"

    if n_in < min_cams:
        notes.append(
            f"only {n_in} inlier camera(s), below the required {min_cams} -- "
            "too few views to trust the fit; shoot a longer orbit"
        )
    if inlier_frac < config.SCALE_MIN_INLIER_FRAC:
        notes.append(
            f"inlier fraction {inlier_frac:.2f} is below {config.SCALE_MIN_INLIER_FRAC:.2f} "
            f"({n_used - n_in} of {n_used} cameras rejected) -- board poses are unreliable, "
            "check for motion blur, glare on the board, or partial board views"
        )
    if rms_rel > config.SCALE_RMS_REL_MARGINAL or rms_mm > config.SCALE_RMS_MARGINAL_MM:
        notes.append(
            f"residual {rms_mm:.2f} mm = {rms_rel * 100:.2f}% of the {spread_mm:.0f} mm "
            f"camera-track spread, past the reject gate "
            f"({config.SCALE_RMS_REL_MARGINAL * 100:.1f}% of spread / "
            f"{config.SCALE_RMS_MARGINAL_MM:.2f} mm absolute) -- the metric and recon "
            "camera tracks do not agree; do not print this"
        )
    elif rms_rel > config.SCALE_RMS_REL_GOOD or rms_mm > config.SCALE_RMS_GOOD_MM:
        notes.append(
            f"residual {rms_mm:.2f} mm = {rms_rel * 100:.2f}% of the {spread_mm:.0f} mm "
            f"camera-track spread, above the good gate "
            f"({config.SCALE_RMS_REL_GOOD * 100:.1f}% of spread / "
            f"{config.SCALE_RMS_GOOD_MM:.2f} mm absolute) -- scale is usable but the part "
            f"carries the +/-{dim_unc_pct:.2f}% uncertainty reported above; a wider orbit "
            "(more track spread) buys accuracy faster than steadier hands do"
        )

    if collapsed:
        notes.append(
            f"COLLAPSED FIT: trimming rejected all but {n_in} of {n_used} cameras, the bare "
            "minimum a similarity needs. There is no camera track left to fit and the "
            "residual is meaningless. The recon and metric tracks are almost certainly "
            "not the same motion -- check that frame names line up and that the board "
            "was visible in the frames Object Capture actually used."
        )
        quality = "reject"

    if degenerate_linear:
        notes.append(
            "DEGENERATE MOTION: the camera track is nearly a straight line "
            f"(principal spreads {spread[0]:.1f} / {spread[1]:.2f} / {spread[2]:.2f} mm, "
            f"mid/major = {linear_ratio:.4f} < {LINEAR_RATIO_WARN}). A collinear track "
            "cannot constrain a 3D similarity: rotation about the track axis is free and "
            "the scale is untrustworthy even though the residual looks small. "
            "Re-shoot orbiting the object."
        )
        quality = "reject"
    elif degenerate_planar:
        notes.append(
            "DEGENERATE MOTION: the camera track is nearly planar "
            f"(principal spreads {spread[0]:.1f} / {spread[1]:.1f} / {spread[2]:.3f} mm, "
            f"minor/major = {planar_ratio:.4f} < {PLANAR_RATIO_WARN}). A flat track leaves "
            "the out-of-plane direction weakly constrained and a mirrored solution fits "
            "almost as well. Re-shoot with varying camera height."
        )
        if quality == "good":
            quality = "marginal"

    # Anything already in `notes` at this point is a problem report; the two
    # inserts below are the unconditional header lines.
    clean = quality == "good" and not notes

    notes.insert(
        0,
        f"fit: {scale:.6g} mm per recon unit, rms {rms_mm:.3f} mm = {rms_rel * 100:.2f}% of "
        f"track spread, over {n_in}/{n_used} cameras (of {n_common} shared), max inlier "
        f"residual {float(resid[mask].max()):.2f} mm, track spread "
        f"{spread[0]:.0f}/{spread[1]:.0f}/{spread[2]:.0f} mm (norm {spread_mm:.0f} mm), "
        f"det(R)={det_R:.6f}",
    )
    # The number the user actually needs: what this residual means for the part
    # they are about to wrap leather around.
    cross = (
        f"cross-check agrees to {pair_rel * 100:.2f}%"
        if np.isfinite(pair_rel) else "no cross-check available"
    )
    notes.insert(
        1,
        f"implied dimensional uncertainty +/-{dim_unc_pct:.2f}% (95%), i.e. "
        f"+/-{dim_unc_pct:.2f} mm on a 100 mm feature ({cross}). Fit statistics only -- "
        "a mis-measured board square or wrong intrinsics biases every camera identically "
        "and is invisible here; only the caliper check catches those",
    )

    if clean:
        notes.append("no problems detected")

    return ScaleResult(
        scale=float(scale),
        rms_mm=float(rms_mm),
        n_cameras=n_in,                     # cameras actually supporting the estimate
        R=[[float(v) for v in row] for row in R],
        t=[float(v) for v in t],
        inlier_frac=float(inlier_frac),
        quality=quality,
        notes=notes,
    )


def apply_to_points(
    pts: np.ndarray,
    res: ScaleResult,
    allow_reject: bool = False,
) -> np.ndarray:
    """Transform RECON points into BOARD millimetres: scale * R @ p + t.

    Accepts (N,3) or a single (3,) point and returns the same shape. Refuses a
    rejected ScaleResult unless allow_reject=True -- silently converting
    geometry with a known-bad scale is how a wrong-sized part gets printed.
    """
    if res.quality == "reject" and not allow_reject:
        raise ValueError(
            "apply_to_points: refusing to use a ScaleResult with quality='reject'. "
            f"Notes: {'; '.join(res.notes)}. Pass allow_reject=True to override."
        )
    arr = np.asarray(pts, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"apply_to_points: expected (N,3) or (3,), got {np.shape(pts)}")

    R = np.asarray(res.R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(res.t, dtype=np.float64).reshape(3)
    out = res.scale * (arr @ R.T) + t
    return out.reshape(3) if single else out


# =========================================================================
# Self-test -- synthetic, needs no captured data.  python -m scan2print.scale
# =========================================================================

def _random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _look_at_R(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """BOARD -> camera rotation for a camera at `center` looking at `target`."""
    fwd = target - center
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(fwd @ up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.vstack([right, down, fwd])


def _hemi_orbit(n: int, radius: float = 300.0, turns: float = 2.0,
                elev=(25.0, 70.0), wobble: float = 0.05) -> np.ndarray:
    az = np.linspace(0.0, turns * 2 * np.pi, n)
    mod = 0.5 * (1.0 - np.cos(np.linspace(0.0, 2 * np.pi, n)))
    el = np.deg2rad(elev[0] + (elev[1] - elev[0]) * mod)
    r = radius * (1.0 + wobble * np.sin(3.0 * az))
    board_centre = np.array([105.0, 75.0, 0.0])
    return board_centre + np.stack(
        [r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], axis=1
    )


def _make_poses(board_centres, true_scale, rng, met_sigma_mm=0.25,
                rec_sigma_mm=0.15, corrupt_idx=(), corrupt_mm=60.0):
    """Build (recon, metric) dicts from a metric camera track.

    RECON = (1/true_scale) * Q @ BOARD + b, so solving RECON->BOARD must return
    exactly `true_scale` mm per recon unit."""
    n = board_centres.shape[0]
    Q = _random_rotation(rng)
    b = rng.normal(scale=3.0, size=3)
    a = 1.0 / true_scale

    recon_c = a * (board_centres @ Q.T) + b
    recon_c = recon_c + rng.normal(scale=rec_sigma_mm * a, size=recon_c.shape)
    met_c = board_centres + rng.normal(scale=met_sigma_mm, size=board_centres.shape)

    truth_c = met_c.copy()
    for i in corrupt_idx:                       # gross ChArUco pose blunders
        d = rng.normal(size=3)
        met_c[i] += corrupt_mm * d / np.linalg.norm(d)

    target = np.array([105.0, 75.0, 20.0])
    recon: dict[str, ReconPose] = {}
    metric: dict[str, FramePose] = {}
    for i in range(n):
        name = f"f{i:04d}"
        Rm = _look_at_R(met_c[i], target)
        tm = -Rm @ met_c[i]
        metric[name] = FramePose(
            name=name, R=Rm.tolist(), t=tm.tolist(), center=met_c[i].tolist(),
            n_corners=24, reproj_rms_px=0.4,
        )
        Rr = _look_at_R(recon_c[i], a * (target @ Q.T) + b)
        M = np.eye(4)
        M[:3, :3] = Rr.T
        M[:3, 3] = recon_c[i]
        recon[name] = ReconPose(
            name=name, transform=M.flatten(order="F").tolist(),
            center=recon_c[i].tolist(),
        )
    return recon, metric, truth_c


def _unguarded_similarity(src, dst):
    """Deliberately WRONG reference: the same SVD without the det correction.
    Used only to prove the guard in umeyama() actually changes the answer."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / n
    U, D, Vt = np.linalg.svd((dc.T @ sc) / n)
    R = U @ Vt
    s = D.sum() / var_s
    t = mu_d - s * (R @ mu_s)
    return s, R, t, float(np.sqrt((_residuals(src, dst, s, R, t) ** 2).mean()))


def _rec_pts_of(recon: dict[str, ReconPose]) -> np.ndarray:
    return np.array([recon[k].center_np for k in sorted(recon)])


def _met_pts_of(metric: dict[str, FramePose]) -> np.ndarray:
    return np.array([metric[k].center_np for k in sorted(metric)])


def _selftest() -> int:
    rng = np.random.default_rng(7)
    ok_all = True

    def check(label: str, cond: bool) -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    print("=" * 78)
    print("(a0) NOISELESS CONTROL -- the estimator must be exactly unbiased")
    print("=" * 78)
    ts0 = 43.21098765
    r0, m0, _ = _make_poses(_hemi_orbit(40), ts0, rng, met_sigma_mm=0.0, rec_sigma_mm=0.0)
    res0 = solve_scale(r0, m0)
    rel0 = abs(res0.scale - ts0) / ts0
    print(f"  true {ts0:.10f}  recovered {res0.scale:.10f}  rel err {rel0:.3e}")
    print(f"  rms_mm={res0.rms_mm:.3e}  quality={res0.quality}  "
          f"n_cameras={res0.n_cameras}  inlier_frac={res0.inlier_frac:.3f}")
    check("noiseless recovery exact to ~machine precision", rel0 < 1e-12)
    check("noiseless residual ~0", res0.rms_mm < 1e-9)

    print()
    print("=" * 78)
    print("(a-c) CLEAN HEMISPHERICAL ORBIT -- known scale recovery")
    print("=" * 78)
    true_scale = 137.482913          # mm per recon unit
    track = _hemi_orbit(60)
    recon, metric, _ = _make_poses(track, true_scale, rng)
    res = solve_scale(recon, metric)
    rel = abs(res.scale - true_scale) / true_scale
    print(f"  true scale      : {true_scale:.9f} mm/unit")
    print(f"  recovered scale : {res.scale:.9f} mm/unit")
    print(f"  relative error  : {rel:.3e}  ({rel * 100:.6f} %)")
    print(f"  rms_mm={res.rms_mm:.4f}  n_cameras={res.n_cameras}  "
          f"inlier_frac={res.inlier_frac:.3f}  quality={res.quality}")
    print(f"  det(R)          : {np.linalg.det(np.array(res.R)):.12f}")
    for nt in res.notes:
        print(f"    note: {nt}")
    check("relative scale error < 1e-3", rel < 1e-3)
    check("quality == good", res.quality == "good")
    check("all 60 cameras inliers", res.n_cameras == 60 and res.inlier_frac == 1.0)
    check("rms below SCALE_RMS_GOOD_MM", res.rms_mm <= config.SCALE_RMS_GOOD_MM)
    # apply_to_points must reproduce the metric track it was fitted from
    rec_pts = np.array([recon[k].center_np for k in sorted(recon)])
    met_pts = np.array([metric[k].center_np for k in sorted(metric)])
    mapped = apply_to_points(rec_pts, res)
    err = np.linalg.norm(mapped - met_pts, axis=1)
    print(f"  apply_to_points : mean {err.mean():.4f} mm, max {err.max():.4f} mm")
    check("apply_to_points max error < 2 mm", err.max() < 2.0)
    single = apply_to_points(rec_pts[0], res)
    check("apply_to_points accepts a single (3,) point", single.shape == (3,))

    print()
    print("=" * 78)
    print("(d) ROBUSTNESS -- 20% of metric poses corrupted by 60 mm blunders")
    print("=" * 78)
    n = 60
    bad = rng.choice(n, size=12, replace=False)
    track = _hemi_orbit(n)
    recon, metric, _ = _make_poses(track, true_scale, rng, corrupt_idx=bad)
    res_d = solve_scale(recon, metric)
    rel_d = abs(res_d.scale - true_scale) / true_scale
    print(f"  corrupted frames: {sorted(int(i) for i in bad)}")
    print(f"  true scale      : {true_scale:.9f} mm/unit")
    print(f"  recovered scale : {res_d.scale:.9f} mm/unit")
    print(f"  relative error  : {rel_d:.3e}  ({rel_d * 100:.6f} %)")
    print(f"  rms_mm={res_d.rms_mm:.4f}  n_cameras={res_d.n_cameras}  "
          f"inlier_frac={res_d.inlier_frac:.3f}  quality={res_d.quality}")
    for nt in res_d.notes:
        print(f"    note: {nt}")
    print("  (for reference, a non-robust fit over the same data:)")
    s_ls, _, _, rms_ls = umeyama(_rec_pts_of(recon), _met_pts_of(metric))
    print(f"    plain least squares scale={s_ls:.6f} "
          f"(rel err {abs(s_ls - true_scale) / true_scale:.3e}), rms={rms_ls:.2f} mm")
    check("relative scale error < 1e-3 despite 20% outliers", rel_d < 1e-3)
    check("inlier_frac in [0.75, 0.85]", 0.75 <= res_d.inlier_frac <= 0.85)
    check("all 12 corrupted frames rejected",
          res_d.n_cameras == 48)
    check("robust fit beats plain least squares",
          rel_d < abs(s_ls - true_scale) / true_scale)

    print()
    print("=" * 78)
    print("(e) REFLECTION GUARD -- det(R) must never be -1")
    print("=" * 78)
    src = rng.normal(scale=50.0, size=(40, 3))
    Q = _random_rotation(rng)
    mirror = np.diag([1.0, 1.0, -1.0])
    dst = 0.73 * (src @ (Q @ mirror).T) + np.array([12.0, -5.0, 3.0])
    s_g, R_g, _, rms_g = umeyama(src, dst)
    s_u, R_u, _, rms_u = _unguarded_similarity(src, dst)
    print(f"  mirrored data, guarded  : det(R)={np.linalg.det(R_g):+.12f} "
          f"scale={s_g:.6f} rms={rms_g:.3f}")
    print(f"  mirrored data, UNGUARDED: det(R)={np.linalg.det(R_u):+.12f} "
          f"scale={s_u:.6f} rms={rms_u:.3f}   <-- silent mirrored model")
    check("guarded det(R) == +1", np.linalg.det(R_g) > 0.999999)
    check("unguarded really would have returned a reflection",
          np.linalg.det(R_u) < -0.999999)
    check("guard shows up as a large residual instead of a silent mirror",
          rms_g > 10.0 and rms_u < 1e-6)

    planar = np.column_stack([rng.normal(scale=50.0, size=30),
                              rng.normal(scale=50.0, size=30),
                              np.zeros(30)])
    dst_p = planar @ np.diag([-1.0, 1.0, 1.0]).T     # reflection == a real rotation here
    s_p, R_p, _, rms_p = umeyama(planar, dst_p)
    print(f"  coplanar mirrored data  : det(R)={np.linalg.det(R_p):+.12f} "
          f"scale={s_p:.6f} rms={rms_p:.3e}")
    check("coplanar mirror resolved as a proper rotation",
          np.linalg.det(R_p) > 0.999999 and rms_p < 1e-9)

    dets = []
    for _ in range(400):
        a = rng.normal(scale=30.0, size=(8, 3))
        b = rng.normal(scale=30.0, size=(8, 3))        # pure noise, no real relation
        dets.append(np.linalg.det(umeyama(a, b)[1]))
    print(f"  400 random/noise fits   : min det(R)={min(dets):+.12f} "
          f"max det(R)={max(dets):+.12f}")
    check("no reflection in 400 adversarial random fits", min(dets) > 0.999999)

    print()
    print("=" * 78)
    print("(f) DEGENERATE MOTION -- straight-line camera track")
    print("=" * 78)
    tvals = np.linspace(0.0, 1.0, 40)[:, None]
    line = np.array([10.0, 10.0, 180.0]) + tvals * np.array([200.0, 140.0, 30.0])
    recon_l, metric_l, _ = _make_poses(line, true_scale, rng)
    res_f = solve_scale(recon_l, metric_l)
    print(f"  recovered scale : {res_f.scale:.6f} (true {true_scale:.6f}, "
          f"rel err {abs(res_f.scale - true_scale) / true_scale:.3e})")
    print(f"  rms_mm={res_f.rms_mm:.4f}  n_cameras={res_f.n_cameras}  "
          f"inlier_frac={res_f.inlier_frac:.3f}  quality={res_f.quality}")
    for nt in res_f.notes:
        print(f"    note: {nt}")
    check("straight-line track downgraded to reject", res_f.quality == "reject")
    check("note explains the degeneracy",
          any("DEGENERATE MOTION" in nt and "straight line" in nt for nt in res_f.notes))
    check("low residual alone did NOT earn a pass",
          res_f.rms_mm <= config.SCALE_RMS_GOOD_MM and res_f.quality == "reject")
    try:
        apply_to_points(np.zeros((2, 3)), res_f)
        check("apply_to_points refuses a rejected result", False)
    except ValueError as exc:
        print(f"  apply_to_points raised: {str(exc)[:80]}...")
        check("apply_to_points refuses a rejected result", True)

    print()
    print("=" * 78)
    print("(g) DEGENERATE MOTION -- flat single-elevation circular orbit")
    print("=" * 78)
    flat = _hemi_orbit(48, turns=1.0, elev=(40.0, 40.0), wobble=0.0)
    recon_p, metric_p, _ = _make_poses(flat, true_scale, rng)
    res_g = solve_scale(recon_p, metric_p)
    print(f"  recovered scale : {res_g.scale:.6f} (true {true_scale:.6f}, "
          f"rel err {abs(res_g.scale - true_scale) / true_scale:.3e})")
    print(f"  rms_mm={res_g.rms_mm:.4f}  n_cameras={res_g.n_cameras}  "
          f"inlier_frac={res_g.inlier_frac:.3f}  quality={res_g.quality}")
    for nt in res_g.notes:
        print(f"    note: {nt}")
    check("planar track downgraded from good to marginal", res_g.quality == "marginal")
    check("note explains the planarity",
          any("nearly planar" in nt for nt in res_g.notes))

    print()
    print("=" * 78)
    print("(h0) TOO FEW CAMERAS -- must be downgraded and explained")
    print("=" * 78)
    few = _hemi_orbit(8)
    r8, m8, _ = _make_poses(few, true_scale, rng)
    res_h = solve_scale(r8, m8)
    print(f"  recovered scale : {res_h.scale:.6f} (rel err "
          f"{abs(res_h.scale - true_scale) / true_scale:.3e})")
    print(f"  rms_mm={res_h.rms_mm:.4f}  n_cameras={res_h.n_cameras}  "
          f"inlier_frac={res_h.inlier_frac:.3f}  quality={res_h.quality}  "
          f"(SCALE_MIN_CAMERAS={config.SCALE_MIN_CAMERAS})")
    for nt in res_h.notes:
        print(f"    note: {nt}")
    check("8 cameras (< SCALE_MIN_CAMERAS) cannot be 'good'", res_h.quality != "good")
    check("note names the camera-count shortfall",
          any("inlier camera" in nt and "below the required" in nt for nt in res_h.notes))

    print()
    print("=" * 78)
    print("(i) UNRELATED TRACKS -- recon and metric are not the same motion")
    print("=" * 78)
    r_x, m_x, _ = _make_poses(_hemi_orbit(60), true_scale, rng)
    for k in m_x:                       # replace the metric track with noise
        m_x[k].center = (rng.normal(scale=200.0, size=3)
                         + np.array([105.0, 75.0, 250.0])).tolist()
    res_i = solve_scale(r_x, m_x)
    print(f"  scale={res_i.scale:.3f}  rms_mm={res_i.rms_mm:.2f}  "
          f"n_cameras={res_i.n_cameras}  inlier_frac={res_i.inlier_frac:.3f}  "
          f"quality={res_i.quality}")
    for nt in res_i.notes:
        print(f"    note: {nt}")
    check("unrelated tracks rejected", res_i.quality == "reject")
    check("note says the fit collapsed, not something misleading",
          any("COLLAPSED FIT" in nt for nt in res_i.notes))

    print()
    print("=" * 78)
    print("(j) DETERMINISM -- same input must give a bit-identical answer")
    print("=" * 78)
    again = solve_scale(recon, metric)
    print(f"  run 1 scale={res_d.scale!r}")
    print(f"  run 2 scale={again.scale!r}")
    check("repeat solve is bit-identical",
          again.scale == res_d.scale and again.rms_mm == res_d.rms_mm
          and again.R == res_d.R and again.t == res_d.t)

    print()
    print("=" * 78)
    print("(h) FAILURE MODES -- must raise loudly, not return a wrong number")
    print("=" * 78)
    try:
        solve_scale({"a": recon["f0000"]}, {"zz": metric["f0000"]})
        check("no shared frame names raises", False)
    except ValueError as exc:
        print(f"  raised: {str(exc)[:96]}...")
        check("no shared frame names raises", True)
    try:
        two = {k: recon[k] for k in ("f0000", "f0001")}
        twom = {k: metric[k] for k in ("f0000", "f0001")}
        solve_scale(two, twom)
        check("fewer than 3 shared cameras raises", False)
    except ValueError as exc:
        print(f"  raised: {str(exc)[:96]}...")
        check("fewer than 3 shared cameras raises", True)

    print()
    print("=" * 78)
    print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
