"""End-to-end metric accuracy test against synthetic ground truth.

Renders a Blender scene in which the object's dimensions, the board's size, and
every camera pose are known exactly, then pushes it through as much of the real
pipeline as currently exists and checks the recovered millimetres against truth.

WHAT THIS TEST ACTUALLY MEASURES
--------------------------------
The pipeline's metric chain has two independent halves:

  A. BOARD METROLOGY -- detect the ChArUco board, solvePnP, get a metric camera
     track in millimetres. This test exercises that half for real, on actual
     rendered pixels, including the H.264 round trip.

  B. RECONSTRUCTION -- Object Capture turns the frames into a mesh plus an
     arbitrary-scale camera track. This test does NOT run Object Capture. It
     substitutes an *ideal* recon track: the ground-truth cameras pushed through
     a known arbitrary similarity transform. Optional jitter sweeps show how
     recon error propagates, but they are reported, not asserted.

So a pass here means: the frame conventions, intrinsics, pose recovery, and
similarity fit are arithmetically correct, and board metrology contributes only
a tiny fraction of a millimetre. It does NOT mean the pipeline achieves this on
a phone video of a real object -- see the caveats in scan2print/synth.py. In
particular a real printed board is scaled by the printer by a few tenths of a
percent, and that error passes straight through everything here undetected.

Run:  ./.venv/bin/python tests/test_pipeline.py
Env:  S2P_TEST_FRAMES (default 30), S2P_TEST_RES (default 1440x1080),
      S2P_TEST_DIR, S2P_TEST_FORCE=1 to force a re-render.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scan2print import config  # noqa: E402
from scan2print.synth import make_scene  # noqa: E402
from scan2print.types import BoardSpec, FramePose, ReconPose  # noqa: E402

# --- tolerances -----------------------------------------------------------
# Calibrated by deliberately corrupting inputs and checking each assertion
# fires. At 60 frames / 1920x1440 the harness measures mean centre error
# 0.04mm (PNG) / 0.05mm (H.264), so these leave ~5x headroom while still
# catching: fx wrong by 1% (5.9mm), square_mm wrong by 0.5mm (6.7mm), a board
# Y-axis sign flip (327mm), and cx wrong by 20px (0.42mm).
#
# Worth knowing: a principal-point error does NOT meaningfully corrupt the
# recovered scale (20px of cx still gave 0.002% scale error), because the
# similarity fit absorbs a consistent lateral shift of the whole camera track.
# Scale is sensitive to fx and to square_mm, not to cx/cy. Hence the separate,
# tighter pose assertion -- the scale check alone would not catch it.
TOL_MEAN_CENTRE_MM = 0.25
TOL_MAX_CENTRE_MM = 3.00
TOL_MEAN_REPROJ_PX = 0.50
TOL_SCALE_REL = 1e-3          # 0.1 %
TOL_DIM_MM = 0.25             # per-axis absolute error on the recovered bbox
MIN_FRAME_DETECT_FRAC = 0.95

# The arbitrary scale a reconstruction comes out at. Deliberately not near 1.0
# so any accidental "scale is already millimetres" shortcut fails loudly.
RECON_MM_PER_UNIT = 0.0137


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
class Fail(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform: dst ~= s * R @ src + t.

    Umeyama 1991. Used as the reference implementation the pipeline's own
    scale solver is checked against.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N,3) arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 correspondences, got {n}")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, sv, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (sc ** 2).sum() / n
    if var_src <= 0:
        raise ValueError("degenerate source track: all points coincide")
    scale = float(np.trace(np.diag(sv) @ S) / var_src)
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def optional_attr(module_name: str, attr: str) -> Callable | None:
    """Import a sibling pipeline module if it exists yet, else None."""
    try:
        mod = __import__(module_name, fromlist=[attr])
    except ImportError:
        return None
    return getattr(mod, attr, None)


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------
def build_scene() -> dict[str, Any]:
    n_frames = int(os.environ.get("S2P_TEST_FRAMES", "30"))
    res = os.environ.get("S2P_TEST_RES", "1440x1080")
    width, height = (int(v) for v in res.lower().split("x"))
    out_dir = Path(os.environ.get("S2P_TEST_DIR", ROOT / "out" / "synth_test"))
    force = os.environ.get("S2P_TEST_FORCE", "") == "1"

    print(f"[scene] {n_frames} frames at {width}x{height} -> {out_dir}")
    t0 = time.time()
    scene = make_scene(
        out_dir,
        object_mm=(50.0, 30.0, 20.0),
        n_frames=n_frames,
        resolution=(width, height),
        force=force,
    )
    verb = "reused" if scene["reused"] else "rendered"
    print(f"[scene] {verb} in {time.time() - t0:.1f}s  timing={scene['timing']}")
    return scene


def detect_track(truth: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Detect the board and solvePnP every frame; return per-frame results.

    ``source`` is "png" (rendered frames) or "video" (H.264 decode), so the test
    can show what the compression round trip costs.
    """
    bs = BoardSpec.from_json(truth["board"])
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bs.dict_name))
    board = cv2.aruco.CharucoBoard(
        (bs.squares_x, bs.squares_y), bs.square_mm, bs.marker_mm, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)

    cam = truth["camera"]
    K = np.array(
        [[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.array(cam["dist"], dtype=np.float64).reshape(-1, 1)

    if source == "video":
        frames: list[np.ndarray] = []
        cap = cv2.VideoCapture(truth["video"]["path"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        check(
            len(frames) == len(truth["poses"]),
            f"video decoded {len(frames)} frames but ground truth has "
            f"{len(truth['poses'])}",
        )
        get = lambda p: frames[p["index"]]  # noqa: E731
    else:
        get = lambda p: cv2.imread(p["file"], cv2.IMREAD_GRAYSCALE)  # noqa: E731

    out = []
    for pose in truth["poses"]:
        img = get(pose)
        check(img is not None, f"could not read frame {pose['name']}")
        ch_corners, ch_ids, _mk, mk_ids = detector.detectBoard(img)
        n_corners = 0 if ch_corners is None else int(len(ch_corners))
        rec: dict[str, Any] = {
            "name": pose["name"],
            "index": pose["index"],
            "n_corners": n_corners,
            "n_markers": 0 if mk_ids is None else int(len(mk_ids)),
            "gt_center": np.array(pose["center"], dtype=np.float64),
            # Object Capture reconstructs the PHYSICAL scene -- it never sees the
            # board -- so a mis-scaled board does not move its cameras. With
            # board_scale == 1 these two are identical.
            "gt_center_physical": np.array(
                pose.get("center_physical", pose["center"]), dtype=np.float64
            ),
            "gt_R": np.array(pose["R"], dtype=np.float64),
            "solved": False,
        }
        if n_corners >= config.MIN_CHARUCO_CORNERS:
            objp, imgp = board.matchImagePoints(ch_corners, ch_ids)
            ok, rvec, tvec = cv2.solvePnP(
                objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
                rms = float(
                    np.sqrt(
                        ((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2)
                        .sum(axis=1)
                        .mean()
                    )
                )
                rec.update(
                    solved=True,
                    R=R,
                    t=tvec.ravel(),
                    center=(-R.T @ tvec).ravel(),
                    reproj_rms_px=rms,
                )
        out.append(rec)
    return out


def synth_recon_track(
    truth: dict[str, Any], solved: list[dict[str, Any]], seed: int = 0
) -> tuple[list[ReconPose], float, np.ndarray, np.ndarray]:
    """Build an IDEAL arbitrary-scale reconstruction track from ground truth.

    Models what Object Capture would emit if it were perfect: the true PHYSICAL
    camera geometry expressed in a frame related to physical millimetres by an
    unknown similarity

        x_physical = s * Ra @ x_recon + ta

    Built from the physical centres, not the nominal board-frame ones, because
    the reconstruction never observes the board. When the printed board is
    mis-scaled the two tracks disagree by exactly that factor, which is what
    turns into a reported-size error.

    Only the frames that actually solved are included, so the two tracks line up.
    """
    rng = np.random.default_rng(seed)
    Ra, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Ra) < 0:
        Ra[:, 0] *= -1.0
    ta = rng.normal(scale=50.0, size=3)
    s = RECON_MM_PER_UNIT

    poses = []
    for rec in solved:
        c_recon = (Ra.T @ (rec["gt_center_physical"] - ta)) / s
        # camera-to-board rotation is gt_R.T; strip the frame change for recon
        C_board = rec["gt_R"].T
        C_recon = Ra.T @ C_board
        M = np.eye(4)
        M[:3, :3] = C_recon
        M[:3, 3] = c_recon
        poses.append(
            ReconPose(
                name=rec["name"],
                transform=list(M.flatten(order="F")),  # column-major
                center=list(c_recon),
            )
        )
    return poses, s, Ra, ta


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_board_detects(truth: dict[str, Any], track: list[dict[str, Any]], label: str) -> None:
    counts = np.array([r["n_corners"] for r in track])
    n_ok = int((counts >= config.MIN_CHARUCO_CORNERS).sum())
    frac = n_ok / len(counts)
    max_possible = BoardSpec.from_json(truth["board"]).n_corners
    print(
        f"[detect:{label}] corners min={counts.min()} mean={counts.mean():.1f} "
        f"max={counts.max()} of {max_possible} possible | "
        f"frames with >={config.MIN_CHARUCO_CORNERS}: {n_ok}/{len(counts)}"
    )
    check(
        frac >= MIN_FRAME_DETECT_FRAC,
        f"only {frac:.0%} of {label} frames reached "
        f"{config.MIN_CHARUCO_CORNERS} ChArUco corners "
        f"(need {MIN_FRAME_DETECT_FRAC:.0%}). The synthetic board is not "
        f"detectable, so the harness proves nothing.",
    )


def test_pose_recovery(track: list[dict[str, Any]], label: str) -> np.ndarray:
    solved = [r for r in track if r["solved"]]
    check(len(solved) >= config.SCALE_MIN_CAMERAS,
          f"only {len(solved)} frames solved, need {config.SCALE_MIN_CAMERAS}")
    errs = np.array([np.linalg.norm(r["center"] - r["gt_center"]) for r in solved])
    rms = np.array([r["reproj_rms_px"] for r in solved])
    print(
        f"[pose:{label}]   centre err mm mean={errs.mean():.4f} "
        f"med={np.median(errs):.4f} max={errs.max():.4f} | "
        f"reproj px mean={rms.mean():.4f} max={rms.max():.4f}"
    )
    check(
        errs.mean() < TOL_MEAN_CENTRE_MM,
        f"mean camera-centre error {errs.mean():.3f}mm exceeds "
        f"{TOL_MEAN_CENTRE_MM}mm -- pose recovery or the frame convention is wrong",
    )
    check(
        errs.max() < TOL_MAX_CENTRE_MM,
        f"worst camera-centre error {errs.max():.3f}mm exceeds {TOL_MAX_CENTRE_MM}mm",
    )
    check(
        rms.mean() < TOL_MEAN_REPROJ_PX,
        f"mean reprojection {rms.mean():.3f}px exceeds {TOL_MEAN_REPROJ_PX}px -- "
        f"intrinsics do not match the renderer",
    )
    return errs


def test_scale_recovery(truth: dict[str, Any], track: list[dict[str, Any]]) -> dict[str, Any]:
    """The money shot: recover true millimetres from an unknown-scale recon."""
    solved = [r for r in track if r["solved"]]
    recon_poses, s_true, _Ra, _ta = synth_recon_track(truth, solved)

    metric = np.array([r["center"] for r in solved])
    recon = np.array([p.center for p in recon_poses])

    s_hat, R_hat, t_hat = umeyama(recon, metric)
    resid = metric - (s_hat * (recon @ R_hat.T) + t_hat)
    fit_rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    rel = abs(s_hat - s_true) / s_true

    true_dims = np.array(truth["object"]["true_dims_mm"], dtype=np.float64)
    recon_dims = true_dims / s_true          # what the recon mesh measures
    recovered = recon_dims * s_hat           # what the pipeline would report
    dim_err = np.abs(recovered - true_dims)

    print(
        f"[scale]          s_true={s_true:.9f} s_hat={s_hat:.9f} "
        f"rel_err={rel * 100:.5f}%  fit_rms={fit_rms:.4f}mm  "
        f"cameras={len(solved)}"
    )
    print(
        f"[dims]           true={np.round(true_dims, 4).tolist()} "
        f"recovered={np.round(recovered, 4).tolist()} "
        f"max_err={dim_err.max():.4f}mm"
    )

    check(
        rel < TOL_SCALE_REL,
        f"scale relative error {rel * 100:.4f}% exceeds {TOL_SCALE_REL * 100:.3f}%",
    )
    check(
        fit_rms < config.SCALE_RMS_GOOD_MM,
        f"similarity fit residual {fit_rms:.3f}mm exceeds the "
        f"'good' gate {config.SCALE_RMS_GOOD_MM}mm",
    )
    check(
        dim_err.max() < TOL_DIM_MM,
        f"recovered dimension error {dim_err.max():.4f}mm exceeds {TOL_DIM_MM}mm",
    )

    # Sensitivity: reported, not asserted. Shows how recon-track error (which
    # this harness does NOT otherwise model) propagates into reported size.
    print("[sensitivity]    recon jitter -> scale error -> worst dimension error")
    rng = np.random.default_rng(12345)
    for jitter_mm in (0.2, 0.5, 1.0, 2.0):
        noisy = recon + rng.normal(scale=jitter_mm / s_true, size=recon.shape)
        s_n, _, _ = umeyama(noisy, metric)
        d_n = np.abs(recon_dims * s_n - true_dims)
        print(
            f"                 {jitter_mm:>4.1f}mm      "
            f"{abs(s_n - s_true) / s_true * 100:>8.5f}%     {d_n.max():.4f}mm"
        )

    return {"scale": s_hat, "rel": rel, "fit_rms": fit_rms, "dim_err": dim_err}


def test_pipeline_scale_module(truth: dict[str, Any], track: list[dict[str, Any]]) -> None:
    """Use scan2print.scale.solve_scale if that module has landed yet.

    Built on the documented FramePose/ReconPose contract in types.py. If the
    real signature differs, the mismatch is reported rather than failing the
    run, since this module is owned by another agent.
    """
    solve_scale = optional_attr("scan2print.scale", "solve_scale")
    if solve_scale is None:
        print("[scale-module]   scan2print.scale.solve_scale not present yet -- "
              "skipped (this test uses its own Umeyama reference fit)")
        return

    solved = [r for r in track if r["solved"]]
    recon_poses, s_true, _, _ = synth_recon_track(truth, solved)
    frame_poses = [
        FramePose(
            name=r["name"],
            R=r["R"].tolist(),
            t=list(r["t"]),
            center=list(r["center"]),
            n_corners=r["n_corners"],
            reproj_rms_px=r["reproj_rms_px"],
        )
        for r in solved
    ]
    # scale.solve_scale(recon: dict[str, ReconPose], metric: dict[str, FramePose])
    recon_by_name = {p.name: p for p in recon_poses}
    metric_by_name = {p.name: p for p in frame_poses}
    try:
        result = solve_scale(recon_by_name, metric_by_name)
    except Exception as exc:  # noqa: BLE001 - other agent's signature
        print(
            f"[scale-module]   solve_scale(dict[str, ReconPose], dict[str, FramePose]) "
            f"raised {type(exc).__name__}: {exc}\n"
            f"                 -> signature changed; update this adapter"
        )
        return
    rel = abs(result.scale - s_true) / s_true
    print(
        f"[scale-module]   solve_scale -> scale={result.scale:.9f} "
        f"rel_err={rel * 100:.5f}% rms={result.rms_mm:.4f}mm quality={result.quality}"
    )
    check(
        rel < TOL_SCALE_REL,
        f"scan2print.scale.solve_scale relative error {rel * 100:.4f}% exceeds "
        f"{TOL_SCALE_REL * 100:.3f}%",
    )


def recover_dims(truth: dict[str, Any], track: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    """Run the full metric chain and return (recovered_dims_mm, scale_rel_err)."""
    solved = [r for r in track if r["solved"]]
    recon_poses, s_true, _, _ = synth_recon_track(truth, solved)
    metric = np.array([r["center"] for r in solved])
    recon = np.array([p.center for p in recon_poses])
    s_hat, _, _ = umeyama(recon, metric)
    recon_dims = np.array(truth["object"]["true_dims_mm"], dtype=np.float64) / s_true
    return recon_dims * s_hat, abs(s_hat - s_true) / s_true


def test_printer_scale_error() -> None:
    """Quantify what a board printed at the wrong scale costs.

    This is the dominant real-world error source and the one the perfect-board
    scenes above cannot see. scan2print.charuco.render_target quantises the
    square to a whole number of pixels at the chosen DPI, so the printed square
    is not exactly the nominal size even before the printer touches it -- at
    300 dpi a 30 mm square renders as 354 px = 29.972 mm, 0.093% small.

    The pipeline believes the nominal size, so every reported millimetre comes
    out scaled by 1/board_scale. This test asserts that the error is exactly the
    predicted amount -- i.e. that the effect is fully understood and could be
    corrected by measuring the printed square with calipers.
    """
    if os.environ.get("S2P_SKIP_PRINTER", "") == "1":
        print("[printer]        skipped (S2P_SKIP_PRINTER=1)")
        return

    nominal = BoardSpec().square_mm
    effective = 354 / 300 * 25.4          # charuco.render_target at 300 dpi
    board_scale = effective / nominal
    out_dir = Path(os.environ.get("S2P_TEST_DIR", ROOT / "out" / "synth_test")).parent / "synth_misprint"

    print(
        f"[printer]        simulating a board printed at {board_scale * 100:.4f}% "
        f"({effective:.4f}mm squares vs {nominal}mm nominal)"
    )
    scene = make_scene(
        out_dir,
        object_mm=(50.0, 30.0, 20.0),
        n_frames=int(os.environ.get("S2P_TEST_FRAMES", "30")),
        resolution=(1440, 1080),
        board_scale=board_scale,
        force=os.environ.get("S2P_TEST_FORCE", "") == "1",
    )
    truth = scene["truth"]
    track = detect_track(truth, "png")
    recovered, rel = recover_dims(truth, track)

    true_dims = np.array(truth["object"]["true_dims_mm"], dtype=np.float64)
    expected = np.array(truth["object"]["expected_reported_mm"], dtype=np.float64)

    print(
        f"[printer]        true={true_dims.tolist()}mm  "
        f"reported={np.round(recovered, 4).tolist()}mm"
    )
    print(
        f"[printer]        predicted={np.round(expected, 4).tolist()}mm  "
        f"prediction err={np.abs(recovered - expected).max():.4f}mm"
    )
    print(
        f"[printer]        COST OF A 0.093% MISPRINT: "
        f"{np.abs(recovered - true_dims).max():.4f}mm on the longest axis "
        f"({np.abs(recovered / true_dims - 1).max() * 100:.4f}%)"
    )

    check(
        np.abs(recovered - expected).max() < TOL_DIM_MM,
        f"mis-scaled-board result {recovered} does not match the predicted "
        f"{expected} -- the board_scale model is wrong",
    )
    # The whole point: the error is real and is NOT caught by the fit residual.
    check(
        np.abs(recovered - true_dims).max() > 0.01,
        "expected a measurable error from a mis-scaled board but saw none; "
        "the board_scale simulation is not actually doing anything",
    )


def test_ground_truth_mesh(truth: dict[str, Any]) -> None:
    """The exported STL must carry the same exact dimensions, in millimetres."""
    stl = truth["object"].get("mesh_stl")
    if not stl or not Path(stl).exists():
        print("[mesh]           no ground-truth STL exported -- skipped")
        return
    try:
        import trimesh
    except ImportError:
        print("[mesh]           trimesh not available -- skipped")
        return
    # process=True merges the duplicated vertices STL always carries; without
    # it every triangle is an island and is_watertight is meaninglessly False.
    mesh = trimesh.load(stl, process=True)
    extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
    true_dims = np.array(truth["object"]["true_dims_mm"], dtype=np.float64)
    # STL is in the physical/world frame, whose axes match BOARD up to signs,
    # so the extents are directly comparable.
    err = np.abs(np.sort(extents) - np.sort(true_dims)).max()
    print(
        f"[mesh]           STL extents={np.round(extents, 4).tolist()}mm "
        f"vs truth={true_dims.tolist()}mm  max_err={err:.5f}mm  "
        f"watertight={mesh.is_watertight} "
        f"volume={mesh.volume:.1f}mm3 vs blender {truth['object']['volume_mm3']:.1f}mm3"
    )
    check(
        err < 0.01,
        f"exported STL bbox differs from ground truth by {err:.4f}mm -- the "
        f"mesh export is applying a unit conversion it should not",
    )


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("scan2print synthetic ground-truth accuracy test")
    print("=" * 78)
    missing = [k for k, v in config.check_tools().items() if not v]
    print(f"[tools] present={ {k: v for k, v in config.check_tools().items()} }")

    for name in ("charuco", "calib", "pose", "scale", "recon"):
        mark = "yes" if optional_attr(f"scan2print.{name}", "__name__") is not None else "not yet"
        print(f"[module] scan2print.{name}: {mark}")

    scene = build_scene()
    truth = scene["truth"]
    print(f"[truth] object true dims = {truth['object']['true_dims_mm']} mm")
    print(f"[truth] board = {truth['board']['squares_x']}x{truth['board']['squares_y']} "
          f"@ {truth['board']['square_mm']}mm squares "
          f"({truth['board']['squares_x'] * truth['board']['square_mm']:.0f} x "
          f"{truth['board']['squares_y'] * truth['board']['square_mm']:.0f} mm)")
    print(f"[truth] {truth['board_texture']['detected_corners']} corners detected in "
          f"the board texture itself (residual "
          f"{truth['board_texture']['max_corner_residual_px']:.2f}px)")

    failures: list[str] = []
    for label in ("png", "video"):
        print("-" * 78)
        track = detect_track(truth, label)
        try:
            test_board_detects(truth, track, label)
            test_pose_recovery(track, label)
            if label == "png":
                test_scale_recovery(truth, track)
                test_pipeline_scale_module(truth, track)
        except Fail as exc:
            failures.append(f"{label}: {exc}")
            print(f"[FAIL:{label}] {exc}")
    print("-" * 78)
    try:
        test_ground_truth_mesh(truth)
    except Fail as exc:
        failures.append(f"mesh: {exc}")
        print(f"[FAIL:mesh] {exc}")
    print("-" * 78)
    try:
        test_printer_scale_error()
    except Fail as exc:
        failures.append(f"printer: {exc}")
        print(f"[FAIL:printer] {exc}")

    print("=" * 78)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS -- recovered dimensions match ground truth within tolerance.")
    print("NOTE: idealised reconstruction; see the module docstring for what")
    print("      this does and does not prove about real-world accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
