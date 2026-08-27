"""scan2print command line -- the thing a human actually types.

    scan2print run <video.mov>     full pipeline, with a real review gate
    scan2print target              print-ready ChArUco target
    scan2print calibrate <dir>     camera intrinsics from board photos
    scan2print synth               synthetic ground-truth accuracy demo
    scan2print doctor              environment / dependency check
    scan2print cube                printer shrinkage calibration cube

Design notes worth knowing before editing this file:

* Heavy modules (cv2, trimesh, Blender wrappers) are imported INSIDE the
  command functions. `scan2print doctor` and `--help` stay instant, and a
  broken optional dependency only breaks the command that needs it.
* `run` writes everything under out/<video-stem>/, including run.json, which
  records every parameter and every measured result. A demo run is therefore
  reproducible and inspectable after the fact.
* The review gate is real, and it keys on more than the solver's own opinion.
  No STL is compensated, no gcode is produced, and nothing is sent to a printer
  until a human confirms. A "reject" scale verdict and a FAILED caliper
  cross-check both block outright without --force; a scale that was never
  compared against the physical object cannot be waved through with --yes.
  The quality verdict itself is verify.assess()'s, not the scale solver's: the
  fit residual is self-consistency only, and the witnesses that can actually
  see a wrong millimetre (reprojection error, frame solve rate, the pairwise
  cross-check, intrinsics provenance, the calipers) can only lower it.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config
from .types import BoardSpec, Intrinsics, ReconPose, ScaleResult

console = Console()

# The arbitrary scale an "ideal" synthetic reconstruction is generated at.
# Deliberately nowhere near 1.0 so any accidental "recon units are already
# millimetres" shortcut fails loudly instead of looking correct.
SYNTH_RECON_MM_PER_UNIT = 0.0137


class CliError(RuntimeError):
    """A user-facing failure: printed as a red panel, exit code 2."""


# ==========================================================================
# small shared helpers
# ==========================================================================

def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


class Stages:
    """Rich stage banners plus honest wall-clock timing for run.json."""

    def __init__(self, con: Console) -> None:
        self.con = con
        self.times: dict[str, float] = {}
        self._n = 0

    @contextmanager
    def __call__(self, name: str, note: str = "") -> Iterator[None]:
        self._n += 1
        label = f"[bold cyan]stage {self._n}[/bold cyan]  {name}"
        if note:
            label += f"  [dim]{note}[/dim]"
        self.con.rule(label, align="left")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.times[name] = round(dt, 3)
            self.con.print(f"[dim]  -> {name} finished in {dt:.2f}s[/dim]")

    def table(self) -> Table:
        t = Table(box=ROUNDED, title="stage timing", title_style="bold", expand=False)
        t.add_column("stage", style="cyan")
        t.add_column("seconds", justify="right")
        for k, v in self.times.items():
            t.add_row(k, f"{v:.2f}")
        t.add_row("[bold]total", f"[bold]{sum(self.times.values()):.2f}")
        return t


def _board_from_args(args: argparse.Namespace) -> BoardSpec:
    """BoardSpec from --board-json and/or the individual overrides."""
    spec = BoardSpec()
    src = getattr(args, "board_json", None)
    if src:
        p = Path(src).expanduser()
        if not p.exists():
            raise CliError(f"--board-json {p} does not exist")
        data = json.loads(p.read_text())
        # Accept a bare BoardSpec dump, a render_target() sidecar ("spec"), or a
        # synth ground-truth file ("board").
        blob = data.get("spec") or data.get("board") or data
        spec = BoardSpec.from_json(blob)
        eff = data.get("effective_square_mm") or blob.get("effective_square_mm")
        if eff and abs(float(eff) - spec.square_mm) > 1e-6:
            console.print(
                f"[yellow]note[/yellow] sidecar records effective_square_mm="
                f"{float(eff):.5f} (rendered), nominal={spec.square_mm:g}. Using the "
                "effective value -- it is what was actually put on paper."
            )
            spec.square_mm = float(eff)
    for attr, field in (("squares_x", "squares_x"), ("squares_y", "squares_y"),
                        ("square_mm", "square_mm"), ("marker_mm", "marker_mm"),
                        ("dict_name", "dict_name")):
        v = getattr(args, attr, None)
        if v is not None:
            setattr(spec, field, v)
    if spec.marker_mm >= spec.square_mm:
        raise CliError(
            f"marker_mm ({spec.marker_mm}) must be smaller than square_mm "
            f"({spec.square_mm}) -- the ArUco marker sits inside the white square."
        )
    return spec


def _default_intrinsics_path() -> Path:
    return config.PROFILES_DIR / "intrinsics.json"


def _load_intrinsics(args: argparse.Namespace, con: Console) -> Intrinsics | None:
    """Explicit --intrinsics, else profiles/intrinsics.json, else None."""
    p = getattr(args, "intrinsics", None)
    if p:
        p = Path(p).expanduser()
        if not p.exists():
            raise CliError(f"--intrinsics {p} does not exist")
        intr = _tag_provenance(Intrinsics.load(p))
        con.print(f"[green]intrinsics[/green] loaded from {p} "
                  f"(fx={intr.fx:.2f} {intr.width}x{intr.height}, rms={intr.rms_px:.3f}px, "
                  f"source={intr.source})")
        return intr
    if getattr(args, "no_intrinsics", False):
        return None
    d = _default_intrinsics_path()
    if d.exists():
        intr = _tag_provenance(Intrinsics.load(d))
        con.print(f"[green]intrinsics[/green] using default profile {d} "
                  f"(fx={intr.fx:.2f} {intr.width}x{intr.height}, source={intr.source})")
        return intr
    return None


def _tag_provenance(intr: Intrinsics) -> Intrinsics:
    """Fill in Intrinsics.source for profiles written before it existed.

    A file with a non-zero calibration RMS can only have come from
    `scan2print calibrate`; anything else stays "unknown", which the review
    gate treats as unmeasured. Never upgrades a source that is already set.
    """
    if intr.source == "unknown" and intr.rms_px > 0.0:
        intr.source = "charuco-calibration"
    return intr


def _first_frame_size(frames_dir: Path) -> tuple[int, int]:
    from PIL import Image

    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in
                   (".png", ".jpg", ".jpeg"))
    if not files:
        raise CliError(f"no frames found in {frames_dir}")
    with Image.open(files[0]) as im:
        return im.size  # (w, h)


def _intrinsics_from_manifest(manifest: dict[str, Any],
                              source_of: Intrinsics | None = None) -> Intrinsics | None:
    """Intrinsics valid for the frames on disk, per frames_manifest.json.

    frames.prepare() rectifies the frames when it is given a calibration, so
    the distortion to use downstream is ZERO -- applying intr.dist again would
    double-correct and bias every ChArUco pose, and therefore the scale.

    The manifest does not record provenance, so it is carried over from the
    calibration the run was given (`source_of`) when the resolutions agree.
    Getting this wrong in the safe direction only means a marginal verdict;
    getting it wrong the other way would let an unmeasured camera model be
    reported as a measured one.
    """
    blob = manifest.get("intrinsics")
    if not blob:
        return None
    out = Intrinsics(
        fx=blob["fx"], fy=blob["fy"], cx=blob["cx"], cy=blob["cy"],
        dist=[0.0] * 5, width=blob["width"], height=blob["height"],
    )
    if (source_of is not None and source_of.width == out.width
            and source_of.height == out.height):
        out.source = source_of.source
        out.rms_px = source_of.rms_px
    return out


def _video_identity(video: Path) -> dict[str, Any]:
    """Path + size + mtime. Path alone cannot see a file edited in place."""
    stat = video.stat()
    return {"path": str(video), "size_bytes": stat.st_size,
            "mtime": round(stat.st_mtime, 3)}


def _source_sidecar(workdir: Path) -> Path:
    return workdir / "source_video.json"


def _check_reuse_source(video: Path, workdir: Path, manifest_path: Path,
                        con: Console) -> None:
    """Refuse to reuse frames that came from a different video.

    --reuse skips extraction, so without this the pipeline happily builds an
    STL out of one scan's frames and names it after a video it never opened.
    frames_manifest.json records the source path; the sidecar this module
    writes at extraction time adds size and mtime, which is what catches a
    file that was re-exported or trimmed in place under the same name.
    """
    manifest = json.loads(manifest_path.read_text())
    recorded = manifest.get("video")
    if recorded and Path(recorded).expanduser().resolve() != video:
        raise CliError(
            f"--reuse refused: the frames in {workdir / 'frames'} were extracted from\n"
            f"    {recorded}\n"
            f"but this run was given\n"
            f"    {video}\n\n"
            "Reusing them would produce an STL named after a video that was never "
            "read, with the dimensions of a different scan. Drop --reuse to "
            "re-extract, or point --out at that video's own working directory.")

    side = _source_sidecar(workdir)
    if not side.exists():
        con.print("[yellow]--reuse:[/yellow] no source_video.json in this workdir, so "
                  "only the video PATH could be checked, not its contents. A file "
                  "edited in place under the same name cannot be detected.")
        return
    was = json.loads(side.read_text())
    now = _video_identity(video)
    drift = [k for k in ("size_bytes", "mtime")
             if was.get(k) is not None and was[k] != now[k]]
    if drift:
        raise CliError(
            f"--reuse refused: {video} has changed since these frames were extracted "
            f"({', '.join(drift)} differ: recorded "
            f"{ {k: was.get(k) for k in drift} } vs now { {k: now[k] for k in drift} }).\n"
            "The frames on disk are from the older file. Drop --reuse to re-extract.")


# ==========================================================================
# objcap (Apple Object Capture) subprocess wrapper
# ==========================================================================

def _run_objcap(frames_dir: Path, out_dir: Path, detail: str = "medium",
                poses_only: bool = False, timeout_s: float = 7200.0,
                con: Console | None = None) -> dict[str, Any]:
    """Run tools/objcap and return its parsed stdout JSON summary."""
    con = con or console
    binp = config.OBJCAP_BIN
    if not binp.exists():
        raise CliError(
            f"Object Capture wrapper not built: {binp} is missing.\n"
            f"Build it with:  cd {config.ROOT / 'tools/objcap'} && swift build -c release"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binp), "--input", str(frames_dir), "--output", str(out_dir),
           "--detail", detail]
    if poses_only:
        cmd.append("--poses-only")
    con.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise CliError(f"objcap timed out after {timeout_s:.0f}s on {frames_dir}")
    # Apple's USD localiser writes unrelated warnings straight to fd 2 from a
    # background thread; only lines it actually owns are prefixed 'objcap: '.
    for line in proc.stderr.splitlines():
        if line.startswith("objcap: "):
            con.print("[dim]" + line + "[/dim]")
    if proc.returncode != 0:
        own = [l for l in proc.stderr.splitlines() if l.startswith("objcap: ")]
        raise CliError(
            "Object Capture failed (exit "
            f"{proc.returncode}).\n" + ("\n".join(own[-6:]) or proc.stderr[-1500:]) +
            "\n\nThe usual cause is input it cannot align: too few frames, motion "
            "blur, or a low-texture/shiny subject. Reshoot slower, add matte spray, "
            "and keep the object filling the frame."
        )
    summary: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                pass
    if not summary:
        raise CliError("objcap produced no JSON summary on stdout:\n" + proc.stdout[-1500:])
    summary["wall_s"] = round(time.perf_counter() - t0, 2)
    return summary


def _load_recon_poses(poses_json: Path) -> tuple[dict[str, ReconPose], dict[str, Any]]:
    if not poses_json.exists():
        raise CliError(f"reconstruction poses missing: {poses_json}")
    data = json.loads(poses_json.read_text())
    poses: dict[str, ReconPose] = {}
    for d in data.get("images", []):
        rp = ReconPose(name=d["name"], transform=d["transform"], center=d["center"])
        poses[rp.name] = rp
    if not poses:
        raise CliError(f"{poses_json} contains no camera poses")
    return poses, data


def _intrinsics_from_objcap(data: dict[str, Any], size: tuple[int, int]) -> Intrinsics | None:
    """Object Capture's own recovered K (macOS 26+), as a fallback calibration."""
    by_name = data.get("intrinsics_by_name") or {}
    if not by_name:
        return None
    Ks = [np.array(v, dtype=np.float64).reshape(3, 3, order="F") for v in by_name.values()]
    K = np.median(np.stack(Ks), axis=0)
    w, h = size
    # source="objectcapture" -> Intrinsics.is_calibrated is False. This is a
    # self-estimate, not a measurement: it landed 0.33% off on ground-truth
    # renders, and 0.33% of focal length is 0.33% of every dimension reported.
    return Intrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                      cy=float(K[1, 2]), dist=[0.0] * 5, width=w, height=h,
                      source="objectcapture")


# ==========================================================================
# doctor
# ==========================================================================

def cmd_doctor(args: argparse.Namespace) -> int:
    con = console
    con.rule("[bold]scan2print doctor", align="left")

    env = Table(box=ROUNDED, title="environment", title_style="bold", expand=False)
    env.add_column("item", style="cyan")
    env.add_column("value")
    env.add_row("python", f"{platform.python_version()}  ({sys.executable})")
    env.add_row("in venv", "yes" if sys.prefix != sys.base_prefix else "NO (use .venv/bin/python)")
    env.add_row("platform", f"{platform.system()} {platform.release()} / {platform.machine()}")
    env.add_row("project root", str(config.ROOT))
    env.add_row("out dir", f"{config.OUT_DIR}  ({'exists' if config.OUT_DIR.exists() else 'will be created'})")
    con.print(env)

    pkgs = Table(box=ROUNDED, title="python packages", title_style="bold", expand=False)
    pkgs.add_column("package", style="cyan")
    pkgs.add_column("version")
    pkgs.add_column("status")
    ok_pkgs = True
    for mod, attr in (("numpy", "__version__"), ("scipy", "__version__"),
                      ("cv2", "__version__"), ("trimesh", "__version__"),
                      ("PIL", "__version__"), ("rich", "__version__"),
                      ("manifold3d", "__version__"), ("shapely", "__version__"),
                      ("rtree", "__version__"), ("networkx", "__version__"),
                      ("fast_simplification", "__version__"),
                      ("mapbox_earcut", "__version__")):
        try:
            m = __import__(mod)
            ver = str(getattr(m, attr, "?"))
            pkgs.add_row(mod, ver, "[green]ok[/green]")
        except Exception as exc:  # noqa: BLE001 - report, never crash the doctor
            ok_pkgs = False
            pkgs.add_row(mod, "-", f"[red]MISSING[/red] {type(exc).__name__}")
    con.print(pkgs)

    tools = config.check_tools()
    hints = {
        "orca": "install Orca-Flashforge to /Applications (slicing only)",
        "blender": "install Blender 5.x to /Applications (USD import + synth renders)",
        "ffmpeg": "brew install ffmpeg",
        "ffprobe": "brew install ffmpeg",
        "objcap": f"cd {config.ROOT / 'tools/objcap'} && swift build -c release",
        "machine_preset": "open Orca-Flashforge once so it writes its system presets",
    }
    paths = {
        "orca": config.ORCA_BIN, "blender": config.BLENDER_BIN,
        "ffmpeg": config.FFMPEG_BIN, "ffprobe": config.FFPROBE_BIN,
        "objcap": config.OBJCAP_BIN, "machine_preset": config.MACHINE_PRESET,
    }
    ttab = Table(box=ROUNDED, title="external tools", title_style="bold", expand=False)
    ttab.add_column("tool", style="cyan")
    ttab.add_column("status")
    ttab.add_column("path / hint")
    for k, present in tools.items():
        ttab.add_row(k, "[green]ok[/green]" if present else "[red]MISSING[/red]",
                     str(paths[k]) if present else f"{paths[k]}  [yellow]({hints[k]})[/yellow]")
    usdcat = Path("/usr/bin/usdcat")
    ttab.add_row("usdcat", "[green]ok[/green]" if usdcat.exists() else "[yellow]missing[/yellow]",
                 str(usdcat) + "  [dim](ships with macOS; USD stage metadata)[/dim]")
    con.print(ttab)

    assets = Table(box=ROUNDED, title="project assets", title_style="bold", expand=False)
    assets.add_column("item", style="cyan")
    assets.add_column("status")
    targets = sorted(config.TARGETS_DIR.glob("*.pdf")) if config.TARGETS_DIR.exists() else []
    assets.add_row("printable targets",
                   f"[green]{len(targets)}[/green] pdf(s) in {config.TARGETS_DIR}" if targets
                   else "[yellow]none -- run: scan2print target[/yellow]")
    ip = _default_intrinsics_path()
    assets.add_row("camera calibration",
                   f"[green]{ip}[/green]" if ip.exists()
                   else "[yellow]none -- run: scan2print calibrate <dir-of-board-photos>[/yellow]")
    cal = config.PROFILES_DIR / "printer_cal.json"
    assets.add_row("printer shrinkage cal",
                   f"[green]{cal}[/green]" if cal.exists()
                   else "[yellow]none -- run: scan2print cube (defaults to literature shrink)[/yellow]")
    con.print(assets)

    missing = [k for k, v in tools.items() if not v]
    if missing or not ok_pkgs:
        con.print(Panel(
            Text("Missing: " + ", ".join(missing + ([] if ok_pkgs else ["python packages"]))
                 + "\nThe pipeline stages that need them will fail with a specific error.",
                 style="yellow"),
            title="not fully equipped", border_style="yellow", expand=False))
        return 1
    con.print(Panel(Text("All external tools and packages present.", style="bold green"),
                    border_style="green", expand=False))
    return 0


# ==========================================================================
# target
# ==========================================================================

def cmd_target(args: argparse.Namespace) -> int:
    from . import charuco

    con = console
    spec = _board_from_args(args)
    out = Path(args.out).expanduser() if args.out else (
        config.TARGETS_DIR /
        f"charuco_{spec.squares_x}x{spec.squares_y}_{spec.square_mm:g}mm_{args.dpi}dpi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    con.rule("[bold]generate ChArUco target", align="left")
    png = charuco.render_target(spec, out, dpi=args.dpi, paper=args.paper,
                                margin_mm=args.margin_mm, write_pdf=not args.no_pdf)
    sidecar = png.with_suffix(".json")
    tbl = Table(box=ROUNDED, show_header=False, expand=False)
    tbl.add_column("k", style="cyan")
    tbl.add_column("v")
    tbl.add_row("png", str(png))
    if not args.no_pdf:
        tbl.add_row("pdf", str(png.with_suffix(".pdf")))
    tbl.add_row("sidecar", str(sidecar))
    if sidecar.exists():
        side = json.loads(sidecar.read_text())
        for k in ("effective_square_mm", "page_mm", "board_mm", "effective_board_mm", "dpi"):
            if k in side:
                tbl.add_row(k, str(side[k]))
    con.print(tbl)
    con.print(Panel(Text(
        "PRINT AT 100% / 'Actual size'. Turn OFF 'fit to page' and 'scale to fit'.\n"
        "Then MEASURE the printed 100 mm caliper bar. If it is not 100.00 mm, the page "
        "was scaled and every millimetre this program reports will be wrong by the same "
        "ratio -- either reprint, or measure one printed square with calipers and pass "
        "that with --square-mm.", style="bold yellow"),
        title="the one step people get wrong", border_style="yellow", expand=False))
    return 0


# ==========================================================================
# calibrate
# ==========================================================================

def cmd_calibrate(args: argparse.Namespace) -> int:
    import cv2

    from . import charuco

    con = console
    d = Path(args.image_dir).expanduser()
    if not d.is_dir():
        raise CliError(f"{d} is not a directory of board photographs")
    exts = (".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".bmp")
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise CliError(f"no images in {d} (looked for {exts})")
    spec = _board_from_args(args)
    con.rule(f"[bold]calibrate from {len(paths)} images", align="left")
    flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0
    if args.fix_k3:
        con.print("[dim]using cv2.CALIB_FIX_K3 (recommended for phone lenses: k3 mostly "
                  "absorbs noise)[/dim]")
    try:
        intr = charuco.calibrate_camera(paths, spec, flags=flags)
    except RuntimeError as exc:
        raise CliError(f"calibration refused: {exc}")
    out = Path(args.out).expanduser() if args.out else _default_intrinsics_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    intr.save(out)
    tbl = Table(box=ROUNDED, title="intrinsics", title_style="bold", expand=False,
                show_header=False)
    tbl.add_column("k", style="cyan")
    tbl.add_column("v")
    tbl.add_row("resolution", f"{intr.width} x {intr.height}")
    tbl.add_row("fx, fy", f"{intr.fx:.3f}, {intr.fy:.3f}")
    tbl.add_row("cx, cy", f"{intr.cx:.3f}, {intr.cy:.3f}")
    tbl.add_row("dist", ", ".join(f"{v:.5g}" for v in intr.dist))
    rms_style = "green" if intr.rms_px < 1.0 else "yellow" if intr.rms_px < 2.0 else "red"
    tbl.add_row("reprojection RMS", f"[{rms_style}]{intr.rms_px:.4f} px[/{rms_style}]")
    tbl.add_row("saved to", str(out))
    con.print(tbl)
    con.print("[dim]This calibration is valid ONLY for this camera at this resolution "
              "and zoom. Re-run it if you change lens, zoom, or capture resolution.[/dim]")
    return 0


# ==========================================================================
# cube (printer shrinkage calibration)
# ==========================================================================

def cmd_cube(args: argparse.Namespace) -> int:
    from . import compensate

    con = console
    if args.measure:
        if len(args.measure) != 3:
            raise CliError("--measure needs exactly three numbers: X Y Z (mm)")
        cal = compensate.update_cal_from_measurement(
            args.material, args.measure, nominal_mm=args.size)
        con.print(Panel(Text(compensate.describe(cal, args.material), style="bold green"),
                        title="printer calibration updated", border_style="green",
                        expand=False))
        con.print(f"[dim]saved to {compensate.cal_path()}[/dim]")
        return 0

    out = Path(args.out).expanduser() if args.out else config.OUT_DIR / "calibration_cube.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    p = compensate.make_calibration_cube(out, size_mm=args.size)
    con.print(f"[green]wrote[/green] {p}  ({p.stat().st_size / 1024:.0f} KiB, "
              f"{args.size:g} mm nominal on every axis)")
    con.print(Panel(Text(
        f"1. Slice and print {p.name} in the filament you will actually use.\n"
        f"2. Let it cool completely (30+ min). Measure X, Y and Z with calipers,\n"
        f"   across the three UNNOTCHED faces. The notch marks the +X+Y corner.\n"
        f"3. Record it:  scan2print cube --measure <X> <Y> <Z> --material {args.material}\n"
        f"   e.g.         scan2print cube --measure 29.88 29.86 30.05 --material PLA\n\n"
        f"Until you do that, `run` pre-scales by the literature estimate for "
        f"{args.material} ({config.DEFAULT_SHRINK.get(args.material, 0.0) * 100:.1f}% shrink), "
        f"which is a guess, not a measurement.", style="yellow"),
        title="how to use the cube", border_style="yellow", expand=False))
    con.print(f"[dim]{compensate.describe(None, args.material)}[/dim]")
    if args.slice:
        from . import slicer

        res = slicer.slice_stl(p, out.parent, material=args.material)
        con.print(res.summary())
    return 0


# ==========================================================================
# run -- the full pipeline
# ==========================================================================

def _compensation_plan(compensate: Any, cal: dict[str, Any] | None, material: str,
                       assume_shrink: bool) -> dict[str, Any]:
    """compensate.compensation_plan(), with a conservative fallback.

    The fallback exists so that an older compensate module cannot cause this
    file to apply an unmeasured correction by accident: no plan means no
    compensation unless the material is actually calibrated.
    """
    plan_fn = getattr(compensate, "compensation_plan", None)
    if plan_fn is not None:
        return plan_fn(cal, material, assume_shrink)
    entry = (cal or {}).get("materials", {}).get(material) or {}
    measured = bool(entry.get("calibrated"))
    return {
        "material": material,
        "factors": {a: float(entry.get(a, 1.0)) if measured else 1.0 for a in "xyz"},
        "applied": measured or assume_shrink,
        "calibrated": measured,
        "source": "measured" if measured else ("estimate" if assume_shrink else "none"),
        "reason": ("measured on a gauge cube" if measured else
                   "uncalibrated: config.APPLY_UNCALIBRATED_SHRINK is False, so nothing "
                   "was applied"),
    }


def _review_gate(quality: str, args: argparse.Namespace, con: Console,
                 caliper: dict[str, Any] | None = None) -> tuple[bool, str]:
    """The genuine stop. Returns (proceed, reason-recorded-in-run.json).

    Two independent verdicts can stop a run, and they stop it for different
    reasons:

      quality  -- the scale solve plus the witnesses assess() folded into it.
                  Self-consistency and camera-model sanity.
      caliper  -- the only comparison against the physical world. It is the
                  ONLY check that can see a mis-measured board square, so a
                  'fail' here blocks exactly like quality='reject', and
                  anything short of a stated-axis 'ok' is enough to take away
                  --yes: an unverified scale must not reach gcode because a
                  script said so.
    """
    caliper = caliper or {}
    cal_verdict = caliper.get("verdict", "no-measurement")
    certified_ok = cal_verdict == "ok" and bool(caliper.get("certifies_scale"))

    blocking = quality == "reject" or cal_verdict == "fail"
    why_blocking = " and ".join(
        ([f"the scale verdict is '{quality}'"] if quality == "reject" else [])
        + ([f"the caliper cross-check FAILED ({caliper.get('error_pct', 0.0):+.3f}% on "
            f"{caliper.get('assumed_dimension', 'the measured dimension')})"]
           if cal_verdict == "fail" else []))

    if blocking and not args.force:
        why: list[str] = []
        if quality == "reject":
            why.append(
                "A rejected verdict means the measured evidence does not support these "
                "millimetres -- either the two camera tracks disagree by more than the "
                "board geometry can explain, or one of the independent witnesses above "
                "is out of range. The report says which.")
        if cal_verdict == "fail":
            why.append(
                "A failed caliper check means the model disagrees with the physical "
                "object you measured. That is the one comparison in this program made "
                "against the real world, and it is the only thing here that can catch a "
                "mis-measured board square -- every other number is blind to it, and "
                "gets BETTER when the board size is wrong.")
        con.print(Panel(Text(
            f"REFUSED: {why_blocking}. Nothing has been sliced and no gcode exists.\n\n"
            + "\n\n".join(why)
            + "\n\nFix the cause, do not paper over it. If you understand that and want "
              "the STL anyway, re-run with --force.",
            style="bold red"),
            title="REVIEW GATE: BLOCKED", border_style="red", expand=False))
        return False, f"blocked: {why_blocking}, --force not given"

    # Short of a stated-axis 'ok', the scale is unverified against reality.
    # --yes may not stand in for that; a human must say so, or say --force.
    unverified_reason = ""
    if not certified_ok and not blocking:
        unverified_reason = {
            "no-measurement": "no caliper reading was supplied",
            "no-axis": "a caliper reading was given but not which dimension it is",
            "ambiguous": "the caliper reading could not be attributed to one dimension",
            "warn": (f"the caliper cross-check WARNED "
                     f"({caliper.get('error_pct', 0.0):+.3f}% on "
                     f"{caliper.get('assumed_dimension', 'the measured dimension')})"),
            "ok": ("the caliper reading was auto-matched to the nearest model extent, "
                   "which proves nothing"),
        }.get(cal_verdict, f"the caliper cross-check returned {cal_verdict!r}")

    if args.yes:
        if blocking:
            con.print("[bold red]--yes + --force: proceeding on a verdict that should "
                      "have stopped this run. The dimensions of this part are not "
                      "trustworthy.[/bold red]")
            return True, f"auto-confirmed by --yes with --force ({why_blocking})"
        if unverified_reason and not args.force:
            con.print(Panel(Text(
                f"NOT CONFIRMED: {unverified_reason}, so this scale has never been "
                "compared against the physical object. --yes is not enough to print an "
                "unverified dimension -- that is the whole failure this tool exists to "
                "prevent.\n\nEither:\n"
                "  * measure one real dimension and re-run with "
                "--caliper-mm <mm> --caliper-axis longest|middle|shortest\n"
                "  * confirm interactively (drop --yes)\n"
                "  * pass --force to record, in run.json, that you accepted an "
                "unverified scale.\n\n"
                "The STL is on disk and correct-as-scanned; only slicing was skipped.",
                style="bold yellow"),
                title="REVIEW GATE: UNVERIFIED SCALE", border_style="yellow", expand=False))
            return False, f"not confirmed: {unverified_reason} and only --yes was given"
        if unverified_reason:
            con.print(f"[bold yellow]--yes + --force: proceeding although "
                      f"{unverified_reason}.[/bold yellow]")
            return True, f"auto-confirmed by --yes with --force ({unverified_reason})"
        con.print(f"[yellow]--yes given: proceeding without interactive confirmation "
                  f"(quality={quality}, caliper=ok on "
                  f"{caliper.get('assumed_dimension', 'a stated dimension')}).[/yellow]")
        return True, f"auto-confirmed by --yes (quality={quality}, caliper=ok)"

    if not sys.stdin.isatty():
        con.print(Panel(Text(
            "No terminal available to confirm on, and --yes was not passed. Stopping "
            "before slicing. Re-run interactively, or pass --yes if you have already "
            "reviewed these numbers.", style="bold yellow"),
            title="REVIEW GATE: NOT CONFIRMED", border_style="yellow", expand=False))
        return False, "not confirmed: non-interactive session and no --yes"

    if blocking:
        con.print(f"[bold red]{why_blocking.capitalize()}, and you passed "
                  f"--force.[/bold red]")
        want = "PRINT ANYWAY"
        ans = input(f'Type exactly "{want}" to continue, anything else aborts: ').strip()
        if ans != want:
            return False, f"user declined at the review gate ({why_blocking})"
        return True, f'user typed "PRINT ANYWAY" ({why_blocking})'

    if unverified_reason:
        con.print(f"[bold yellow]Unverified: {unverified_reason}. Nothing here has been "
                  f"compared against the physical object.[/bold yellow]")
    prompt = ("Do these dimensions match the real object? Proceed to compensation and "
              "slicing? [y/N]: ")
    ans = input(prompt).strip().lower()
    if ans not in ("y", "yes"):
        return False, "user declined at the review gate"
    return True, ("user confirmed at the review gate"
                  + (f" ({unverified_reason})" if unverified_reason else ""))


def cmd_run(args: argparse.Namespace) -> int:
    from . import charuco, frames as frames_mod, mesh as mesh_mod, scale as scale_mod, verify

    con = console
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise CliError(f"video not found: {video}")
    spec = _board_from_args(args)
    # Validate the caliper axis NOW, not after the reconstruction: a typo here
    # used to cost a full Object Capture run before it was noticed.
    try:
        verify.resolve_caliper_axis(args.caliper_axis)
    except ValueError as exc:
        raise CliError(f"--caliper-axis: {exc}")
    if args.caliper_mm is not None and args.caliper_axis is None and not sys.stdin.isatty():
        con.print("[yellow]note[/yellow] --caliper-mm was given without --caliper-axis, so "
                  "the cross-check cannot be attributed to a dimension and the review "
                  "gate will stop this run. Add --caliper-axis longest|middle|shortest.")
    workdir = (Path(args.out).expanduser() if args.out
               else config.OUT_DIR / video.stem)
    workdir.mkdir(parents=True, exist_ok=True)
    run_json = workdir / "run.json"

    rec: dict[str, Any] = {
        "schema": "scan2print.run/1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv[1:],
        "video": str(video),
        "workdir": str(workdir),
        "params": {
            "board": spec.to_json(), "frames_target": args.frames,
            "detail": args.detail, "material": args.material,
            "layer_mm": args.layer_mm, "dry_run": args.dry_run,
            "force": args.force, "yes": args.yes, "reuse": args.reuse,
            "keep_intermediates": args.keep_intermediates,
            "z_tol_mm": args.z_tol, "max_faces": args.max_faces,
            "no_compensate": args.no_compensate,
            "assume_shrink": args.assume_shrink,
            "caliper_mm": args.caliper_mm,
            "caliper_axis": args.caliper_axis,
        },
        "stages": {},
        "status": "started",
    }

    def flush() -> None:
        run_json.write_text(json.dumps(_jsonable(rec), indent=2))

    flush()
    st = Stages(con)
    con.print(Panel(Text(
        f"video   : {video}\nworkdir : {workdir}\nboard   : {spec.squares_x}x{spec.squares_y} "
        f"@ {spec.square_mm:g} mm squares ({spec.width_mm:g} x {spec.height_mm:g} mm)",
        style="bold"), title="scan2print run", border_style="cyan", expand=False))

    try:
        # -- 1. frames -----------------------------------------------------
        frames_dir = workdir / "frames"
        manifest_path = workdir / "frames_manifest.json"
        intr = _load_intrinsics(args, con)
        if intr is None:
            con.print(Panel(Text(
                "No camera calibration. Frames will NOT be undistorted and the pipeline "
                "will fall back to the intrinsics Object Capture recovers for itself "
                "(typically within ~0.2% of the truth on a good scan, but unvalidated on "
                "your lens). For metrology you want a real calibration:\n"
                "    scan2print calibrate <folder-of-board-photos> --fix-k3",
                style="yellow"), title="running uncalibrated", border_style="yellow",
                expand=False))
        with st("frames", f"target {args.frames}"):
            if args.reuse and manifest_path.exists() and any(frames_dir.glob("*.png")):
                # Before trusting them: are these frames even from THIS video?
                _check_reuse_source(video, workdir, manifest_path, con)
                frame_paths = sorted(frames_dir.glob("*.png"))
                con.print(f"[yellow]--reuse:[/yellow] keeping {len(frame_paths)} existing "
                          f"frames in {frames_dir} (extracted from this same video)")
            else:
                try:
                    frame_paths = frames_mod.prepare(
                        video, workdir, spec=spec, intr=intr, target=args.frames,
                        keep_candidates=args.keep_intermediates)
                except frames_mod.FrameError as exc:
                    raise CliError(f"frame extraction failed: {exc}")
                # Path, size and mtime of what was actually read, so a later
                # --reuse can tell an edited-in-place file from the original.
                _source_sidecar(workdir).write_text(
                    json.dumps(_video_identity(video), indent=2))
            manifest = json.loads(manifest_path.read_text())
            rec["stages"]["frames"] = {
                "n_frames": len(frame_paths), "dir": str(frames_dir),
                "source_video": manifest.get("video"),
                "source_identity": _video_identity(video),
                "undistorted": manifest.get("undistorted"),
                "selection": manifest.get("selection"),
                "probe": manifest.get("probe"),
            }
            gaps = (manifest.get("selection") or {}).get("n_gap_buckets")
            if gaps:
                con.print(f"[yellow]warning[/yellow] {gaps} temporal bucket(s) had no "
                          "usable frame -- the reconstruction will be thin in those "
                          "angular sectors.")
        flush()

        # -- 2. object capture --------------------------------------------
        recon_dir = workdir / "recon"
        poses_json = recon_dir / "poses.json"
        with st("object capture", f"detail={args.detail}"):
            if args.reuse and poses_json.exists():
                con.print(f"[yellow]--reuse:[/yellow] keeping existing reconstruction in "
                          f"{recon_dir}")
                oc_summary = {"reused": True}
            else:
                oc_summary = _run_objcap(frames_dir, recon_dir, detail=args.detail, con=con)
                con.print(f"[green]objcap[/green] {oc_summary.get('n_poses')} poses from "
                          f"{oc_summary.get('n_input_images')} images in "
                          f"{oc_summary.get('elapsed_s', oc_summary.get('wall_s'))}s "
                          f"(name_source={oc_summary.get('name_source')})")
            recon_poses, recon_data = _load_recon_poses(poses_json)
            model_files = sorted(recon_dir.glob("model.usd*"))
            rec["stages"]["objcap"] = {
                "summary": oc_summary, "n_recon_poses": len(recon_poses),
                "model": [str(p) for p in model_files],
            }
            if not model_files:
                raise CliError(
                    f"Object Capture produced poses but no mesh in {recon_dir}. "
                    "Re-run without --poses-only, or at a lower --detail.")
        flush()

        # -- 3. metric poses from the board -------------------------------
        with st("charuco metric poses"):
            eff_intr = _intrinsics_from_manifest(manifest, intr)
            if eff_intr is None:
                size = _first_frame_size(frames_dir)
                eff_intr = _intrinsics_from_objcap(recon_data, size)
                if eff_intr is None:
                    raise CliError(
                        "No camera intrinsics available: no calibration was supplied and "
                        "this Object Capture output carries no intrinsics_by_name. Run "
                        "`scan2print calibrate` and pass --intrinsics.")
                con.print(f"[yellow]using Object Capture's own recovered intrinsics[/yellow] "
                          f"fx={eff_intr.fx:.2f} cx={eff_intr.cx:.2f} cy={eff_intr.cy:.2f} "
                          f"at {eff_intr.width}x{eff_intr.height} -- a self-estimate, not "
                          f"a measurement; quality will be capped at 'marginal'")
            metric = charuco.poses_for_dir(frames_dir, spec, eff_intr)
            n_share = len(set(metric) & set(recon_poses))
            con.print(f"[green]board poses[/green] {len(metric)}/{len(frame_paths)} frames "
                      f"solved; {n_share} shared with the reconstruction")
            rms = [p.reproj_rms_px for p in metric.values()]
            # These two are the run-level camera-model witnesses: a wrong K
            # moves them by an order of magnitude while leaving the similarity
            # residual looking excellent. They are gated below, in assess().
            reproj_mean = round(float(np.mean(rms)), 4) if rms else None
            reproj_max = round(float(np.max(rms)), 4) if rms else None
            solve_frac = len(metric) / float(len(frame_paths) or 1)
            con.print(f"[{'green' if (reproj_mean or 0) <= config.RUN_REPROJ_GOOD_PX else 'yellow'}]"
                      f"reprojection RMS[/] mean {reproj_mean} px / max {reproj_max} px "
                      f"(gate: good <= {config.RUN_REPROJ_GOOD_PX:g}, reject > "
                      f"{config.RUN_REPROJ_MARGINAL_PX:g}); frames solved "
                      f"{solve_frac * 100:.0f}% (min {config.MIN_FRAME_SOLVE_FRAC * 100:.0f}%)")
            rec["stages"]["charuco"] = {
                "n_solved": len(metric), "n_frames": len(frame_paths),
                "n_shared_with_recon": n_share,
                "solved_frac": round(solve_frac, 4),
                "intrinsics": _jsonable(eff_intr),
                "intrinsics_calibrated": bool(eff_intr.is_calibrated),
                "reproj_rms_px": {"mean": reproj_mean, "max": reproj_max},
            }
            if n_share < config.SCALE_MIN_CAMERAS:
                raise CliError(
                    f"only {n_share} frames have BOTH a board pose and a reconstruction "
                    f"pose (need >= {config.SCALE_MIN_CAMERAS}). The board must be fully "
                    "visible in the same frames the object is reconstructed from. "
                    "Reshoot with the board in view throughout.")
        flush()

        # -- 4. scale ------------------------------------------------------
        with st("scale solve"):
            try:
                res: ScaleResult = scale_mod.solve_scale(recon_poses, metric)
            except ValueError as exc:
                raise CliError(f"scale solve impossible: {exc}")
            con.print(f"[bold]{res.notes[0] if res.notes else ''}[/bold]")
            rec["stages"]["scale"] = res.to_json()
        flush()

        # -- 5. mesh -------------------------------------------------------
        raw_model = model_files[0]
        stl_path = workdir / f"{video.stem}_scan.stl"
        with st("mesh conditioning", raw_model.name):
            try:
                m, mstats = mesh_mod.condition(
                    raw_model, res, spec, stl_path, z_tol=args.z_tol,
                    max_faces=args.max_faces, flatten_base=not args.no_flatten_base,
                    allow_reject=True)  # gate happens below, after the human sees it
            except mesh_mod.MeshError as exc:
                raise CliError(f"mesh conditioning failed: {exc}")
            rec["stages"]["mesh"] = {"stl": str(stl_path), "stats": mstats}
            con.print(f"[green]wrote[/green] {stl_path} "
                      f"({stl_path.stat().st_size / 1024:.0f} KiB)")
        flush()

        # -- 6. report + review gate --------------------------------------
        with st("verification report"):
            # The caliper check runs FIRST so its verdict can appear in the
            # report next to the numbers it contradicts.
            caliper = args.caliper_mm
            caliper_axis = args.caliper_axis
            if caliper is None and sys.stdin.isatty() and not args.yes:
                raw = input("Measure ONE real dimension of the object with calipers and "
                            "type it in mm (Enter to skip -- but then nothing here has "
                            "been checked against reality): ").strip()
                if raw:
                    try:
                        caliper = float(raw)
                    except ValueError:
                        con.print("[yellow]not a number; skipping the caliper check[/yellow]")
            if caliper is not None and caliper_axis is None and sys.stdin.isatty():
                caliper_axis = input(
                    "Which dimension is that? [longest / middle / shortest / x / y / z]: "
                ).strip() or None
            try:
                cal_res = verify.caliper_gate(mstats, caliper, axis=caliper_axis)
            except ValueError as exc:
                raise CliError(f"--caliper-axis: {exc}")
            rec["stages"]["caliper"] = cal_res

            witnesses = verify.assess(
                res, reproj_mean_px=reproj_mean, reproj_max_px=reproj_max,
                frames_solved=len(metric), frames_total=len(frame_paths),
                intrinsics=eff_intr, caliper=cal_res)
            summary = verify.report(res, mstats, spec, console=con, witnesses=witnesses,
                                    extras={"frames used": f"{n_share} of {len(frame_paths)}"})
            rec["stages"]["verify"] = summary
            verify.format_caliper_result(cal_res, spec, console=con)
        flush()

        con.rule("[bold]REVIEW GATE", align="left")
        quality = witnesses["quality"]
        proceed, reason = _review_gate(quality, args, con, caliper=cal_res)
        # Everything that had a vote is recorded, not just the one that used to.
        rec["gate"] = {
            "quality": quality,
            "solver_quality": res.quality,
            "quality_downgrades": witnesses["downgrades"],
            "caliper_verdict": cal_res["verdict"],
            "caliper_certifies_scale": bool(cal_res.get("certifies_scale")),
            "caliper_error_pct": cal_res.get("error_pct"),
            "reproj_mean_px": reproj_mean,
            "frames_solved_frac": round(solve_frac, 4),
            "intrinsics_source": eff_intr.source,
            "proceeded": proceed,
            "reason": reason,
        }
        flush()
        if not proceed:
            rec["status"] = "stopped_at_review_gate"
            rec["timing_s"] = st.times
            flush()
            con.print(st.table())
            con.print(f"[bold]STL (uncompensated, as scanned):[/bold] {stl_path}")
            con.print(f"[dim]run record: {run_json}[/dim]")
            return 3

        # -- 7. shrinkage compensation ------------------------------------
        print_stl = stl_path
        with st("shrinkage compensation", args.material):
            from . import compensate

            cal = compensate.load_cal()
            # compensate.compensation_plan() is the single decision point for
            # measured / nothing / unmeasured-estimate; run.json records which
            # of the three it was, so the machine-readable record cannot claim
            # a correction the console called a guess.
            plan = _compensation_plan(compensate, cal, args.material,
                                      bool(args.assume_shrink))
            if args.no_compensate:
                con.print("[yellow]--no-compensate: printing the as-scanned STL[/yellow]")
                rec["stages"]["compensate"] = {
                    "applied": False, "material": args.material, "source": "none",
                    "factors": [1.0, 1.0, 1.0], "factors_measured": False,
                    "reason": "--no-compensate given on the command line",
                }
            elif not plan["applied"]:
                # config.APPLY_UNCALIBRATED_SHRINK is False on purpose: silently
                # pre-scaling by a literature guess writes an unmeasured number
                # into a part whose whole point is a measured one.
                con.print(Panel(Text(
                    plan["reason"] + "\n\nNothing was pre-scaled: the print STL is the "
                    "scan STL. To make this real:\n"
                    "    scan2print cube            (print it, let it cool, measure it)\n"
                    f"    scan2print cube --measure <X> <Y> <Z> --material {args.material}\n"
                    "Or accept the unmeasured literature figure explicitly with "
                    "--assume-shrink.", style="yellow"),
                    title="shrinkage: uncalibrated, nothing applied",
                    border_style="yellow", expand=False))
                rec["stages"]["compensate"] = {
                    "applied": False, "material": args.material,
                    "source": plan["source"], "factors": [1.0, 1.0, 1.0],
                    "factors_measured": False, "reason": plan["reason"],
                }
            else:
                measured = plan["source"] == "measured"
                con.print(compensate.describe(cal, args.material,
                                              assume_shrink=bool(args.assume_shrink)))
                if not measured:
                    con.print("[bold yellow]--assume-shrink: pre-scaling by an UNMEASURED "
                              "literature estimate. This is a guess, and run.json records "
                              "it as one.[/bold yellow]")
                cm = compensate.apply(m, material=args.material, cal=cal,
                                      assume_shrink=bool(args.assume_shrink))
                print_stl = workdir / f"{video.stem}_print.stl"
                cm.export(str(print_stl))
                factors = [plan["factors"][a] for a in "xyz"]
                rec["stages"]["compensate"] = {
                    "applied": True, "material": args.material,
                    "factors": [round(float(v), 6) for v in factors],
                    "factors_measured": bool(measured),
                    "source": plan["source"],
                    "reason": plan["reason"],
                    "stl": str(print_stl),
                    "bbox_mm": [round(float(v), 4) for v in cm.extents],
                }
                con.print(f"[green]wrote[/green] {print_stl}  "
                          f"pre-scaled by {np.round(factors, 5).tolist()} "
                          f"({'measured' if measured else 'ASSUMED, not measured'})")
        flush()

        # -- 8. slice ------------------------------------------------------
        # 'complete' is a claim about the run, and a run whose only comparison
        # against the physical object failed (or never happened) has not
        # completed anything worth calling complete.
        cal_verdict = cal_res["verdict"]
        if cal_verdict == "ok" and cal_res.get("certifies_scale"):
            suffix = ""
        elif cal_verdict == "fail":
            suffix = "_caliper_failed"
        else:
            suffix = "_unverified"
        if args.dry_run:
            con.print(Panel(Text("--dry-run: stopping before slicing. The printable STL "
                                 f"is:\n{print_stl}", style="bold yellow"),
                            border_style="yellow", expand=False))
            rec["status"] = "dry_run_complete" + suffix
        else:
            with st("slice", config.PRINTER_MODEL):
                from . import slicer

                try:
                    sr = slicer.slice_stl(
                        print_stl, workdir / "gcode", material=args.material,
                        layer_mm=args.layer_mm, process=args.process,
                        filament=args.filament)
                except Exception as exc:  # slicer raises SliceError and friends
                    raise CliError(f"slicing failed: {exc}")
                con.print(sr.summary())
                for w in sr.warnings:
                    con.print(f"[yellow]slicer warning[/yellow] {w}")
                rec["stages"]["slice"] = {
                    "gcode": str(sr.gcode), "print_time_s": sr.print_time_s,
                    "print_time": sr.print_time_str, "filament_g": sr.filament_g,
                    "filament_cm3": sr.filament_cm3, "layers": sr.layers,
                    "max_z_mm": sr.max_z_mm, "machine": sr.machine,
                    "process": sr.process, "filament": sr.filament,
                    "command": sr.command, "warnings": sr.warnings,
                }
            rec["status"] = "complete" + suffix
        flush()
        if suffix:
            con.print(Panel(Text(
                ("The caliper cross-check FAILED and this run continued anyway under "
                 f"--force ({cal_res.get('error_pct', 0.0):+.3f}% on "
                 f"{cal_res.get('assumed_dimension', 'the measured dimension')})."
                 if cal_verdict == "fail" else
                 "This scale was never verified against the physical object.")
                + f"\nrun.json records status='{rec['status']}', not 'complete'. "
                "Measure the part before you cut leather to it.", style="bold yellow"),
                title="unverified result", border_style="yellow", expand=False))

        # -- wrap up -------------------------------------------------------
        if not args.keep_intermediates:
            shutil.rmtree(workdir / "candidates", ignore_errors=True)
        rec["timing_s"] = st.times
        flush()
        con.rule("[bold green]done", align="left")
        con.print(st.table())
        final = Table(box=ROUNDED, show_header=False, title="artefacts", title_style="bold",
                      expand=False)
        final.add_column("k", style="cyan")
        final.add_column("v")
        final.add_row("scan STL (as measured)", str(stl_path))
        if print_stl != stl_path:
            final.add_row("print STL (shrink-compensated)", str(print_stl))
        gc = rec["stages"].get("slice", {}).get("gcode")
        if gc:
            final.add_row("gcode", str(gc))
        final.add_row("run record", str(run_json))
        con.print(final)
        return 0

    except CliError:
        rec["status"] = "error"
        rec["timing_s"] = st.times
        flush()
        raise
    except Exception as exc:  # unexpected: still leave a usable run record
        rec["status"] = "crash"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["timing_s"] = st.times
        flush()
        raise


# ==========================================================================
# synth -- synthetic ground-truth accuracy demo
# ==========================================================================

def _ideal_recon(metric_centers_by_name: dict[str, np.ndarray],
                 mm_per_unit: float, seed: int = 11
                 ) -> tuple[dict[str, ReconPose], float, np.ndarray, np.ndarray]:
    """An error-free reconstruction track: truth pushed through a known similarity.

    Returns (recon poses, true mm-per-unit, true R, true t) where
    p_board_mm = mm_per_unit * R @ p_recon + t.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    t = rng.normal(scale=50.0, size=3)
    out: dict[str, ReconPose] = {}
    for name, c in metric_centers_by_name.items():
        p_recon = Q.T @ (np.asarray(c, dtype=float) - t) / mm_per_unit
        M = np.eye(4)
        M[:3, 3] = p_recon
        out[name] = ReconPose(name=name, transform=list(M.flatten(order="F")),
                              center=[float(v) for v in p_recon])
    return out, mm_per_unit, Q, t


def cmd_synth(args: argparse.Namespace) -> int:
    import cv2  # noqa: F401  (imported by charuco; kept explicit for the error message)

    from . import charuco, mesh as mesh_mod, scale as scale_mod, synth as synth_mod, verify

    con = console
    out = Path(args.out).expanduser() if args.out else config.OUT_DIR / "synth_cli"
    st = Stages(con)
    dims = tuple(float(v) for v in args.object_mm)
    if len(dims) != 3:
        raise CliError("--object-mm needs three numbers: X Y Z")

    gt_path = out / "ground_truth.json"
    with st("render synthetic scene", f"{args.frames} frames"):
        if args.reuse and gt_path.exists():
            truth = synth_mod.load_truth(out)
            con.print(f"[yellow]--reuse:[/yellow] existing scene at {out}")
        else:
            if not config.BLENDER_BIN.exists():
                raise CliError(f"Blender not found at {config.BLENDER_BIN}; the synthetic "
                               "demo renders its scene with it.")
            scene = synth_mod.make_scene(
                out, object_mm=dims, n_frames=args.frames,
                resolution=tuple(args.resolution), board_scale=args.board_scale,
                force=True, quiet=not args.verbose)
            truth = scene["truth"]
    # The scene on disk is the authority: with --reuse the rendered board scale can
    # differ from whatever was passed on this invocation.
    board_scale = float(truth.get("board_scale", args.board_scale))
    true_dims = np.array(truth["object"]["true_dims_mm"], dtype=float)
    expected_dims = np.array(truth["object"].get("expected_reported_mm", true_dims), dtype=float)
    spec = BoardSpec.from_json(truth["board"])
    cam = truth["camera"]
    intr = Intrinsics(fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
                      dist=list(cam["dist"]), width=cam["width"], height=cam["height"])
    frames_dir = Path(truth["frames"]["dir"])
    con.print(f"[bold]true object dims[/bold] {true_dims.tolist()} mm   "
              f"board {spec.squares_x}x{spec.squares_y} @ {spec.square_mm:g} mm"
              + (f"   [yellow]board printed at {board_scale * 100:.4f}%[/yellow]"
                 if board_scale != 1.0 else ""))

    # -- board metrology, for real, on the rendered pixels -----------------
    with st("charuco metric poses", f"{truth['frames']['count']} rendered frames"):
        metric = charuco.poses_for_dir(frames_dir, spec, intr)
        if not metric:
            raise CliError("no board poses recovered from the rendered frames")
        # Ground truth is in raw OpenCV board coordinates; charuco.py works in
        # the scan2print BOARD frame. Map before comparing.
        gt_center = {}
        for p in truth["poses"]:
            stem = Path(p["name"]).stem
            gt_center[stem] = charuco.cv_points_to_board(
                np.array(p["center"], dtype=float).reshape(1, 3), spec).reshape(3)
        errs = np.array([
            float(np.linalg.norm(metric[k].center_np - gt_center[k]))
            for k in metric if k in gt_center])
        rms = np.array([metric[k].reproj_rms_px for k in metric])
        con.print(f"[green]detected[/green] {len(metric)}/{truth['frames']['count']} frames; "
                  f"camera-centre error mean {errs.mean():.4f} mm / max {errs.max():.4f} mm; "
                  f"reprojection RMS mean {rms.mean():.4f} px")

    # -- reconstruction track ---------------------------------------------
    recon_mode = args.recon
    if recon_mode == "auto":
        recon_mode = "objcap" if config.OBJCAP_BIN.exists() else "ideal"
    model_files: list[Path] = []
    with st("reconstruction", recon_mode):
        recon: dict[str, ReconPose] = {}
        if recon_mode == "objcap":
            recon_dir = out / "recon"
            try:
                if args.reuse and (recon_dir / "poses.json").exists():
                    con.print(f"[yellow]--reuse:[/yellow] {recon_dir}")
                else:
                    _run_objcap(frames_dir, recon_dir, detail=args.detail, con=con)
                recon, _ = _load_recon_poses(recon_dir / "poses.json")
                model_files = sorted(recon_dir.glob("model.usd*"))
            except CliError as exc:
                con.print(Panel(Text(str(exc), style="yellow"),
                                title="Object Capture failed on the synthetic render; "
                                      "falling back to the ideal recon track",
                                border_style="yellow", expand=False))
                recon_mode = "ideal"
        if recon_mode == "ideal":
            # Built from GROUND TRUTH centres, not from the detected ones: fitting
            # the detected track against itself would be a tautology and would
            # report 0.0000% error no matter how bad the board metrology was.
            recon, s_true, Q_true, t_true = _ideal_recon(gt_center, SYNTH_RECON_MM_PER_UNIT)
            con.print(f"[dim]ideal recon track: GROUND-TRUTH camera centres pushed through "
                      f"a known similarity (s={s_true} mm/unit). Measures board metrology "
                      f"+ the similarity fit; does NOT exercise Object Capture.[/dim]")

    # -- scale solve --------------------------------------------------------
    scale_err_pct: float | None = None
    with st("scale solve"):
        res = scale_mod.solve_scale(recon, metric)
        con.print(f"[bold]{res.notes[0] if res.notes else ''}[/bold]")
        # Isolate the pipeline's OWN error: refit the same recon track against the
        # ground-truth camera centres instead of the detected ones. The difference
        # between the two scales is exactly what board detection contributed --
        # it separates "our metrology" from "Object Capture's mesh quality".
        shared = sorted(set(recon) & set(gt_center))
        if len(shared) >= 3:
            s_ref, _, _, rms_ref = scale_mod.umeyama(
                np.array([recon[k].center_np for k in shared]),
                np.array([gt_center[k] for k in shared]))
            scale_err_pct = 100.0 * (res.scale - s_ref) / s_ref
            con.print(f"[bold]scale vs ground truth[/bold] {res.scale:.6g} vs {s_ref:.6g} "
                      f"mm/unit -> [{'green' if abs(scale_err_pct) < 0.05 else 'yellow'}]"
                      f"{scale_err_pct:+.5f}%[/] "
                      f"[dim](truth-track fit residual {rms_ref:.3f} mm)[/dim]")

    # -- dimensions ---------------------------------------------------------
    mstats: dict[str, Any] | None = None
    with st("recover dimensions"):
        if recon_mode == "objcap" and model_files:
            stl = out / "synth_scan.stl"
            m, mstats = mesh_mod.condition(model_files[0], res, spec, stl,
                                           allow_reject=True)
            recovered = np.array(sorted(mstats["obb_mm"], reverse=True), dtype=float)
            ref = np.array(sorted(expected_dims.tolist(), reverse=True), dtype=float)
            con.print(f"[green]wrote[/green] {stl}")
        else:
            import trimesh

            truth_mesh = trimesh.load(truth["object"]["mesh_stl"], process=True)
            v_world = np.asarray(truth_mesh.vertices, dtype=float)
            # world -> raw OpenCV board frame -> scan2print BOARD frame
            v_board = charuco.cv_points_to_board(v_world * np.array([1.0, -1.0, -1.0]), spec)
            # The ground-truth camera centres are written in NOMINAL board units
            # (what a correct pipeline reports given the board it was told about),
            # while the truth mesh is in physical millimetres. With a mis-printed
            # board those differ by 1/board_scale, and the mesh has to live in the
            # same frame as the cameras or the comparison is meaningless.
            v_board = v_board / board_scale
            v_recon = (Q_true.T @ (v_board - t_true).T).T / s_true
            back = scale_mod.apply_to_points(v_recon, res, allow_reject=True)
            recovered = back.max(axis=0) - back.min(axis=0)
            ref = expected_dims
        err = recovered - ref
        pct = 100.0 * err / ref

    con.rule("[bold]accuracy vs ground truth", align="left")
    acc = Table(box=ROUNDED, title=f"recovered dimensions  (recon = {recon_mode})",
                title_style="bold", expand=False)
    acc.add_column("axis", style="cyan")
    acc.add_column("true mm", justify="right")
    acc.add_column("recovered mm", justify="right")
    acc.add_column("error mm", justify="right")
    acc.add_column("error %", justify="right")
    for i, name in enumerate(("d0", "d1", "d2")):
        style = "green" if abs(pct[i]) < 0.25 else "yellow" if abs(pct[i]) < 1.0 else "red"
        acc.add_row(name, f"{ref[i]:.4f}", f"{recovered[i]:.4f}",
                    f"[{style}]{err[i]:+.4f}[/{style}]", f"[{style}]{pct[i]:+.4f}[/{style}]")
    con.print(acc)
    if board_scale != 1.0:
        con.print(f"[yellow]board_scale={board_scale}: the board was PHYSICALLY "
                  f"{board_scale * 100:.4f}% of nominal while the pipeline believed "
                  f"{spec.square_mm:g} mm squares. 'true mm' above is what a correct "
                  f"pipeline should report given that mis-scaled board "
                  f"({true_dims.tolist()} mm is the object's actual size). This error is "
                  f"invisible to the fit residual -- only calipers catch it.[/yellow]")

    witnesses = verify.assess(
        res, reproj_mean_px=float(rms.mean()), reproj_max_px=float(rms.max()),
        frames_solved=len(metric), frames_total=int(truth["frames"]["count"]),
        intrinsics=intr)
    summary = verify.report(res, mstats, spec, console=con, witnesses=witnesses,
                            extras={"camera-centre error": f"mean {errs.mean():.4f} mm, "
                                                           f"max {errs.max():.4f} mm",
                                    "recon source": recon_mode})
    con.print("[dim]note: the intrinsics witness reads 'unknown provenance' here because "
              "these are the renderer's exact camera parameters -- neither a measured "
              "calibration nor an estimate. In the synthetic scene the camera model is "
              "ground truth; on a real scan it would not be.[/dim]")
    con.print(st.table())

    result = {
        "schema": "scan2print.synth_report/1",
        "recon": recon_mode, "out": str(out),
        "scale_error_pct_vs_truth": scale_err_pct,
        "true_dims_mm": true_dims.tolist(),
        "reference_dims_mm": ref.tolist(),
        "recovered_dims_mm": [round(float(v), 5) for v in recovered],
        "error_mm": [round(float(v), 5) for v in err],
        "error_pct": [round(float(v), 5) for v in pct],
        "pose_centre_error_mm": {"mean": float(errs.mean()), "max": float(errs.max())},
        "reproj_rms_px": {"mean": float(rms.mean()), "max": float(rms.max())},
        "scale": summary, "timing_s": st.times,
        "caveat": synth_mod.CAVEAT,
    }
    (out / "synth_report.json").write_text(json.dumps(_jsonable(result), indent=2))
    con.print(f"[dim]report: {out / 'synth_report.json'}[/dim]")

    worst = float(np.max(np.abs(pct)))
    tol = args.tol_pct

    # WHAT IS ASSERTED, AND WHY IT DIFFERS BY MODE.
    # ideal recon : the only error source is board metrology + the similarity fit,
    #               so the recovered dimensions themselves are the pass criterion.
    # objcap      : the mesh also carries Object Capture's own surface error and
    #               loses whatever sits below the ground cut, neither of which this
    #               program can fix. The number that IS ours is the recovered
    #               scale, measured against the same reconstruction fitted to the
    #               ground-truth camera track -- so that is what gets asserted, and
    #               the dimensions are reported honestly alongside it.
    #
    # EXIT CODES. "PASS" is reserved for "every number checked is in tolerance",
    # because this is the command the demo runs and a green banner over a 4.7%
    # dimensional error is exactly the kind of confident wrongness this tool is
    # supposed to catch.
    #   0  everything in tolerance
    #   1  the scale -- this program's own job -- is out of tolerance
    #   3  scale correct, mesh dimensions out of tolerance (Object Capture's job)
    #   4  a deliberately mis-printed board reproduced its expected error
    if recon_mode == "objcap" and scale_err_pct is not None:
        scale_ok = abs(scale_err_pct) <= tol
        mesh_ok = worst <= tol
        body = (
            f"SCALE (this program's job): {scale_err_pct:+.5f}% vs ground truth "
            f"(tolerance {tol:g}%) -- {'in tolerance' if scale_ok else 'OUT OF TOLERANCE'}.\n"
            f"MESH (Object Capture's job): worst dimension off by {worst:.3f}%, "
            f"{np.abs(err).max():.3f} mm (tolerance {tol:g}%) -- "
            f"{'in tolerance' if mesh_ok else 'OUT OF TOLERANCE'}.\n\n"
            "Those are different failures, and the distinction is real: the mesh error "
            "is surface reconstruction plus the geometry lost below the --z-tol ground "
            "cut (default 1.0 mm above the board); it is not a scale error and "
            "re-shooting will not remove all of it. Object Capture cannot see the face "
            "the object is resting on.\n\n"
            "It is still a dimensional error in the part you would print, and a replica "
            "you form leather around is only as good as its worst dimension -- so this "
            "does not get to be called a pass.\n"
            + synth_mod.CAVEAT)
        if scale_ok and mesh_ok:
            head, style, code = "PASS -- ", "bold green", 0
        elif scale_ok:
            head, style, code = ("SCALE CORRECT, MESH OUT OF TOLERANCE -- ",
                                 "bold yellow", 3)
        else:
            head, style, code = "FAIL (scale) -- ", "bold red", 1
        con.print(Panel(Text(head + body, style=style),
                        border_style=style.split()[-1], expand=False))
        return code

    if worst <= tol:
        con.print(Panel(Text(
            f"PASS -- worst dimensional error {worst:.4f}% (tolerance {tol:g}%).\n"
            + synth_mod.CAVEAT, style="bold green"), border_style="green", expand=False))
        return 0
    if board_scale != 1.0:
        con.print(Panel(Text(
            f"EXPECTED ERROR REPRODUCED (not a pass) -- worst dimensional error "
            f"{worst:.4f}% (tolerance {tol:g}%).\n--board-scale was {board_scale:g}, so the "
            "board was deliberately mis-printed and the pipeline was never told. This is "
            "the error mode the fit residual CANNOT see, and only calipers catch it. "
            "Exit code 4: the demonstration worked, the dimensions are still wrong.",
            style="bold yellow"), border_style="yellow", expand=False))
        return 4
    con.print(Panel(Text(
        f"OUT OF TOLERANCE -- worst dimensional error {worst:.4f}% "
        f"(tolerance {tol:g}%).", style="bold yellow"),
        border_style="yellow", expand=False))
    return 1


# ==========================================================================
# argument parsing
# ==========================================================================

def _add_board_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("board geometry (defaults: 7x5 @ 30 mm squares)")
    g.add_argument("--board-json", metavar="PATH",
                   help="BoardSpec json, or a target sidecar written by `target`")
    g.add_argument("--squares-x", type=int, default=None)
    g.add_argument("--squares-y", type=int, default=None)
    g.add_argument("--square-mm", type=float, default=None,
                   help="MEASURED printed square size in mm -- the single most "
                        "important number in this program")
    g.add_argument("--marker-mm", type=float, default=None)
    g.add_argument("--dict-name", default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="scan2print",
        description="Phone video of an object on a ChArUco board -> true-millimetre "
                    "printable mesh -> sliced gcode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with:  scan2print doctor    then    scan2print target",
    )
    ap.add_argument("--version", action="version", version="scan2print 0.1.0")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # --- run ---
    r = sub.add_parser("run", help="full pipeline on a scan video",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r.add_argument("video", help="the scan video (mov/mp4)")
    _add_board_args(r)
    r.add_argument("--out", default=None, help="working dir (default out/<video-stem>)")
    r.add_argument("--frames", type=int, default=config.TARGET_FRAMES,
                   help="how many frames to feed the reconstruction")
    r.add_argument("--detail", default="medium",
                   choices=["preview", "reduced", "medium", "full", "raw"],
                   help="Object Capture mesh detail")
    r.add_argument("--intrinsics", default=None,
                   help="camera calibration json (default: profiles/intrinsics.json)")
    r.add_argument("--no-intrinsics", action="store_true",
                   help="ignore the default calibration profile")
    r.add_argument("--material", default="PLA", help="filament material")
    r.add_argument("--layer-mm", type=float, default=None, help="layer height")
    r.add_argument("--process", default=None, help="Orca process preset NAME")
    r.add_argument("--filament", default=None, help="Orca filament preset NAME")
    r.add_argument("--z-tol", type=float, default=1.0,
                   help="mm above the board plane to cut the ground away")
    r.add_argument("--max-faces", type=int, default=200_000)
    r.add_argument("--no-flatten-base", action="store_true",
                   help="do not drop the ground cut back to z=0 (use when the object "
                        "is not resting on the board)")
    r.add_argument("--no-compensate", action="store_true",
                   help="skip filament shrinkage pre-scaling")
    r.add_argument("--assume-shrink", action="store_true",
                   help="pre-scale by the UNMEASURED literature shrinkage when this "
                        "material has no gauge-cube calibration (default: compensate "
                        "nothing and say so)")
    r.add_argument("--caliper-mm", type=float, default=None,
                   help="a real measured dimension, for the cross-check -- the only "
                        "check that can catch a mis-measured board")
    r.add_argument("--caliper-axis", default=None,
                   help="WHICH dimension --caliper-mm is: longest/middle/shortest, or "
                        "x/y/z for a print axis. Required with --caliper-mm; 'nearest' "
                        "opts into the old auto-match, which certifies nothing")
    r.add_argument("--dry-run", action="store_true", help="stop before slicing")
    r.add_argument("--yes", action="store_true",
                   help="skip the interactive confirmation (still refuses a rejected "
                        "fit, a failed caliper check, or an unverified scale unless "
                        "--force)")
    r.add_argument("--force", action="store_true",
                   help="allow proceeding on a REJECTED scale verdict, a FAILED caliper "
                        "cross-check, or a scale never checked against the object")
    r.add_argument("--reuse", action="store_true",
                   help="reuse existing frames/reconstruction in the working dir")
    r.add_argument("--keep-intermediates", action="store_true",
                   help="keep candidate frames and other scratch files")
    r.set_defaults(func=cmd_run)

    # --- target ---
    t = sub.add_parser("target", help="generate the printable ChArUco target",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_board_args(t)
    t.add_argument("--dpi", type=int, default=300)
    t.add_argument("--paper", default="auto", choices=["auto", "letter", "a4", "a3"])
    t.add_argument("--margin-mm", type=float, default=10.0)
    t.add_argument("--no-pdf", action="store_true")
    t.add_argument("--out", default=None, help="output PNG path")
    t.set_defaults(func=cmd_target)

    # --- calibrate ---
    c = sub.add_parser("calibrate", help="camera intrinsics from board photographs",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    c.add_argument("image_dir", help="folder of photos of the board from many angles")
    _add_board_args(c)
    c.add_argument("--out", default=None,
                   help="where to write intrinsics (default profiles/intrinsics.json)")
    c.add_argument("--fix-k3", action="store_true",
                   help="fix k3=0; recommended for phone lenses")
    c.set_defaults(func=cmd_calibrate)

    # --- synth ---
    s = sub.add_parser("synth", help="synthetic ground-truth accuracy demo",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s.add_argument("--out", default=None)
    s.add_argument("--frames", type=int, default=30)
    s.add_argument("--object-mm", type=float, nargs=3, default=[50.0, 30.0, 20.0],
                   metavar=("X", "Y", "Z"))
    s.add_argument("--resolution", type=int, nargs=2, default=[1440, 1080],
                   metavar=("W", "H"))
    s.add_argument("--recon", default="auto", choices=["auto", "objcap", "ideal"],
                   help="auto = real Object Capture if built, else the ideal track")
    s.add_argument("--detail", default="medium",
                   choices=["preview", "reduced", "medium", "full", "raw"])
    s.add_argument("--board-scale", type=float, default=1.0,
                   help="model a printer that scaled the target (e.g. 0.999)")
    s.add_argument("--tol-pct", type=float, default=0.25,
                   help="pass/fail tolerance on the worst dimension")
    s.add_argument("--reuse", action="store_true", help="reuse an existing render")
    s.add_argument("--verbose", action="store_true", help="show Blender output")
    s.set_defaults(func=cmd_synth)

    # --- doctor ---
    d = sub.add_parser("doctor", help="check tools, packages and project assets")
    d.set_defaults(func=cmd_doctor)

    # --- cube ---
    q = sub.add_parser("cube", help="printer shrinkage calibration cube",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    q.add_argument("--size", type=float, default=30.0, help="nominal cube size in mm")
    q.add_argument("--out", default=None, help="STL path")
    q.add_argument("--material", default="PLA")
    q.add_argument("--measure", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="record caliper readings of the printed cube instead of "
                        "emitting a new one")
    q.add_argument("--slice", action="store_true", help="also slice the cube")
    q.set_defaults(func=cmd_cube)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        console.print(Panel(Text(str(exc), style="bold red"), title="scan2print: failed",
                            border_style="red", expand=False))
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
