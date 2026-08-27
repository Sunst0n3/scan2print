"""Video -> sharp, evenly-distributed still frames for Object Capture.

Motion blur is the number-one cause of a failed photogrammetry run, so this
module is deliberately picky: it over-extracts candidates, scores every one
for sharpness, and then keeps the best frame from each slice of the orbit.

Frame *names* produced here (frame_00001.png ...) are the join key for the
rest of the pipeline: pose.py stamps a FramePose onto each name and the
Object Capture wrapper reports a ReconPose under the same name. Renaming or
re-ordering them breaks the scale solve, so the numbering is strictly
temporal and the mapping back to source timestamps is recorded in
frames_manifest.json.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from scan2print import config
from scan2print.types import BoardSpec, Intrinsics

__all__ = [
    "FrameError",
    "probe",
    "extract",
    "sharpness",
    "select_sharp",
    "prepare",
]

_console = Console()

# Over-extraction factor. We score 3x more candidates than we keep so that
# every temporal bucket has real alternatives to choose between; with a 1x
# extraction there is nothing to reject and the blur filter is decorative.
CANDIDATE_FACTOR = 3

# A temporal bucket whose every frame fell below the blur floor still gets one
# frame back -- its own best -- provided that frame reaches this fraction of
# the clip's median sharpness. Rationale: a merely soft view of an orbit
# sector is far less damaging to the reconstruction than no view at all, but a
# genuinely smeared one is worse than nothing. Real motion blur drops the
# Laplacian variance by an order of magnitude, so 0.5x median cleanly
# separates "slightly soft" from "smeared".
SALVAGE_FRACTION = 0.5


class FrameError(RuntimeError):
    """Raised when frame extraction or filtering cannot produce a usable set.

    This is a metrology pipeline: a short, blurry, or gappy frame set silently
    poisons the reconstruction and therefore the measured scale. Better to
    stop here with a specific message than to hand Object Capture garbage.
    """


# --------------------------------------------------------------------------
# subprocess plumbing
# --------------------------------------------------------------------------


def _require_tools() -> None:
    for name, path in (("ffmpeg", config.FFMPEG_BIN), ("ffprobe", config.FFPROBE_BIN)):
        if not Path(path).exists():
            raise FrameError(f"{name} not found at {path!s}; fix config.{name.upper()}_BIN")


def _run(cmd: Sequence[str], what: str) -> str:
    """Run a tool with an argv list (never shell=True) and return stdout."""
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise FrameError(
            f"{what} failed (exit {proc.returncode}):\n  "
            + "\n  ".join(tail or ["<no output>"])
        )
    return proc.stdout


# --------------------------------------------------------------------------
# 1. probe
# --------------------------------------------------------------------------


def _parse_rate(text: str | None) -> float:
    """ffprobe reports frame rates as the rational string 'num/den'."""
    if not text:
        return 0.0
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            n, d = float(num), float(den)
        except ValueError:
            return 0.0
        return n / d if d else 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _rotation_of(stream: dict[str, Any]) -> float:
    """Recover display rotation in degrees, normalised to [0, 360).

    iPhone/QuickTime files store orientation as a 3x3 display matrix in
    stream side data, NOT as pixels: the encoded raster is landscape even for
    a video shot in portrait. ffprobe surfaces it as side_data_list ->
    'rotation' (older muxers used the tags.rotate string instead). Ignoring it
    yields sideways frames, which Object Capture will happily reconstruct into
    a sideways object.
    """
    rot: float | None = None
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rot = float(sd["rotation"])
                break
            except (TypeError, ValueError):
                pass
    if rot is None:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag is not None:
            try:
                rot = float(tag)
            except (TypeError, ValueError):
                rot = None
    if rot is None:
        return 0.0
    return float(round(rot) % 360)


def probe(video: Path) -> dict[str, Any]:
    """Return container/stream facts for `video`.

    Keys: path, duration_s, fps, width, height (coded raster),
    display_width/display_height (after rotation), rotation_deg, codec,
    pix_fmt, n_frames (exact when the container says so, else estimated),
    n_frames_estimated (bool).
    """
    _require_tools()
    video = Path(video).expanduser().resolve()
    if not video.exists():
        raise FrameError(f"video not found: {video}")

    out = _run(
        [
            config.FFPROBE_BIN,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video),
        ],
        "ffprobe",
    )
    try:
        meta = json.loads(out)
    except json.JSONDecodeError as exc:
        raise FrameError(f"could not parse ffprobe JSON for {video}: {exc}") from exc

    vstreams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise FrameError(f"no video stream in {video}")
    st = vstreams[0]

    fps = _parse_rate(st.get("avg_frame_rate")) or _parse_rate(st.get("r_frame_rate"))
    if fps <= 0:
        raise FrameError(f"could not determine frame rate for {video}")

    duration = 0.0
    for src in (st.get("duration"), (meta.get("format") or {}).get("duration")):
        try:
            duration = float(src)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break

    n_frames = 0
    estimated = False
    for key in ("nb_frames", "nb_read_frames"):
        try:
            n_frames = int(st.get(key))
        except (TypeError, ValueError):
            continue
        if n_frames > 0:
            break
    if n_frames <= 0:
        if duration <= 0:
            raise FrameError(
                f"{video} reports neither duration nor frame count; the file is "
                "probably truncated or still being written"
            )
        n_frames = int(round(duration * fps))
        estimated = True
    if duration <= 0:
        duration = n_frames / fps

    w, h = int(st.get("width", 0)), int(st.get("height", 0))
    rot = _rotation_of(st)
    swap = round(rot) % 180 == 90
    info: dict[str, Any] = {
        "path": str(video),
        "duration_s": round(duration, 4),
        "fps": round(fps, 6),
        "width": w,
        "height": h,
        "display_width": h if swap else w,
        "display_height": w if swap else h,
        "rotation_deg": rot,
        "codec": st.get("codec_name", "?"),
        "pix_fmt": st.get("pix_fmt", "?"),
        "n_frames": n_frames,
        "n_frames_estimated": estimated,
    }
    return info


# --------------------------------------------------------------------------
# 2. extract
# --------------------------------------------------------------------------


def _ffmpeg_with_progress(cmd: Sequence[str], duration_s: float, label: str) -> None:
    """Run ffmpeg, driving a rich bar off its -progress stream."""
    argv = [str(c) for c in cmd]
    with tempfile.TemporaryFile("w+") as errf:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=errf, text=True, bufsize=1
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=_console,
            transient=True,
        ) as bar:
            task = bar.add_task(label, total=max(duration_s, 0.001))
            assert proc.stdout is not None
            for line in proc.stdout:
                key, _, val = line.strip().partition("=")
                if key == "out_time_us" and val.strip().lstrip("-").isdigit():
                    bar.update(task, completed=min(int(val) / 1e6, duration_s))
                elif key == "progress" and val.strip() == "end":
                    bar.update(task, completed=duration_s)
        proc.wait()
        if proc.returncode != 0:
            errf.seek(0)
            tail = errf.read().strip().splitlines()[-15:]
            raise FrameError(
                f"ffmpeg failed (exit {proc.returncode}):\n  "
                + "\n  ".join(tail or ["<no stderr>"])
            )


def extract(
    video: Path,
    out_dir: Path,
    target: int = config.TARGET_FRAMES,
    info: dict[str, Any] | None = None,
) -> list[Path]:
    """Extract ~CANDIDATE_FACTOR * target candidate frames evenly across `video`.

    Output is lossless 8-bit PNG: Object Capture keys off fine texture detail
    and JPEG ringing around high-contrast edges (exactly what a ChArUco board
    is made of) both softens the sharpness metric and degrades feature
    matching. `out_dir` is wiped and recreated.
    """
    _require_tools()
    video = Path(video).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if info is None:
        info = probe(video)

    duration = float(info["duration_s"])
    src_fps = float(info["fps"])
    total = int(info["n_frames"])
    if duration <= 0 or total <= 0:
        raise FrameError(f"{video} has no usable frames (duration={duration}, n={total})")

    n_want = min(CANDIDATE_FACTOR * int(target), total)
    if n_want < 1:
        raise FrameError(f"target={target} yields no candidate frames")

    # Sample rate for the fps filter. Clamp to the source rate so ffmpeg never
    # duplicates frames -- a duplicated frame is a fake "extra view" that adds
    # no information but does dilute the per-bucket competition.
    rate = min(n_want / duration, src_fps)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "cand_%05d.png"

    cmd = [
        config.FFMPEG_BIN,
        "-hide_banner", "-nostdin", "-y",
        "-loglevel", "error",
        "-nostats",
        "-progress", "pipe:1",
        "-i", str(video),
        # No explicit transpose: ffmpeg >= 2.7 applies the display matrix
        # automatically (autorotate). probe() records the rotation so
        # _verify_orientation() below can catch it if that ever changes.
        "-vf", f"fps={rate:.9f}",
        # Force 8-bit RGB. iPhone HEVC is frequently 10-bit HLG; without this
        # ffmpeg emits 16-bit PNGs that other tools read inconsistently.
        "-pix_fmt", "rgb24",
        "-compression_level", "1",   # lossless either way, just faster
        "-an", "-sn", "-dn",
        "-map", "0:v:0",
        str(pattern),
    ]
    _ffmpeg_with_progress(cmd, duration, f"extracting {n_want} candidates")

    paths = sorted(out_dir.glob("cand_*.png"))
    if not paths:
        raise FrameError(
            f"ffmpeg produced no frames from {video}; the stream may be corrupt "
            "or use a codec this build cannot decode"
        )
    _verify_orientation(paths[0], info)
    return paths


def _verify_orientation(sample: Path, info: dict[str, Any]) -> None:
    """Fail loudly if ffmpeg's autorotate did not do what probe() predicted."""
    img = cv2.imread(str(sample), cv2.IMREAD_COLOR)
    if img is None:
        raise FrameError(f"could not read extracted frame {sample}")
    h, w = img.shape[:2]
    exp = (int(info["display_width"]), int(info["display_height"]))
    if (w, h) != exp:
        raise FrameError(
            f"extracted frame is {w}x{h} but probe predicted {exp[0]}x{exp[1]} "
            f"(rotation={info['rotation_deg']} deg). ffmpeg autorotate did not "
            "apply the display matrix; frames would be sideways."
        )


# --------------------------------------------------------------------------
# 3. sharpness
# --------------------------------------------------------------------------


def sharpness(img_path: Path) -> float:
    """Variance of the Laplacian on greyscale -- higher is sharper.

    Computed at full resolution: downscaling hides exactly the fine-detail
    loss we are trying to detect. The number is only meaningful *relative to
    other frames of the same scene at the same resolution*, which is all
    select_sharp() ever uses it for.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FrameError(f"could not read image {img_path}")
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _score_all(paths: Sequence[Path]) -> list[float]:
    scores: list[float] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    ) as bar:
        task = bar.add_task("scoring sharpness", total=len(paths))
        for p in paths:
            scores.append(sharpness(p))
            bar.advance(task)
    return scores


# --------------------------------------------------------------------------
# 4. select_sharp
# --------------------------------------------------------------------------


def select_sharp(
    paths: Sequence[Path],
    target: int,
    reject_pct: float = config.BLUR_REJECT_PERCENTILE,
    scores: Sequence[float] | None = None,
    report: dict[str, Any] | None = None,
) -> list[Path]:
    """Keep ~target frames: sharpest-per-temporal-bucket, not sharpest-overall.

    WHY BUCKETS INSTEAD OF A GLOBAL TOP-N:
    A global "keep the N sharpest" ranking is a trap. Sharpness correlates
    strongly with lighting and with how fast the operator happened to be
    walking, both of which vary systematically around the orbit. Take the top
    N globally and you get a dense clump of frames from the bright, slow side
    of the object and *zero* frames from the dim or hurried side. Object
    Capture does not error on that -- it quietly reconstructs a partial hull,
    and the missing angular sector shows up as a warped surface and a biased
    camera track, which then corrupts the metric scale fit. Even angular
    coverage beats peak per-frame sharpness every time.

    So: split the candidates into `target` equal temporal buckets (time is a
    proxy for orbit angle for a steady walk-around) and take the sharpest
    frame in each.

    Two floors, not one. The `reject_pct` percentile floor is *relative* to
    this clip, so on a uniformly good video it still discards a quarter of the
    frames -- and if that quarter happens to be temporally clustered it
    deletes whole sectors, reintroducing the very bias buckets exist to
    prevent. So a bucket emptied by the percentile floor gets its own best
    frame back if that frame still clears SALVAGE_FRACTION x the clip's median
    sharpness (soft is better than absent). Only a bucket that fails even the
    salvage floor is left as a real gap, and gaps are reported loudly.

    Any shortfall against `target` is topped up round-robin by bucket rank --
    rank 2 of every bucket before rank 3 of any -- so the fill stays even
    instead of re-clumping into whichever sector happened to be best lit.
    """
    paths = [Path(p) for p in paths]
    n = len(paths)
    if n == 0:
        raise FrameError("select_sharp() got no candidate frames")
    target = max(1, int(target))

    vals = list(scores) if scores is not None else _score_all(paths)
    if len(vals) != n:
        raise FrameError(f"score/path length mismatch: {len(vals)} vs {n}")
    arr = np.asarray(vals, dtype=np.float64)

    floor = float(np.percentile(arr, float(reject_pct))) if reject_pct > 0 else -math.inf
    # Never let the salvage gate be stricter than the primary gate.
    salvage = min(SALVAGE_FRACTION * float(np.median(arr)), floor)

    n_buckets = max(1, min(target, n))
    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    for i in range(n):
        buckets[i * n_buckets // n].append(i)

    ranked: list[list[int]] = []      # per bucket, above-floor frames, sharpest-first
    chosen: list[int] = []
    salvaged: list[int] = []
    gaps: list[int] = []              # bucket indices with no usable frame at all
    for b, idxs in enumerate(buckets):
        good = sorted((i for i in idxs if arr[i] >= floor), key=lambda i: arr[i],
                      reverse=True)
        ranked.append(good)
        if good:
            chosen.append(good[0])
            continue
        best = max(idxs, key=lambda i: arr[i])
        if arr[best] >= salvage:
            chosen.append(best)
            salvaged.append(best)
        else:
            gaps.append(b)

    # Top-up: rank 2 of every bucket, then rank 3, ... never all extras from
    # one bucket, so the temporal spread survives the fill. When a rank pass
    # can only be partially applied (the target is reached mid-pass) the
    # buckets it draws from are spread evenly across the orbit rather than
    # taken in bucket order -- otherwise every partial pass front-loads the
    # start of the clip, which is the same clumping bug in a new disguise.
    rank = 1
    while len(chosen) < target:
        elig = [b for b in range(n_buckets) if len(ranked[b]) > rank]
        if not elig:
            break
        need = target - len(chosen)
        if need < len(elig):
            picks = np.linspace(0, len(elig) - 1, need).round().astype(int)
            elig = [elig[i] for i in dict.fromkeys(picks.tolist())]
        chosen.extend(ranked[b][rank] for b in elig)
        rank += 1

    chosen.sort()
    kept = [paths[i] for i in chosen]
    salv_set = {paths[i] for i in salvaged}

    _console.print(
        f"[green]selected[/green] {len(kept)} / {n} candidates "
        f"({n - len(kept)} rejected; blur floor={floor:.1f} at p{reject_pct:g}, "
        f"salvage floor={salvage:.1f}; {n_buckets} temporal buckets, "
        f"{len(salvaged)} salvaged, {len(gaps)} gap(s))"
    )
    if gaps:
        _console.print(
            f"[yellow]warning:[/yellow] {len(gaps)} of {n_buckets} orbit sectors had "
            "no usable frame (all below the salvage floor) -- those viewing angles "
            "are unrepresented and the reconstruction may be incomplete there"
        )
    if report is not None:
        report.update(
            blur_floor=floor,
            salvage_floor=salvage,
            n_buckets=n_buckets,
            n_salvaged=len(salvaged),
            n_gap_buckets=len(gaps),
            gap_buckets=gaps,
            salvaged_paths=salv_set,
        )
    return kept


# --------------------------------------------------------------------------
# 5. prepare
# --------------------------------------------------------------------------


def _undistort_all(
    srcs: Sequence[Path], dsts: Sequence[Path], intr: Intrinsics
) -> Intrinsics:
    """Rectify frames with cv2.undistort.

    WHY: Object Capture fits its own camera model and is happiest with images
    that already match a clean pinhole. Feeding it rectified frames removes
    lens distortion as an error source, and -- more importantly for this
    pipeline -- makes the RECON camera track geometrically consistent with the
    ChArUco-derived BOARD track, since both then describe the same undistorted
    camera. Residual distortion shows up as a bowed camera path and inflates
    ScaleResult.rms_mm.

    NOTE FOR pose.py: the frames written by prepare() are already rectified,
    so solvePnP on them must be given D = zeros, not intr.D. The manifest
    records `undistorted: true` and the effective intrinsics.
    """
    first = cv2.imread(str(srcs[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FrameError(f"could not read {srcs[0]}")
    h, w = first.shape[:2]
    use = intr if (intr.width == w and intr.height == h) else intr.scaled_to(w, h)
    if intr.width not in (0, w) or intr.height not in (0, h):
        _console.print(
            f"[yellow]note:[/yellow] intrinsics calibrated at "
            f"{intr.width}x{intr.height}, rescaled to {w}x{h}"
        )
    K, D = use.K, use.D

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    ) as bar:
        task = bar.add_task("undistorting", total=len(srcs))
        for s, d in zip(srcs, dsts):
            img = first if s == srcs[0] else cv2.imread(str(s), cv2.IMREAD_COLOR)
            if img is None:
                raise FrameError(f"could not read {s}")
            if not cv2.imwrite(str(d), cv2.undistort(img, K, D)):
                raise FrameError(f"could not write {d}")
            bar.advance(task)
    return use


def prepare(
    video: Path,
    out_dir: Path,
    spec: BoardSpec | None = None,
    intr: Intrinsics | None = None,
    target: int = config.TARGET_FRAMES,
    reject_pct: float = config.BLUR_REJECT_PERCENTILE,
    keep_candidates: bool = False,
) -> list[Path]:
    """probe -> extract -> select -> (undistort) -> out_dir/frames/frame_NNNNN.png

    `out_dir/frames` and `out_dir/candidates` are deleted and recreated so a
    rerun can never mix in stale frames from a previous video. Returns the
    final frame paths in temporal order. Also writes out_dir/frames_manifest.json
    mapping each output name back to its source candidate and sharpness score.

    Raises FrameError if fewer than config.MIN_FRAMES survive filtering.
    """
    video = Path(video).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    target = int(max(config.MIN_FRAMES, min(int(target), config.MAX_FRAMES)))

    info = probe(video)
    tbl = Table(title=f"probe: {video.name}", show_header=False, title_style="bold")
    for k in (
        "duration_s", "fps", "width", "height", "display_width", "display_height",
        "rotation_deg", "codec", "pix_fmt", "n_frames", "n_frames_estimated",
    ):
        tbl.add_row(k, str(info[k]))
    _console.print(tbl)

    cand_dir = out_dir / "candidates"
    frames_dir = out_dir / "frames"
    for d in (cand_dir, frames_dir):
        if d.exists():
            shutil.rmtree(d)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cands = extract(video, cand_dir, target=target, info=info)
    _console.print(f"[green]extracted[/green] {len(cands)} candidate frames")

    scores = _score_all(cands)
    report: dict[str, Any] = {}
    keep = select_sharp(cands, target, reject_pct=reject_pct, scores=scores,
                        report=report)
    salvaged = report.pop("salvaged_paths", set())

    if len(keep) < config.MIN_FRAMES:
        raise FrameError(
            f"only {len(keep)} frames survived blur filtering, need at least "
            f"{config.MIN_FRAMES} (config.MIN_FRAMES). Source: {video.name}, "
            f"{len(cands)} candidates from {info['duration_s']:.1f}s. "
            "Reshoot with more light / a slower orbit, or shoot for longer."
        )

    score_by_path = {p: s for p, s in zip(cands, scores)}
    idx_by_path = {p: i for i, p in enumerate(cands)}
    finals = [frames_dir / f"frame_{i:05d}.png" for i in range(1, len(keep) + 1)]

    eff_intr: Intrinsics | None = None
    if intr is not None:
        eff_intr = _undistort_all(keep, finals, intr)
    else:
        for s, d in zip(keep, finals):
            shutil.copy2(s, d)

    # Candidate index -> approximate source timestamp. Candidates were sampled
    # at a constant rate across the whole clip, so index maps linearly to time.
    step = info["duration_s"] / max(len(cands), 1)
    manifest = {
        "video": str(video),
        "probe": info,
        "target": target,
        "reject_percentile": reject_pct,
        "n_candidates": len(cands),
        "n_selected": len(keep),
        "selection": report,
        "undistorted": eff_intr is not None,
        "intrinsics": (
            {"fx": eff_intr.fx, "fy": eff_intr.fy, "cx": eff_intr.cx, "cy": eff_intr.cy,
             "width": eff_intr.width, "height": eff_intr.height,
             "dist": [0.0] * 5, "dist_applied": list(eff_intr.dist)}
            if eff_intr is not None else None
        ),
        "board": spec.to_json() if spec is not None else None,
        "frames": [
            {
                "name": f.name,
                "source": s.name,
                "t_s": round((idx_by_path[s] + 0.5) * step, 4),
                "sharpness": round(score_by_path[s], 3),
                "salvaged": s in salvaged,
            }
            for f, s in zip(finals, keep)
        ],
    }
    (out_dir / "frames_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not keep_candidates:
        shutil.rmtree(cand_dir, ignore_errors=True)

    sel = np.array([score_by_path[p] for p in keep])
    _console.print(
        f"[bold green]prepared[/bold green] {len(finals)} frames -> {frames_dir} "
        f"(sharpness min={sel.min():.1f} med={np.median(sel):.1f} max={sel.max():.1f}"
        + (", undistorted" if eff_intr is not None else "") + ")"
    )
    return finals


def _cli(argv: Iterable[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract sharp frames from a scan video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--target", type=int, default=config.TARGET_FRAMES)
    ap.add_argument("--reject-pct", type=float, default=config.BLUR_REJECT_PERCENTILE)
    ap.add_argument("--intrinsics", type=Path, default=None)
    ap.add_argument("--keep-candidates", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)

    intr = Intrinsics.load(a.intrinsics) if a.intrinsics else None
    try:
        prepare(a.video, a.out_dir, intr=intr, target=a.target,
                reject_pct=a.reject_pct, keep_candidates=a.keep_candidates)
    except FrameError as exc:
        _console.print(f"[bold red]error:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
