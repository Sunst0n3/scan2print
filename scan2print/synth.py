"""Synthetic ground-truth validation harness.

Renders a Blender scene in which *every* physical dimension is known exactly:
a ChArUco board of the correct printed size, an object whose axis-aligned
bounding box is forced to an exact millimetre triple, and a camera orbiting on
a hemispherical path with exactly-known intrinsics and extrinsics. Running that
scene through the pipeline turns "does the scale look right?" into a number.

HONESTY NOTE -- WHAT THIS DOES AND DOES NOT PROVE
-------------------------------------------------
Blender output is a BEST CASE input, not a representative one. Specifically:

  * The renders are noise-free, perfectly focused, and have zero motion blur,
    zero rolling shutter, and zero auto-exposure/auto-white-balance drift.
  * The lens is an exact pinhole. Distortion coefficients are all zero, so
    this harness does not exercise the distortion half of the calibration at
    all. A real phone lens has meaningful radial distortion.
  * The board is perfectly flat and perfectly printed. A real printed target
    is scaled by the printer by a few tenths of a percent and is never flat;
    that error goes straight into the reported millimetres and is the single
    largest error source in real use. This harness CANNOT see it.
  * The object is rigid, opaque, matte and heavily textured -- ideal for
    photogrammetry. Shiny, dark, or featureless real objects are much harder.

So: a good number here proves the *geometry and arithmetic* of the pipeline are
correct (frame conventions, pose recovery, similarity fit, unit bookkeeping).
It does not prove the pipeline hits that accuracy on a phone video of a real
object. Treat this as a necessary condition, not a sufficient one. The only
honest end-to-end validation is a printed part measured with calipers.

FRAME CONVENTIONS (important, and empirically verified against OpenCV 5)
-----------------------------------------------------------------------
OpenCV's CharucoBoard object frame, confirmed by matching
``board.getChessboardCorners()`` against ``board.matchImagePoints()`` output on
a generated board image:

    board point (x, y, 0)  ->  board image pixel (x*ppmm - 0.5, y*ppmm - 0.5)

i.e. the origin is the OUTER top-left corner of the board (not the first inner
chessboard corner), +X runs right along the image and +Y runs DOWN the image.
Because that frame is right-handed and OpenCV cameras look down +Z_cam, the
board's +Z axis points AWAY from the camera, into the back of the board. The
camera therefore sits at NEGATIVE Z in the board frame.

(Note for the rest of the codebase: types.py currently documents +Z as pointing
"out of the board towards the camera". That is the opposite of what OpenCV 5
actually produces. This module follows OpenCV, because that is what solvePnP
will hand the pipeline at runtime.)

For a scene that is comprehensible to a human, Blender WORLD is a "physical"
frame with the board's printed face up (+Z) and cameras above it. The two are
related by a fixed 180-degree rotation about X:

    p_board = M @ p_world,   M = diag(1, -1, -1),   M == M^-1

Ground-truth poses are written in BOTH frames. The BOARD-frame ones are what
the pipeline should reproduce.

USAGE
-----
    from scan2print.synth import make_scene
    scene = make_scene(Path("out/synth"), object_mm=(50, 30, 20), n_frames=60)

This file is dual-mode: imported normally it is the host-side API, and it is
ALSO the script Blender executes (``blender --background --python synth.py --
spec.json``). Keeping both halves in one file avoids a second file drifting out
of sync with the spec schema.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:  # the Blender half
    import bpy  # type: ignore

    IN_BLENDER = True
except ModuleNotFoundError:
    IN_BLENDER = False


# ===========================================================================
# HOST SIDE -- everything below runs under the project venv (cv2 + numpy)
# ===========================================================================
if not IN_BLENDER:
    import cv2
    import numpy as np

    try:
        from . import config
        from .types import BoardSpec, Intrinsics
    except ImportError:  # running as a loose script
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scan2print import config  # type: ignore
        from scan2print.types import BoardSpec, Intrinsics  # type: ignore

    SCHEMA = "scan2print.synth/1"

    CAVEAT = (
        "Synthetic best-case render: no sensor noise, no defocus, no motion blur, "
        "no rolling shutter, zero lens distortion, and a perfectly flat perfectly "
        "scaled board. Validates pipeline geometry and unit bookkeeping only; it "
        "does not bound real-world accuracy on a phone video."
    )

    # -------------------------------------------------------------------
    # board texture
    # -------------------------------------------------------------------
    def _cv_dictionary(name: str):
        attr = getattr(cv2.aruco, name, None)
        if attr is None:
            raise ValueError(
                f"unknown ArUco dictionary {name!r}; "
                f"cv2.aruco has no attribute of that name"
            )
        return cv2.aruco.getPredefinedDictionary(attr)

    def _cv_board(spec: BoardSpec):
        return cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_mm,
            spec.marker_mm,
            _cv_dictionary(spec.dict_name),
        )

    def validate_board_image(
        spec: BoardSpec, img: "np.ndarray", px_per_mm: float, margin_px: int = 0
    ) -> dict[str, Any]:
        """Detect the board in its own texture and confirm the corners land where
        the board's object points say they should.

        This is the guard that stops a wrongly-scaled or mirrored texture from
        silently poisoning every measurement downstream.
        """
        board = _cv_board(spec)
        det = cv2.aruco.CharucoDetector(board)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _mk_corners, mk_ids = det.detectBoard(gray)
        if ch_corners is None or len(ch_corners) == 0:
            raise RuntimeError(
                "board texture failed self-detection: CharucoDetector found no "
                "corners in the image this module just generated. The texture is "
                "unusable; do not render with it."
            )
        objp, imgp = board.matchImagePoints(ch_corners, ch_ids)
        objp = objp.reshape(-1, 3)
        imgp = imgp.reshape(-1, 2)
        # board (x,y,0) -> pixel (x*ppmm - 0.5 + margin, y*ppmm - 0.5 + margin)
        pred = np.stack(
            [
                objp[:, 0] * px_per_mm - 0.5 + margin_px,
                objp[:, 1] * px_per_mm - 0.5 + margin_px,
            ],
            axis=1,
        )
        resid = float(np.abs(pred - imgp).max())
        if resid > 1.5:
            raise RuntimeError(
                f"board texture geometry check failed: max corner residual "
                f"{resid:.2f}px against the analytic position. The texture is not "
                f"laid out the way the detector expects (wrong scale, wrong "
                f"margin, or mirrored)."
            )
        return {
            "detected_corners": int(len(ch_corners)),
            "expected_corners": int(spec.n_corners),
            "detected_markers": int(0 if mk_ids is None else len(mk_ids)),
            "max_corner_residual_px": resid,
        }

    def board_texture(
        spec: BoardSpec,
        path: Path,
        px_per_mm: float = 10.0,
        margin_mm: float = 12.0,
        use_charuco_module: bool = False,
    ) -> dict[str, Any]:
        """Write the ChArUco board texture PNG and self-verify it detects.

        By default the texture is generated from the very same
        ``cv2.aruco.CharucoBoard`` instance the detector is constructed from,
        which is the strongest available guarantee that the synthetic board
        matches the detector's expectations.

        ``use_charuco_module=True`` routes through ``scan2print.charuco``
        instead (written by another agent). That path is still validated by
        ``validate_board_image`` before it is allowed to be used -- an
        unverified target is worse than no target in a metrology tool.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        margin_px = int(round(margin_mm * px_per_mm))
        w_px = int(round(spec.width_mm * px_per_mm))
        h_px = int(round(spec.height_mm * px_per_mm))
        source = "cv2.aruco.CharucoBoard.generateImage"
        notes: list[str] = []

        core = None
        if use_charuco_module:
            try:
                from . import charuco as _charuco  # type: ignore

                render_target = getattr(_charuco, "render_target", None)
                if render_target is None:
                    notes.append(
                        "scan2print.charuco exists but has no render_target(); "
                        "fell back to direct cv2 generation"
                    )
                else:
                    core = render_target(spec, px_per_mm=px_per_mm)  # type: ignore[call-arg]
                    if not isinstance(core, np.ndarray):
                        notes.append(
                            f"charuco.render_target returned {type(core).__name__}, "
                            f"not an ndarray; fell back to direct cv2 generation"
                        )
                        core = None
                    else:
                        source = "scan2print.charuco.render_target"
            except ImportError:
                notes.append(
                    "scan2print.charuco not importable yet; used direct cv2 generation"
                )
            except Exception as exc:  # noqa: BLE001 - want the real text
                notes.append(
                    f"scan2print.charuco.render_target raised "
                    f"{type(exc).__name__}: {exc}; fell back to direct cv2 generation"
                )
                core = None

        if core is None:
            core = _cv_board(spec).generateImage((w_px, h_px))

        if core.ndim == 3:
            core = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
        if core.shape[:2] != (h_px, w_px):
            raise RuntimeError(
                f"board image is {core.shape[1]}x{core.shape[0]}px but "
                f"{w_px}x{h_px}px was required for {px_per_mm} px/mm at "
                f"{spec.width_mm}x{spec.height_mm}mm"
            )

        padded = cv2.copyMakeBorder(
            core,
            margin_px,
            margin_px,
            margin_px,
            margin_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        info = validate_board_image(spec, padded, px_per_mm, margin_px)
        if not cv2.imwrite(str(path), padded):
            raise RuntimeError(f"cv2.imwrite failed for {path}")

        info.update(
            {
                "path": str(path),
                "px_per_mm": px_per_mm,
                "margin_mm": margin_mm,
                "margin_px": margin_px,
                "image_px": [int(padded.shape[1]), int(padded.shape[0])],
                "board_mm": [spec.width_mm, spec.height_mm],
                "source": source,
                "notes": notes,
            }
        )
        return info

    # -------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------
    def _run(cmd: list[str], what: str, quiet: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-40:])
            raise RuntimeError(
                f"{what} failed (exit {proc.returncode}).\n"
                f"command: {' '.join(cmd)}\n--- output tail ---\n{tail}"
            )
        if not quiet:
            print(proc.stdout)
        return proc

    def _ffprobe(path: Path) -> dict[str, Any]:
        proc = _run(
            [
                str(config.FFPROBE_BIN),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            "ffprobe",
        )
        raw = json.loads(proc.stdout)
        st = (raw.get("streams") or [{}])[0]
        fmt = raw.get("format") or {}
        return {
            "codec": st.get("codec_name"),
            "width": int(st.get("width", 0)),
            "height": int(st.get("height", 0)),
            "nb_frames": int(st.get("nb_frames") or 0),
            "frame_rate": st.get("r_frame_rate"),
            "pix_fmt": st.get("pix_fmt"),
            "duration_s": float(fmt.get("duration") or 0.0),
            "size_bytes": int(fmt.get("size") or 0),
        }

    def _intrinsics_from_optics(
        lens_mm: float, sensor_mm: float, width: int, height: int
    ) -> Intrinsics:
        """Exact pinhole intrinsics for a Blender camera with sensor_fit=HORIZONTAL.

        Blender puts the principal point at the continuous image centre
        (width/2, height/2). OpenCV indexes pixel *centres*, so the same point
        is at (width/2 - 0.5, height/2 - 0.5) in detector coordinates.
        """
        f_px = lens_mm / sensor_mm * width
        return Intrinsics(
            fx=f_px,
            fy=f_px,
            cx=width / 2.0 - 0.5,
            cy=height / 2.0 - 0.5,
            dist=[0.0] * 5,
            width=int(width),
            height=int(height),
            rms_px=0.0,
        )

    # -------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------
    def make_scene(
        out_dir: str | os.PathLike,
        object_mm: tuple[float, float, float] = (50.0, 30.0, 20.0),
        n_frames: int = 60,
        *,
        board: "BoardSpec | None" = None,
        resolution: tuple[int, int] = (1920, 1440),
        lens_mm: float = 35.0,
        sensor_mm: float = 36.0,
        radius_mm: float = 380.0,
        elev_deg: tuple[float, float] = (30.0, 66.0),
        turns: float = 2.25,
        seed: int = 7,
        samples: int = 48,
        px_per_mm: float = 10.0,
        margin_mm: float = 12.0,
        fps: int = 30,
        board_scale: float = 1.0,
        engine: str = "BLENDER_EEVEE",
        use_charuco_module: bool = False,
        force: bool = True,
        quiet: bool = True,
    ) -> dict[str, Any]:
        """Render a synthetic scene with exact ground truth.

        Returns a dict with keys ``video``, ``frames_dir``, ``frames``,
        ``ground_truth`` (path), ``truth`` (the parsed ground-truth dict),
        ``true_dims_mm``, ``poses``, ``intrinsics`` and ``timing``.

        ``board_scale`` models the single largest real-world error source: a
        consumer printer that does not reproduce the target at exactly 100%.
        It multiplies the board's PHYSICAL size while the pipeline goes on
        believing the nominal ``square_mm``, so every recovered millimetre comes
        out scaled by ``1/board_scale``. The object itself is unaffected, so
        ``true_dims_mm`` stays exact and ``expected_reported_mm`` records what a
        correct pipeline should report given the mis-scaled board.
        Leave at 1.0 for a perfectly printed board.

        Set ``force=False`` to reuse an existing render in ``out_dir`` when its
        frame count and object dimensions already match -- useful in tests.
        """
        if board_scale <= 0:
            raise ValueError(f"board_scale must be positive, got {board_scale}")
        out_dir = Path(out_dir).resolve()
        frames_dir = out_dir / "frames"
        truth_path = out_dir / "ground_truth.json"
        video_path = out_dir / "scene.mp4"
        spec = board or BoardSpec()
        object_mm = tuple(float(v) for v in object_mm)  # type: ignore[assignment]

        if len(object_mm) != 3 or min(object_mm) <= 0:
            raise ValueError(f"object_mm must be three positive numbers, got {object_mm!r}")
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {n_frames}")

        if not force and truth_path.exists() and video_path.exists():
            try:
                cached = json.loads(truth_path.read_text())
                ok = (
                    cached.get("schema") == SCHEMA
                    and len(cached.get("poses", [])) == n_frames
                    and [round(v, 6) for v in cached["object"]["requested_mm"]]
                    == [round(v, 6) for v in object_mm]
                )
                if ok:
                    cached["reused"] = True
                    return _scene_return(cached, out_dir, frames_dir, video_path, truth_path)
            except Exception:  # noqa: BLE001 - a broken cache just means re-render
                pass

        for tool, path in (("blender", config.BLENDER_BIN), ("ffmpeg", config.FFMPEG_BIN),
                           ("ffprobe", config.FFPROBE_BIN)):
            if not Path(path).exists():
                raise FileNotFoundError(f"{tool} not found at {path}")

        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        tex_info = board_texture(
            spec,
            out_dir / "board_texture.png",
            px_per_mm=px_per_mm,
            margin_mm=margin_mm,
            use_charuco_module=use_charuco_module,
        )

        width, height = int(resolution[0]), int(resolution[1])
        if width % 2 or height % 2:
            raise ValueError("resolution must be even in both axes for yuv420p H.264")

        result_path = out_dir / "blender_result.json"
        scene_spec = {
            "out_dir": str(out_dir),
            "frames_dir": str(frames_dir),
            "result_json": str(result_path),
            "mesh_stl": str(out_dir / "object_truth.stl"),
            "board_png": tex_info["path"],
            "board": spec.to_json(),
            "board_w_mm": spec.width_mm,
            "board_h_mm": spec.height_mm,
            "margin_mm": margin_mm,
            "board_scale": float(board_scale),
            "object_mm": list(object_mm),
            "n_frames": int(n_frames),
            "resolution": [width, height],
            "lens_mm": float(lens_mm),
            "sensor_mm": float(sensor_mm),
            "radius_mm": float(radius_mm),
            "elev_deg": [float(elev_deg[0]), float(elev_deg[1])],
            "turns": float(turns),
            "seed": int(seed),
            "samples": int(samples),
            "engine": engine,
        }
        spec_path = out_dir / "scene_spec.json"
        spec_path.write_text(json.dumps(scene_spec, indent=2))

        t0 = time.time()
        _run(
            [
                str(config.BLENDER_BIN),
                "--background",
                "--factory-startup",
                "--python",
                str(Path(__file__).resolve()),
                "--",
                str(spec_path),
            ],
            "blender render",
            quiet=quiet,
        )
        blender_seconds = time.time() - t0

        if not result_path.exists():
            raise RuntimeError(
                f"blender exited 0 but wrote no result file at {result_path}; "
                f"the scene script did not reach the end"
            )
        result = json.loads(result_path.read_text())

        rendered = sorted(frames_dir.glob("frame_*.png"))
        if len(rendered) != n_frames:
            raise RuntimeError(
                f"expected {n_frames} rendered frames in {frames_dir}, found {len(rendered)}"
            )

        t0 = time.time()
        if video_path.exists():
            video_path.unlink()
        _run(
            [
                str(config.FFMPEG_BIN),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            "ffmpeg encode",
        )
        encode_seconds = time.time() - t0
        probe = _ffprobe(video_path)
        if probe["width"] != width or probe["height"] != height:
            raise RuntimeError(
                f"encoded video is {probe['width']}x{probe['height']}, expected {width}x{height}"
            )

        intr = _intrinsics_from_optics(lens_mm, sensor_mm, width, height)
        bl_intr = result["intrinsics"]
        for key in ("fx", "fy", "cx", "cy"):
            if abs(bl_intr[key] - getattr(intr, key)) > 1e-6:
                raise RuntimeError(
                    f"intrinsics disagreement on {key}: blender said {bl_intr[key]}, "
                    f"host computed {getattr(intr, key)}"
                )

        truth = {
            "schema": SCHEMA,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "caveat": CAVEAT,
            "frame_note": (
                "poses are BOARD frame per OpenCV 5 CharucoBoard: origin at the outer "
                "top-left board corner, +X right along the board image, +Y down it, "
                "+Z into the back of the board -- so camera centres have negative Z. "
                "center_world is the same camera in the physical Blender frame "
                "(board face up, +Z up), related by p_board = diag(1,-1,-1) @ p_world."
            ),
            "board": spec.to_json(),
            "board_scale": float(board_scale),
            "board_texture": tex_info,
            "object": result["object"],
            "camera": {
                **{
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "cx": intr.cx,
                    "cy": intr.cy,
                    "dist": list(intr.dist),
                    "width": intr.width,
                    "height": intr.height,
                },
                "lens_mm": lens_mm,
                "sensor_mm": sensor_mm,
                "model": "exact pinhole, zero distortion",
            },
            "frames": {
                "dir": str(frames_dir),
                "pattern": "frame_%04d.png",
                "count": len(rendered),
                "fps": fps,
            },
            "video": {"path": str(video_path), **probe},
            "path_params": {
                "radius_mm": radius_mm,
                "elev_deg": list(elev_deg),
                "turns": turns,
                "seed": seed,
            },
            "poses": result["poses"],
            "timing": {
                "blender_total_s": round(blender_seconds, 2),
                "render_s": round(result["timing"]["render_seconds"], 2),
                "per_frame_s": round(result["timing"]["per_frame_seconds"], 3),
                "encode_s": round(encode_seconds, 2),
            },
            "blender_version": result.get("blender_version"),
            "engine": engine,
        }
        truth_path.write_text(json.dumps(truth, indent=2))
        return _scene_return(truth, out_dir, frames_dir, video_path, truth_path)

    def _scene_return(
        truth: dict[str, Any],
        out_dir: Path,
        frames_dir: Path,
        video_path: Path,
        truth_path: Path,
    ) -> dict[str, Any]:
        return {
            "out_dir": str(out_dir),
            "video": str(video_path),
            "frames_dir": str(frames_dir),
            "frames": [str(p) for p in sorted(frames_dir.glob("frame_*.png"))],
            "ground_truth": str(truth_path),
            "board_texture": truth["board_texture"]["path"],
            "mesh_stl": truth["object"].get("mesh_stl"),
            "true_dims_mm": truth["object"]["true_dims_mm"],
            "intrinsics": truth["camera"],
            "poses": truth["poses"],
            "timing": truth["timing"],
            "truth": truth,
            "reused": truth.get("reused", False),
        }

    def load_truth(out_dir: str | os.PathLike) -> dict[str, Any]:
        """Read back a previously rendered scene's ground truth."""
        path = Path(out_dir) / "ground_truth.json"
        if not path.exists():
            raise FileNotFoundError(f"no ground truth at {path}; run make_scene first")
        return json.loads(path.read_text())


# ===========================================================================
# BLENDER SIDE -- runs under Blender's bundled Python (bpy + mathutils only)
# ===========================================================================
else:
    import bmesh  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore

    def _clear_scene() -> None:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def _new_material(name: str):
        mat = bpy.data.materials.new(name)
        try:
            mat.use_nodes = True
        except Exception:  # noqa: BLE001
            pass
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (600, 0)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (300, 0)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat, nt, bsdf

    def _set_input(node, name: str, value) -> None:
        if name in node.inputs:
            node.inputs[name].default_value = value

    def _board_material(png_path: str):
        mat, nt, bsdf = _new_material("BoardMat")
        img = bpy.data.images.load(png_path)
        img.colorspace_settings.name = "sRGB"
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Cubic"
        tex.extension = "EXTEND"
        tex.location = (0, 0)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        # Matte paper: high roughness, low specular, so markers never blow out.
        _set_input(bsdf, "Roughness", 0.88)
        _set_input(bsdf, "Metallic", 0.0)
        _set_input(bsdf, "Specular IOR Level", 0.28)
        return mat

    def _object_material(seed: int):
        """Richly textured matte material.

        Photogrammetry needs high-frequency, non-repeating surface detail. A
        smooth untextured primitive produces no feature matches and will not
        reconstruct at all, so this is not cosmetic.
        """
        mat, nt, bsdf = _new_material("ObjectMat")
        coord = nt.nodes.new("ShaderNodeTexCoord")
        coord.location = (-900, 0)

        # base colour: low-frequency mottling through a multi-stop ramp
        n1 = nt.nodes.new("ShaderNodeTexNoise")
        n1.location = (-650, 200)
        _set_input(n1, "Scale", 16.0 + (seed % 5))
        _set_input(n1, "Detail", 10.0)
        _set_input(n1, "Roughness", 0.55)
        nt.links.new(coord.outputs["Generated"], n1.inputs["Vector"])

        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.location = (-420, 200)
        el = ramp.color_ramp.elements
        el[0].position = 0.30
        el[0].color = (0.14, 0.10, 0.07, 1.0)
        el[1].position = 0.72
        el[1].color = (0.78, 0.66, 0.46, 1.0)
        m = ramp.color_ramp.elements.new(0.48)
        m.color = (0.46, 0.28, 0.16, 1.0)
        m2 = ramp.color_ramp.elements.new(0.60)
        m2.color = (0.30, 0.44, 0.34, 1.0)
        nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        # roughness break-up so highlights are not uniform
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.location = (-650, -120)
        _set_input(vor, "Scale", 26.0)
        nt.links.new(coord.outputs["Generated"], vor.inputs["Vector"])
        rramp = nt.nodes.new("ShaderNodeValToRGB")
        rramp.location = (-420, -120)
        rramp.color_ramp.elements[0].position = 0.0
        rramp.color_ramp.elements[0].color = (0.42, 0.42, 0.42, 1.0)
        rramp.color_ramp.elements[1].position = 1.0
        rramp.color_ramp.elements[1].color = (0.86, 0.86, 0.86, 1.0)
        nt.links.new(vor.outputs["Distance"], rramp.inputs["Fac"])
        nt.links.new(rramp.outputs["Color"], bsdf.inputs["Roughness"])

        # fine bump: the shading cue that actually drives dense matching
        n2 = nt.nodes.new("ShaderNodeTexNoise")
        n2.location = (-650, -400)
        _set_input(n2, "Scale", 55.0)
        _set_input(n2, "Detail", 8.0)
        nt.links.new(coord.outputs["Generated"], n2.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.location = (-200, -400)
        _set_input(bump, "Strength", 0.45)
        _set_input(bump, "Distance", 0.4)
        nt.links.new(n2.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        _set_input(bsdf, "Metallic", 0.0)
        _set_input(bsdf, "Specular IOR Level", 0.32)
        return mat

    def _table_material():
        mat, nt, bsdf = _new_material("TableMat")
        coord = nt.nodes.new("ShaderNodeTexCoord")
        coord.location = (-600, 0)
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.location = (-380, 0)
        _set_input(n, "Scale", 0.012)
        _set_input(n, "Detail", 6.0)
        nt.links.new(coord.outputs["Object"], n.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.location = (-160, 0)
        ramp.color_ramp.elements[0].color = (0.18, 0.15, 0.13, 1.0)
        ramp.color_ramp.elements[1].color = (0.34, 0.29, 0.24, 1.0)
        nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        _set_input(bsdf, "Roughness", 0.75)
        _set_input(bsdf, "Specular IOR Level", 0.2)
        return mat

    def _make_board_plane(spec: dict, margin_mm: float, png_path: str, board_scale: float = 1.0):
        """Textured plane whose printed face lies exactly on world z=0.

        World is the physical frame: board face up (+Z), board point (bx, by, 0)
        sits at world (bx, -by, 0). Blender's UV origin is bottom-left, so a
        straight u=x / v=y mapping puts image row 0 (by=0) at world y=0 and the
        last image row (by=board_h) at world y=-board_h, which is exactly the
        un-mirrored layout the detector needs.
        """
        # board_scale != 1 makes the PHYSICAL board a different size from the
        # nominal square_mm the detector assumes -- i.e. a printer that did not
        # print at 100%. Scaling about the board origin keeps (0,0,0) fixed.
        k = float(board_scale)
        w = float(spec["board_w_mm"]) * k
        h = float(spec["board_h_mm"]) * k
        m = float(margin_mm) * k
        x0, x1 = -m, w + m
        y0, y1 = -(h + m), m

        verts = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        me = bpy.data.meshes.new("BoardMesh")
        me.from_pydata(verts, [], [(0, 1, 2, 3)])
        me.update()
        uv = me.uv_layers.new(name="UVMap")
        for loop in me.loops:
            uv.data[loop.index].uv = uvs[loop.vertex_index]
        ob = bpy.data.objects.new("Board", me)
        bpy.context.collection.objects.link(ob)
        ob.data.materials.append(_board_material(png_path))
        return ob

    def _make_table(spec: dict, board_scale: float = 1.0):
        w = float(spec["board_w_mm"]) * float(board_scale)
        h = float(spec["board_h_mm"]) * float(board_scale)
        cx, cy = w / 2.0, -h / 2.0
        s = 900.0
        verts = [
            (cx - s, cy - s, -0.6),
            (cx + s, cy - s, -0.6),
            (cx + s, cy + s, -0.6),
            (cx - s, cy + s, -0.6),
        ]
        me = bpy.data.meshes.new("TableMesh")
        me.from_pydata(verts, [], [(0, 1, 2, 3)])
        me.update()
        ob = bpy.data.objects.new("Table", me)
        bpy.context.collection.objects.link(ob)
        ob.data.materials.append(_table_material())
        return ob

    def _activate(ob) -> None:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob

    def _world_bbox(ob) -> tuple[Vector, Vector]:
        mw = ob.matrix_world
        pts = [mw @ v.co for v in ob.data.vertices]
        if not pts:
            raise RuntimeError("object has no vertices")
        mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        return mn, mx

    def _make_object(dims, center_xy, seed: int):
        """Asymmetric, bumpy, well-textured solid rescaled to an EXACT bbox.

        The shape is built by whatever modifier stack looks interesting; the
        exactness comes from measuring the resulting bounding box and scaling it
        onto the requested millimetres afterwards. That way the ground truth is
        exact regardless of how the geometry was produced.
        """
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
        body = bpy.context.object
        body.name = "TargetObject"

        # corner slice -- kills the symmetry so pose is unambiguous
        bpy.ops.mesh.primitive_cube_add(size=1.5, location=(1.05, 1.05, 0.95))
        cutter_a = bpy.context.object
        cutter_a.name = "CutterA"
        cutter_a.rotation_euler = (0.62, 0.41, 0.33)

        # groove along X on one face -- a crisp, measurable feature
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, -1.02, 0.15))
        cutter_b = bpy.context.object
        cutter_b.name = "CutterB"
        cutter_b.scale = (3.2, 0.30, 0.30)
        cutter_b.rotation_euler = (0.0, 0.10, 0.0)

        for cut in (cutter_a, cutter_b):
            mod = body.modifiers.new(f"bool_{cut.name}", "BOOLEAN")
            mod.object = cut
            mod.operation = "DIFFERENCE"
            try:
                mod.solver = "EXACT"
            except Exception:  # noqa: BLE001
                pass
            cut.hide_render = True

        bev = body.modifiers.new("bevel", "BEVEL")
        bev.width = 0.07
        bev.segments = 3
        bev.limit_method = "ANGLE"
        bev.angle_limit = math.radians(28.0)

        # SIMPLE (not Catmull-Clark) subdivision: this exists purely to give the
        # displace modifier enough vertices to work with. Smoothing subdivision
        # would round the machined character of the block away entirely.
        sub = body.modifiers.new("subsurf", "SUBSURF")
        sub.subdivision_type = "SIMPLE"
        sub.levels = 4
        sub.render_levels = 4

        tex = bpy.data.textures.new("ObjDisp", type="CLOUDS")
        tex.noise_scale = 0.34
        tex.noise_depth = 4
        disp = body.modifiers.new("displace", "DISPLACE")
        disp.texture = tex
        disp.strength = 0.032
        disp.mid_level = 0.5

        tw = body.modifiers.new("twist", "SIMPLE_DEFORM")
        tw.deform_method = "TWIST"
        tw.deform_axis = "Z"
        tw.angle = math.radians(18.0)

        tp = body.modifiers.new("taper", "SIMPLE_DEFORM")
        tp.deform_method = "TAPER"
        tp.deform_axis = "Z"
        tp.factor = 0.20

        _activate(body)
        bpy.ops.object.convert(target="MESH")
        for cut in (cutter_a, cutter_b):
            bpy.data.objects.remove(cut, do_unlink=True)

        _activate(body)
        mn, mx = _world_bbox(body)
        size = mx - mn
        if min(size) <= 0:
            raise RuntimeError(f"degenerate object bbox {tuple(size)}")
        body.scale = (dims[0] / size.x, dims[1] / size.y, dims[2] / size.z)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        mn, mx = _world_bbox(body)
        body.location = body.location + Vector(
            (
                center_xy[0] - (mn.x + mx.x) / 2.0,
                center_xy[1] - (mn.y + mx.y) / 2.0,
                -mn.z,  # sit exactly on the board plane
            )
        )
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        mn, mx = _world_bbox(body)
        final = (mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)
        err = max(abs(final[i] - dims[i]) for i in range(3))
        if err > 1e-4:
            raise RuntimeError(
                f"failed to force exact object dimensions: wanted {tuple(dims)}, "
                f"got {final} (max error {err:.6f} mm)"
            )
        if abs(mn.z) > 1e-4:
            raise RuntimeError(f"object does not sit on the board plane: min z = {mn.z}")

        # Applying the boolean modifiers merges the cutters' (empty) material
        # slots into this mesh, leaving slot 0 unassigned with every polygon
        # pointing at it. Wipe the slots and force every face onto our material,
        # or the object renders as untextured default grey and will not
        # reconstruct at all.
        body.data.materials.clear()
        body.data.materials.append(_object_material(seed))
        for poly in body.data.polygons:
            poly.material_index = 0
        bm = bmesh.new()
        bm.from_mesh(body.data)
        volume = abs(bm.calc_volume(signed=True))
        n_tris = sum(max(0, len(f.verts) - 2) for f in bm.faces)
        bm.free()
        return body, final, mn, mx, volume, n_tris

    def _setup_world_and_lights(seed: int, light_scale: float = 1.0) -> None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        try:
            world.use_nodes = True
        except Exception:  # noqa: BLE001
            pass
        bg = world.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs["Color"].default_value = (0.52, 0.55, 0.60, 1.0)
            bg.inputs["Strength"].default_value = 0.30 * light_scale

        # SUN lamps only: their irradiance is distance-independent, so the
        # millimetre-scale world (1 BU = 1 mm) needs no wattage gymnastics.
        #
        # Levels are tuned so the board's white paper lands near 0.85 rather
        # than clipping at 1.0. Clipped whites destroy the local contrast the
        # corner refiner needs, and clipped object highlights erase exactly the
        # surface texture photogrammetry depends on.
        for name, rot, energy, colour, softness in (
            ("KeySun", (math.radians(38), 0.0, math.radians(35)),
             1.55 * light_scale, (1.0, 0.97, 0.92), 0.10),
            ("FillSun", (math.radians(58), 0.0, math.radians(215)),
             0.55 * light_scale, (0.88, 0.93, 1.0), 0.25),
            ("RimSun", (math.radians(72), 0.0, math.radians(120)),
             0.30 * light_scale, (1.0, 1.0, 1.0), 0.30),
        ):
            data = bpy.data.lights.new(name, type="SUN")
            data.energy = energy
            data.color = colour
            data.angle = softness
            ob = bpy.data.objects.new(name, data)
            ob.rotation_euler = rot
            bpy.context.collection.objects.link(ob)

    def _setup_camera(spec: dict):
        data = bpy.data.cameras.new("Cam")
        data.lens = float(spec["lens_mm"])
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = float(spec["sensor_mm"])
        data.shift_x = 0.0
        data.shift_y = 0.0
        data.clip_start = 1.0
        data.clip_end = 100000.0
        ob = bpy.data.objects.new("Cam", data)
        bpy.context.collection.objects.link(ob)
        bpy.context.scene.camera = ob
        return ob

    def _setup_render(spec: dict) -> None:
        sc = bpy.context.scene
        sc.render.engine = spec["engine"]
        sc.render.resolution_x = int(spec["resolution"][0])
        sc.render.resolution_y = int(spec["resolution"][1])
        sc.render.resolution_percentage = 100
        sc.render.pixel_aspect_x = 1.0
        sc.render.pixel_aspect_y = 1.0
        sc.render.film_transparent = False
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = "8"
        sc.render.image_settings.compression = 15
        # 'Standard' keeps the board's black/white contrast intact. AgX (the
        # Blender default) tone-maps highlights and shadows towards mid-grey,
        # which measurably hurts marker detection at grazing angles.
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.look = "None"
        sc.view_settings.exposure = 0.0
        sc.view_settings.gamma = 1.0
        try:
            sc.eevee.taa_render_samples = int(spec["samples"])
        except Exception:  # noqa: BLE001
            pass
        try:
            sc.eevee.use_shadows = True
        except Exception:  # noqa: BLE001
            pass

    def _camera_path(spec: dict, target: Vector):
        """Deterministic hemispherical spiral, jittered to look hand-held."""
        n = int(spec["n_frames"])
        rnd = random.Random(int(spec["seed"]))
        r0 = float(spec["radius_mm"])
        e0, e1 = float(spec["elev_deg"][0]), float(spec["elev_deg"][1])
        turns = float(spec["turns"])
        out = []
        for i in range(n):
            f = i / n
            azim = 2.0 * math.pi * turns * f + math.radians(23.0)
            elev = math.radians(e0 + (e1 - e0) * (0.5 - 0.5 * math.cos(2.0 * math.pi * f)))
            elev += math.radians(rnd.uniform(-1.6, 1.6))
            azim += math.radians(rnd.uniform(-1.6, 1.6))
            r = r0 * (1.0 + 0.055 * math.sin(4.0 * math.pi * f) + rnd.uniform(-0.02, 0.02))
            pos = Vector(
                (
                    target.x + r * math.cos(elev) * math.cos(azim),
                    target.y + r * math.cos(elev) * math.sin(azim),
                    target.z + r * math.sin(elev),
                )
            )
            aim = target + Vector(
                (rnd.uniform(-9, 9), rnd.uniform(-9, 9), rnd.uniform(-5, 5))
            )
            out.append((pos, aim, math.degrees(azim), math.degrees(elev), r))
        return out

    # world (physical, board face up) <-> OpenCV ChArUco board frame
    _M = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))

    def _pose_board_frame(cam, board_scale: float = 1.0):
        """Return (R, t, center_nominal, center_physical, p_world), OpenCV convention.

        With a mis-scaled board the physical scene (board scaled by k, camera at
        p) projects identically to (nominal board, camera at p/k) -- uniformly
        scaling the whole scene about the board origin does not change the
        image. So solvePnP, which assumes the nominal square_mm, recovers the
        camera at c/k. That is what gets reported as ground truth, because it is
        what the pipeline will actually produce.
        """
        mw = cam.matrix_world
        r_bw = mw.to_quaternion().to_matrix()  # guaranteed orthonormal
        p_w = mw.to_translation()
        # x_cvcam = M @ R_bw^T @ (M @ x_board - p_w)
        R = _M @ r_bw.transposed() @ _M
        c_phys = _M @ p_w
        c_nom = c_phys / float(board_scale)
        t = -(R @ c_nom)
        return R, t, c_nom, c_phys, p_w

    def _export_mesh(ob, path: str) -> str | None:
        _activate(ob)
        try:
            bpy.ops.wm.stl_export(
                filepath=path,
                export_selected_objects=True,
                global_scale=1.0,
                use_scene_unit=False,
                apply_modifiers=True,
                ascii_format=False,
            )
            return path
        except Exception as exc:  # noqa: BLE001
            print(f"S2P_WARN stl export failed: {type(exc).__name__}: {exc}")
            return None

    def _blender_main() -> None:
        argv = sys.argv
        if "--" not in argv:
            raise SystemExit("synth.py (blender mode) requires: -- <scene_spec.json>")
        spec_path = argv[argv.index("--") + 1]
        with open(spec_path) as fh:
            spec = json.load(fh)

        _clear_scene()
        _setup_render(spec)
        _setup_world_and_lights(int(spec["seed"]), float(spec.get("light_scale", 1.0)))

        k = float(spec.get("board_scale", 1.0))
        _make_table(spec, k)
        _make_board_plane(spec, float(spec["margin_mm"]), spec["board_png"], k)

        dims = [float(v) for v in spec["object_mm"]]
        center_xy = (
            float(spec["board_w_mm"]) * k / 2.0,
            -float(spec["board_h_mm"]) * k / 2.0,
        )
        obj, true_dims, mn, mx, volume, n_tris = _make_object(dims, center_xy, int(spec["seed"]))
        mesh_stl = _export_mesh(obj, spec["mesh_stl"])

        cam = _setup_camera(spec)
        target = Vector((center_xy[0], center_xy[1], true_dims[2] * 0.45))
        path = _camera_path(spec, target)

        width, height = int(spec["resolution"][0]), int(spec["resolution"][1])
        f_px = float(spec["lens_mm"]) / float(spec["sensor_mm"]) * width

        frames_dir = spec["frames_dir"]
        os.makedirs(frames_dir, exist_ok=True)

        poses = []
        t_render0 = time.time()
        for i, (pos, aim, azim, elev, radius) in enumerate(path):
            cam.location = pos
            cam.rotation_euler = (aim - pos).to_track_quat("-Z", "Y").to_euler()
            bpy.context.view_layer.update()

            R, t, c_nom, c_phys, p_w = _pose_board_frame(cam, k)
            name = f"frame_{i:04d}.png"
            out_png = os.path.join(frames_dir, name)
            bpy.context.scene.render.filepath = out_png
            bpy.ops.render.render(write_still=True)

            poses.append(
                {
                    "name": name,
                    "index": i,
                    "file": out_png,
                    "R": [list(R[r]) for r in range(3)],
                    "t": list(t),
                    "center": list(c_nom),
                    "center_physical": list(c_phys),
                    "center_world": list(p_w),
                    "azim_deg": azim,
                    "elev_deg": elev,
                    "radius_mm": radius,
                }
            )
            if (i + 1) % 10 == 0 or i == 0:
                print(f"S2P_PROGRESS rendered {i + 1}/{len(path)}")
        render_seconds = time.time() - t_render0

        # bbox in the OpenCV board frame (y and z flip, so min/max swap), then
        # into the NOMINAL frame the pipeline recovers by dividing out k
        b_lo = [min(mn.x, mx.x) / k, min(-mn.y, -mx.y) / k, min(-mn.z, -mx.z) / k]
        b_hi = [max(mn.x, mx.x) / k, max(-mn.y, -mx.y) / k, max(-mn.z, -mx.z) / k]

        result = {
            "blender_version": bpy.app.version_string,
            "object": {
                "requested_mm": dims,
                "true_dims_mm": [float(v) for v in true_dims],
                "expected_reported_mm": [float(v) / k for v in true_dims],
                "board_scale": k,
                "bbox_min_world_mm": list(mn),
                "bbox_max_world_mm": list(mx),
                "bbox_min_board_mm": b_lo,
                "bbox_max_board_mm": b_hi,
                "center_world_xy_mm": list(center_xy),
                "volume_mm3": float(volume),
                "triangles": int(n_tris),
                "mesh_stl": mesh_stl,
                "note": (
                    "true_dims_mm is the axis-aligned bounding box of the actual "
                    "rendered surface, forced to the requested millimetres by "
                    "rescaling after all modifiers were applied. The object rests "
                    "with its lowest point exactly on the board plane (z=0). "
                    "expected_reported_mm is what a CORRECT pipeline should "
                    "report: it differs from true_dims_mm only when board_scale "
                    "!= 1, i.e. when the printed board is not at 100% scale."
                ),
            },
            "intrinsics": {
                "fx": f_px,
                "fy": f_px,
                "cx": width / 2.0 - 0.5,
                "cy": height / 2.0 - 0.5,
                "width": width,
                "height": height,
                "dist": [0.0] * 5,
            },
            "poses": poses,
            "timing": {
                "render_seconds": render_seconds,
                "per_frame_seconds": render_seconds / max(1, len(path)),
            },
        }
        with open(spec["result_json"], "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"S2P_RESULT_JSON {spec['result_json']}")
        print(
            f"S2P_DONE frames={len(poses)} render_s={render_seconds:.1f} "
            f"per_frame_s={render_seconds / max(1, len(path)):.2f} "
            f"dims={true_dims}"
        )

    if __name__ == "__main__":
        _blender_main()
