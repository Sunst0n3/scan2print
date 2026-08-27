"""Mesh conditioning -- raw Object Capture output to a printable metric solid.

The chain is:

    load()             any reconstruction format  ->  trimesh.Trimesh in RECON units
    apply_scale()      RECON  ->  BOARD millimetres (this is where the metrology lands)
    remove_ground()    drop the board/table plane, keep the largest real component
    make_watertight()  degenerate/duplicate cleanup, winding, hole fill
    decimate()         cap the face count for the slicer
    orient_for_print() largest stable face on z=0, centred on the bed origin
    stats()            measured bbox / volume / bed fit
    condition()        all of the above, ending in an STL

Self-test:  python -m scan2print.mesh

WHY THE LOADER IS FUSSY ABOUT UNITS AND AXES
Object Capture writes USDZ, and neither trimesh 5.0 nor any pure-Python reader in
this venv can open USD, so the file has to go through Blender. Blender's USD
importer does two things that would each silently destroy the metric chain:

  1. `apply_unit_conversion_scale` defaults to True, which multiplies all
     geometry by the stage's `metersPerUnit`. A stage authored at
     metersPerUnit=0.01 comes back 100x too small. Measured on this machine.
  2. The importer *always* rotates a `upAxis = "Y"` stage into Blender's Z-up
     convention, mapping usd (x,y,z) -> blender (x,-z,y). There is no flag to
     turn this off.

The ScaleResult from scale.py is fitted against camera centres read straight out
of the reconstruction in its own raw coordinates, so the mesh must be returned in
those same raw coordinates or the similarity transform is applied to a rotated,
rescaled copy and every dimension comes out wrong. load() therefore forces the
unit conversion off and undoes the up-axis rotation itself.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

try:  # normal package import
    from scan2print import config
    from scan2print.types import BoardSpec, ScaleResult
except ImportError:  # importable when the repo root is not already on sys.path
    # NOTE: `python scan2print/mesh.py` does NOT work, and cannot be made to.
    # That form puts scan2print/ at sys.path[0], where types.py shadows the
    # stdlib `types` module and CPython fails during its own startup, long
    # before this file runs. Use `python -m scan2print.mesh`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scan2print import config
    from scan2print.types import BoardSpec, ScaleResult


# --- module-local tuning ---------------------------------------------------

USD_EXTS = {".usdz", ".usdc", ".usda", ".usd"}
USDCAT_BIN = Path("/usr/bin/usdcat")          # ships with macOS
BLENDER_TIMEOUT_S = 900
USDCAT_TIMEOUT_S = 300

# Components whose centroid lands further than this outside the printed board
# footprint are reconstruction debris (table, hands, wall), not the subject.
GROUND_LATERAL_MARGIN_MM = 60.0

# A decimation that moves any bounding-box extent by more than this is not a
# simplification any more, it is a shape change, and a metrology tool must say so.
DECIMATE_BBOX_TOL_MM = 0.25

# Default height of the ground cut above the board plane. 1.0 mm was the
# original value and is too close to the plane: a measured z_tol sweep on a
# real Object Capture model had the board leaking back into the kept component
# as soon as the cut dropped below it (footprint 47.2 mm at 1.0, 53.8 at 0.5,
# 79.2 at 0.25, 202.7 at 0.1). The cut now costs nothing in height *or*
# footprint -- the ring is extruded back down rather than collapsed -- so the
# only remaining pressure is to stay clear of the plane, and 1.5 mm buys 50%
# more margin than the value that was already marginal. Anything much larger
# starts extrapolating the wall over a distance where a curved profile is no
# longer locally linear.
DEFAULT_Z_TOL_MM = 1.5

# Base reconstruction. The ground cut must sit above z = 0 or it eats the board
# plane, so the lowest ring of the kept solid is a cross-section taken z_tol mm
# up. On a tapered wall that cross-section is NOT the footprint: dropping it
# straight down under-reports the base by z_tol*tan(taper) per side (measured on
# truncated cones of true base diameter 40.0 mm at z_tol=1.0: 0 deg -> 0.000 mm,
# 15 deg -> -0.536 mm, 30 deg -> -1.155 mm, 45 deg -> -2.000 mm, with the height
# exactly right in every case so nothing in the report flagged it). The ring is
# therefore extruded down along the local wall direction instead.
#
# The extrapolation is capped: past this slope the wall is nearly horizontal at
# the cut (a flange, a fillet, or board-plane crud fused to the object) and
# extrapolating it invents a footprint rather than recovering one. 1.5 was
# chosen by measurement, not taste: on the real Object Capture model in
# out/run_demo (true 50 x 30 x 20 mm) the recovered oriented box saturates there
# -- caps of 1.5 and 3.0 both give 49.06 x 29.46 x 19.96, while a cap of 3.0 at
# a cut low enough to still catch board crud runs away to 54.36 mm. It also
# bounds the worst case: no point of the base can move more than 1.5*z_tol mm.
# Every truncated-cone taper up to 45 deg is recovered exactly well inside it.
BASE_MAX_TAPER_SLOPE = 1.5            # ~56.3 deg from vertical

# Second, independent cap on the same extrapolation: no ring vertex may travel
# more than this fraction of the ring's own smaller extent. Without it a slope
# pointing inwards on a small feature can push the ring through itself.
BASE_MAX_MOVE_FRAC = 0.25

# How far the footprint is allowed to move per millimetre of z_tol before the
# base is declared untrustworthy. Measured by re-cutting the mesh at several
# heights and re-extruding: on a base that is genuinely resolvable the corrected
# footprint barely depends on where the cut was taken.
#
# The verdict uses only the probes AT OR ABOVE the chosen cut. Probes below it
# are reported too, but they are not evidence against the current answer -- the
# board plane always leaks in as the cut approaches z=0, which is the entire
# reason the cut exists. What they do measure is how much margin is left, so
# they are surfaced as `leak_below_cut_pct`.
FOOTPRINT_SENSITIVITY_WARN_PCT = 2.0   # % of footprint per mm of z_tol
FOOTPRINT_LEAK_NOTE_PCT = 5.0          # % swing below the cut worth mentioning
FOOTPRINT_PROBE_FACTORS = (0.5, 1.0, 1.5, 2.0)

# Blender maps a Y-up USD stage into its own Z-up world as
# usd (x, y, z) -> blender (x, -z, y). This is the inverse, applied to Blender
# output to recover raw USD coordinates.
_BLENDER_ZUP_TO_USD_YUP = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


class MeshError(RuntimeError):
    """Raised when conditioning cannot produce a trustworthy result."""


# --- format handling -------------------------------------------------------

def _as_trimesh(obj: Any, source: str) -> trimesh.Trimesh:
    """Collapse whatever trimesh.load returned into a single Trimesh."""
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise MeshError(f"{source}: scene contains no triangle meshes")
        # concatenate() bakes each geometry's scene transform in, so multi-prim
        # reconstructions keep their relative placement.
        return trimesh.util.concatenate(obj.dump())
    raise MeshError(f"{source}: loaded object is a {type(obj).__name__}, not a mesh")


def _run(cmd: list[str], timeout: int, what: str) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise MeshError(f"{what}: executable not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MeshError(f"{what}: timed out after {timeout}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
        raise MeshError(f"{what}: exit {proc.returncode}\n" + "\n".join(tail))
    return proc


def usd_layer_metadata(path: Path) -> dict[str, Any]:
    """Read `metersPerUnit` and `upAxis` off a USD stage without loading geometry.

    Returns {} if usdcat is unavailable or the read fails; callers must cope,
    because a missing upAxis only means we cannot *confirm* the convention.
    """
    if not USDCAT_BIN.exists():
        return {}
    try:
        proc = _run([str(USDCAT_BIN), "--layerMetadata", str(path)],
                    USDCAT_TIMEOUT_S, "usdcat --layerMetadata")
    except MeshError:
        return {}
    text = proc.stdout
    out: dict[str, Any] = {}
    m = re.search(r"metersPerUnit\s*=\s*([0-9eE.+-]+)", text)
    if m:
        out["metersPerUnit"] = float(m.group(1))
    m = re.search(r'upAxis\s*=\s*"([XYZ])"', text)
    if m:
        out["upAxis"] = m.group(1)
    return out


_BLENDER_USD_TO_OBJ = r'''
import bpy, sys
argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(
    filepath=src,
    scale=1.0,
    apply_unit_conversion_scale=False,   # keep raw stage units; see mesh.py docstring
    import_materials=False,
    import_cameras=False,
    import_lights=False,
    import_curves=False,
    import_volumes=False,
    import_shapes=False,
    import_skeletons=False,
    import_blendshapes=False,
    import_points=False,
    merge_parent_xform=True,
)
n = sum(1 for o in bpy.data.objects if o.type == "MESH")
if n == 0:
    raise SystemExit("scan2print: no mesh prims found in " + src)
bpy.ops.wm.obj_export(
    filepath=dst,
    forward_axis="Y", up_axis="Z",       # Blender-native: no axis conversion
    global_scale=1.0,
    apply_modifiers=True,
    export_materials=False,
    export_uv=False,
    export_normals=False,
    export_triangulated_mesh=True,
)
print("scan2print: converted %d mesh object(s)" % n)
'''


def _usd_to_obj(src: Path, dst: Path) -> None:
    """Convert a USD/USDZ file to OBJ via headless Blender, preserving raw coords."""
    if not config.BLENDER_BIN.exists():
        raise MeshError(
            f"cannot read {src.name}: trimesh 5.0 has no USD loader and Blender is "
            f"missing at {config.BLENDER_BIN}"
        )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_BLENDER_USD_TO_OBJ)
        script = fh.name
    try:
        _run([str(config.BLENDER_BIN), "--background", "--factory-startup",
              "--python", script, "--", str(src), str(dst)],
             BLENDER_TIMEOUT_S, f"blender USD->OBJ ({src.name})")
    finally:
        Path(script).unlink(missing_ok=True)
    if not dst.exists() or dst.stat().st_size == 0:
        raise MeshError(f"blender produced no OBJ for {src}")


_USDA_ARRAY = r"\[(.*?)\]"


def _usda_fallback_mesh(src: Path) -> trimesh.Trimesh:
    """Parse meshes straight out of a flattened USDA. Fallback when Blender fails.

    Deliberately narrow: it understands points / faceVertexCounts /
    faceVertexIndices plus `matrix4d xformOp:transform`, `xformOp:translate` and
    `xformOp:scale`. Anything else (rotate, orient, inherited xforms) raises,
    because quietly ignoring a transform would move the geometry relative to the
    camera track the scale was fitted from.
    """
    if not USDCAT_BIN.exists():
        raise MeshError("usdcat not available for the USDA fallback reader")
    with tempfile.TemporaryDirectory() as td:
        flat = Path(td) / "flat.usda"
        _run([str(USDCAT_BIN), "-f", "-o", str(flat), str(src)],
             USDCAT_TIMEOUT_S, f"usdcat --flatten ({src.name})")
        text = flat.read_text(errors="replace")

    for bad in ("xformOp:rotate", "xformOp:orient"):
        if bad in text:
            raise MeshError(
                f"USDA fallback reader cannot handle {bad}; install/repair Blender "
                f"at {config.BLENDER_BIN} to read {src.name}"
            )

    blocks = re.split(r"\bdef\s+Mesh\s+", text)[1:]
    if not blocks:
        raise MeshError(f"no `def Mesh` prims found in flattened {src.name}")

    parts: list[trimesh.Trimesh] = []
    for blk in blocks:
        pm = re.search(r"point3f\[\]\s+points\s*=\s*" + _USDA_ARRAY, blk, re.S)
        cm = re.search(r"int\[\]\s+faceVertexCounts\s*=\s*" + _USDA_ARRAY, blk, re.S)
        im = re.search(r"int\[\]\s+faceVertexIndices\s*=\s*" + _USDA_ARRAY, blk, re.S)
        if not (pm and cm and im):
            continue
        pts = np.array([float(x) for x in re.findall(r"-?[\d.eE+-]+", pm.group(1))],
                       dtype=np.float64).reshape(-1, 3)
        counts = np.array([int(x) for x in re.findall(r"-?\d+", cm.group(1))], dtype=np.int64)
        idx = np.array([int(x) for x in re.findall(r"-?\d+", im.group(1))], dtype=np.int64)

        faces: list[list[int]] = []
        o = 0
        for c in counts:
            poly = idx[o:o + c]
            o += int(c)
            for k in range(1, len(poly) - 1):          # fan triangulation
                faces.append([int(poly[0]), int(poly[k]), int(poly[k + 1])])
        if not faces:
            continue

        mm = re.search(r"matrix4d\s+xformOp:transform\s*=\s*\((.*?)\)\s*\n", blk, re.S)
        if mm:
            vals = [float(x) for x in re.findall(r"-?[\d.eE+-]+", mm.group(1))]
            if len(vals) != 16:
                raise MeshError(f"malformed xformOp:transform in {src.name}")
            # USD matrices are row-vector convention: p_world = p_local * M
            pts = np.hstack([pts, np.ones((len(pts), 1))]) @ np.array(vals).reshape(4, 4)
            pts = pts[:, :3]
        tm = re.search(r"xformOp:translate\s*=\s*\(([^)]*)\)", blk)
        sm = re.search(r"xformOp:scale\s*=\s*\(([^)]*)\)", blk)
        if sm:
            pts = pts * np.array([float(x) for x in sm.group(1).split(",")])
        if tm:
            pts = pts + np.array([float(x) for x in tm.group(1).split(",")])

        parts.append(trimesh.Trimesh(vertices=pts, faces=np.array(faces, dtype=np.int64),
                                     process=False))

    if not parts:
        raise MeshError(f"USDA fallback reader extracted no faces from {src.name}")
    return parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)


def load(path: str | Path) -> trimesh.Trimesh:
    """Load a reconstruction from disk in its own raw (RECON) coordinates.

    Handles .usdz/.usdc/.usda/.usd via Blender (with a usdcat-based fallback) and
    everything trimesh can read natively (.obj/.ply/.stl/.glb/.off/.dae/...).
    No unit conversion and no axis conversion is applied: the returned mesh is in
    exactly the coordinates the reconstruction was authored in, which is what
    ScaleResult was fitted against.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"mesh.load: no such file: {p}")
    p = p.resolve()
    ext = p.suffix.lower()

    if ext not in USD_EXTS:
        try:
            obj = trimesh.load(p, force="mesh", process=False)
        except Exception as exc:
            raise MeshError(f"mesh.load: trimesh could not read {p.name}: {exc}") from exc
        mesh = _as_trimesh(obj, p.name)
        if len(mesh.faces) == 0:
            raise MeshError(f"mesh.load: {p.name} contains no faces")
        return mesh

    if ext == ".usdz":
        # USDZ is an uncompressed zip; a corrupt one is worth catching here
        # rather than inside Blender's log noise.
        if not zipfile.is_zipfile(p):
            raise MeshError(f"mesh.load: {p.name} has a .usdz extension but is not a zip")

    meta = usd_layer_metadata(p)
    up = meta.get("upAxis", "Y")   # USD's own default is Y when unspecified

    with tempfile.TemporaryDirectory() as td:
        obj_path = Path(td) / "recon.obj"
        try:
            _usd_to_obj(p, obj_path)
            loaded = trimesh.load(obj_path, force="mesh", process=False)
            mesh = _as_trimesh(loaded, p.name)
            if up == "Y":
                # Undo the up-axis rotation Blender's importer forces on us.
                mesh.apply_transform(_BLENDER_ZUP_TO_USD_YUP)
        except MeshError as blender_exc:
            try:
                mesh = _usda_fallback_mesh(p)   # already in raw USD coordinates
            except MeshError as fallback_exc:
                raise MeshError(
                    f"mesh.load: could not read {p.name}.\n"
                    f"  blender route: {blender_exc}\n"
                    f"  usdcat route:  {fallback_exc}"
                ) from blender_exc

    if len(mesh.faces) == 0:
        raise MeshError(f"mesh.load: {p.name} contains no faces")
    mesh.metadata["usd_metersPerUnit"] = meta.get("metersPerUnit")
    mesh.metadata["usd_upAxis"] = meta.get("upAxis")
    return mesh


# --- metric transform ------------------------------------------------------

def scale_matrix(res: ScaleResult, allow_reject: bool = False) -> np.ndarray:
    """Build the 4x4 RECON -> BOARD-mm similarity: x_board = scale * R @ x_recon + t.

    Validates R is a proper rotation. A reflection here would produce a mirrored
    part that measures correctly and is the wrong handedness -- the exact failure
    scale.umeyama() goes out of its way to prevent -- so it is refused.
    """
    if res.quality == "reject" and not allow_reject:
        raise MeshError(
            "mesh: refusing a ScaleResult with quality='reject'. "
            f"rms={res.rms_mm:.2f} mm, notes: {'; '.join(res.notes) or 'none'}. "
            "Pass allow_reject=True to override."
        )
    R = np.asarray(res.R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(res.t, dtype=np.float64).reshape(3)
    s = float(res.scale)
    if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite(s):
        raise MeshError("mesh: ScaleResult contains NaN/Inf")
    if s <= 0.0:
        raise MeshError(f"mesh: ScaleResult.scale must be positive, got {s}")
    orth = float(np.abs(R @ R.T - np.eye(3)).max())
    if orth > 1e-6:
        raise MeshError(f"mesh: ScaleResult.R is not orthonormal (max |RR^T-I| = {orth:.2e})")
    det = float(np.linalg.det(R))
    if det < 0.0:
        raise MeshError(
            f"mesh: ScaleResult.R is a reflection (det = {det:+.6f}). Applying it "
            "would mirror the part. Refusing."
        )

    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def apply_scale(mesh: trimesh.Trimesh, res: ScaleResult,
                allow_reject: bool = False) -> trimesh.Trimesh:
    """Transform a RECON mesh into BOARD millimetres. Returns a new mesh."""
    out = mesh.copy()
    out.apply_transform(scale_matrix(res, allow_reject=allow_reject))
    return out


# --- ground removal --------------------------------------------------------

def _component_size(part: trimesh.Trimesh) -> float:
    """Robust 'how big is this lump' score, used to pick the subject component."""
    try:
        v = float(abs(part.convex_hull.volume))
        if np.isfinite(v) and v > 0.0:
            return v
    except Exception:
        pass
    return float(part.area)


def _cut_and_select(mesh: trimesh.Trimesh, spec: BoardSpec, z: float,
                    cap: bool = True) -> tuple[trimesh.Trimesh, bool, str | None, int, int]:
    """Slice at z, discard off-board debris, return the largest survivor.

    Returns (component, capped, cap_error, n_components, n_on_board).
    """
    capped = cap
    cap_error: str | None = None
    try:
        cut = mesh.slice_plane([0.0, 0.0, z], [0.0, 0.0, 1.0], cap=cap)
    except Exception as exc:
        # Capping needs shapely + mapbox_earcut, and rtree as soon as more than
        # one component crosses the plane. Without them take the open cut and
        # let make_watertight() try to close it -- but say so, because an
        # uncapped base is the difference between a solid and a shell.
        if not cap:
            raise
        capped = False
        cap_error = f"{type(exc).__name__}: {exc}"
        cut = mesh.slice_plane([0.0, 0.0, z], [0.0, 0.0, 1.0], cap=False)
    if cut is None or len(cut.faces) == 0:
        raise MeshError(f"remove_ground: slice at z={z} mm produced no geometry")

    parts = cut.split(only_watertight=False)
    if len(parts) == 0:
        parts = [cut]

    # Board footprint, generously padded: the subject may overhang the target,
    # but the table edge and the operator's hands will not be anywhere near.
    m = GROUND_LATERAL_MARGIN_MM
    x0, x1 = -m, spec.width_mm + m
    y0, y1 = -m, spec.height_mm + m
    near = []
    for part in parts:
        c = part.centroid
        if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
            near.append(part)
    if not near:
        near = parts   # everything is off-board; better to keep the biggest than nothing

    return max(near, key=_component_size), capped, cap_error, len(parts), len(near)


def _edge_keys(faces: np.ndarray, n_vert: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges of every face plus an order-independent int64 key each."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lo = e.min(axis=1).astype(np.int64)
    hi = e.max(axis=1).astype(np.int64)
    return e, lo * np.int64(n_vert) + hi


def _wall_slopes(mesh: trimesh.Trimesh, wall_mask: np.ndarray) -> np.ndarray:
    """Per-vertex d(xy)/d(-z) of the surface, from the adjacent wall faces.

    A point stays on a locally planar wall of unit normal n when it moves by
    dxy = dz * n_z * n_xy / |n_xy|^2 as it descends dz. For a cone of taper t
    (n = (cos t * radial, sin t)) that reduces to exactly dz*tan(t) outwards --
    i.e. the true footprint, which is the whole point of doing this. The sign
    takes care of itself: a wall that leans in below the cut, like a sphere
    resting on the board, produces an inward slope and a shrinking base.

    Returned unclamped; the caller applies both caps.
    """
    faces = mesh.faces[wall_mask]
    n_vert = len(mesh.vertices)
    slope = np.zeros((n_vert, 2), dtype=np.float64)
    if len(faces) == 0:
        return slope

    # Area-weighted normal sum over the wall faces touching each vertex. Cap
    # faces are excluded by the mask: their normal is -Z and carries no
    # information about which way the wall leans.
    w = mesh.area_faces[wall_mask][:, None] * mesh.face_normals[wall_mask]
    idx = faces.ravel()
    acc = np.empty((n_vert, 3), dtype=np.float64)
    for c in range(3):
        acc[:, c] = np.bincount(idx, weights=np.repeat(w[:, c], 3), minlength=n_vert)

    nxy = acc[:, :2]
    l2 = (nxy * nxy).sum(axis=1)
    good = l2 > 1e-24
    slope[good] = (acc[good, 2] / l2[good])[:, None] * nxy[good]
    return slope


def _extrude_base_to_ground(mesh: trimesh.Trimesh, z_tol: float,
                            capped: bool) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Replace the flat z=z_tol cut with a skirt extruded down to z=0.

    The old behaviour -- snapping every cut vertex to z=0 -- reproduced the
    cross-section 1 mm up as the footprint and so undersized every tapered
    object while getting the height exactly right, which is precisely the shape
    of error that no report catches. Here the cut ring is left where it is and a
    new ring is built at z=0, offset laterally along the local wall direction,
    so the geometry above the cut is untouched and the base is the extrapolated
    footprint rather than a copy of the cross-section.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    n = len(v)
    on_cut = v[:, 2] <= z_tol + 1e-6
    if not on_cut.any():
        raise MeshError(
            f"remove_ground: nothing sits on the z={z_tol} mm cut plane, so the "
            "base cannot be extruded to the board; the subject is floating above it"
        )

    cap_mask = on_cut[f].all(axis=1)
    slope = _wall_slopes(mesh, ~cap_mask)

    if capped and cap_mask.any():
        # The skirt takes the cap's place, so it inherits the cap's winding.
        cap_e, cap_k = _edge_keys(f[cap_mask], n)
        _, wall_k = _edge_keys(f[~cap_mask], n)
        ring = cap_e[np.isin(cap_k, wall_k)]
    else:
        # Open cut: the skirt hangs off the boundary, so it runs the other way
        # from the single wall face that owns each boundary edge.
        all_e, all_k = _edge_keys(f, n)
        _, inv, cnt = np.unique(all_k, return_inverse=True, return_counts=True)
        ring = all_e[cnt[inv] == 1][:, ::-1]
        ring = ring[on_cut[ring].all(axis=1)]
    if len(ring) == 0:
        raise MeshError(
            f"remove_ground: found no boundary ring at the z={z_tol} mm cut, so the "
            "base cannot be extruded to the board"
        )

    ring_v = np.unique(ring)
    dz = float(z_tol)
    # Cap the extrapolation twice over: by wall slope, and by the ring's own
    # size so an inward-leaning small feature cannot be pushed through itself.
    ring_xy = v[ring_v, :2]
    ring_span = float(min(np.ptp(ring_xy[:, 0]), np.ptp(ring_xy[:, 1])))
    slope_cap = BASE_MAX_TAPER_SLOPE
    if ring_span > 0.0 and dz > 0.0:
        slope_cap = min(slope_cap, BASE_MAX_MOVE_FRAC * ring_span / dz)
    mag = np.linalg.norm(slope, axis=1)
    hot = mag > slope_cap
    if hot.any():
        slope = slope.copy()
        slope[hot] *= (slope_cap / mag[hot])[:, None]

    oc = np.flatnonzero(on_cut)
    newidx = np.full(n, -1, dtype=np.int64)
    newidx[oc] = n + np.arange(len(oc), dtype=np.int64)
    ground = np.column_stack([v[oc, :2] + dz * slope[oc], np.zeros(len(oc))])
    verts = np.vstack([v, ground])

    p, q = ring[:, 0], ring[:, 1]
    pp, qq = newidx[p], newidx[q]
    skirt = np.vstack([np.column_stack([p, q, qq]),
                       np.column_stack([p, qq, pp])])
    if capped and cap_mask.any():
        faces = np.vstack([f[~cap_mask], skirt, newidx[f[cap_mask]]])
    else:
        faces = np.vstack([f, skirt])

    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.merge_vertices()
    out.remove_unreferenced_vertices()

    fixed = verts[newidx[ring_v], :2]
    taper = np.degrees(np.arctan(np.linalg.norm(slope[ring_v], axis=1)))
    info = {
        "applied": True,
        "z_mm": round(dz, 4),
        "ring_vertices": int(len(ring_v)),
        "taper_deg_median": round(float(np.median(taper)), 3),
        "taper_deg_p95": round(float(np.percentile(taper, 95.0)), 3),
        "slope_cap": round(float(slope_cap), 4),
        "clamped_ring_vertices": int(hot[ring_v].sum()),
        # What the old collapse-to-z=0 behaviour would have reported, next to
        # what the extrusion actually produced. The difference is the bug.
        "footprint_collapsed_mm": [round(float(np.ptp(ring_xy[:, 0])), 4),
                                   round(float(np.ptp(ring_xy[:, 1])), 4)],
        "footprint_extruded_mm": [round(float(np.ptp(fixed[:, 0])), 4),
                                  round(float(np.ptp(fixed[:, 1])), 4)],
    }
    info["footprint_gain_mm"] = [round(a - b, 4) for a, b in
                                 zip(info["footprint_extruded_mm"],
                                     info["footprint_collapsed_mm"])]
    return out, info


def _footprint_sensitivity(mesh: trimesh.Trimesh, spec: BoardSpec, z_tol: float,
                           flatten_base: bool) -> dict[str, Any]:
    """Re-cut at several heights and report how much the footprint moves.

    This is the honest answer to "is the base trustworthy". Two independent
    things attack the bottom of the model -- Object Capture truncates the object
    where it contacts the support surface, and the board plane leaks back into
    the kept component as the cut approaches z=0 (measured on a real model:
    footprint 47.2 mm at z_tol=1.0, 53.8 at 0.5, 79.2 at 0.25, 202.7 at 0.1).
    Neither is recoverable from the mesh, but both are visible as a footprint
    that depends on where the cut was taken, and that is worth saying out loud.
    """
    probes: list[dict[str, Any]] = []
    for k in FOOTPRINT_PROBE_FACTORS:
        z = z_tol * float(k)
        row: dict[str, Any] = {"z_mm": round(z, 4), "factor": float(k)}
        try:
            part, _, _, _, _ = _cut_and_select(mesh, spec, z, cap=False)
            row["component_extent_mm"] = [round(float(v), 4) for v in part.extents[:2]]
            # Measure the base ring itself, not the component's bounding box:
            # on anything wider above than below (a sphere, a mushroom) the box
            # is dominated by geometry far from the cut and would report a rock
            # steady footprint while the base moved.
            _, info = _extrude_base_to_ground(part, z, capped=False)
            row["footprint_raw_mm"] = info["footprint_collapsed_mm"]
            row["footprint_mm"] = (info["footprint_extruded_mm"] if flatten_base
                                   else info["footprint_collapsed_mm"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        probes.append(row)

    ok = [r for r in probes if "footprint_mm" in r]
    out: dict[str, Any] = {"probes": probes,
                           "warn_pct_per_mm": FOOTPRINT_SENSITIVITY_WARN_PCT}
    at_or_above = [r for r in ok if r["factor"] >= 1.0]
    below = [r for r in ok if r["factor"] < 1.0]
    here = next((r for r in ok if r["factor"] == 1.0), None)

    if here is not None and below:
        # Not a verdict, a margin: how hard the board plane is pressing on the
        # cut from underneath.
        ref = max(here["footprint_mm"]) or 1.0
        leak = max(max(r["footprint_mm"]) - max(here["footprint_mm"]) for r in below)
        out["leak_below_cut_pct"] = round(100.0 * leak / ref, 3)
        if out["leak_below_cut_pct"] > FOOTPRINT_LEAK_NOTE_PCT:
            out["leak_note"] = (
                f"cutting at {below[0]['z_mm']} mm instead of {here['z_mm']} mm grows the "
                f"footprint by {out['leak_below_cut_pct']}%: that is the board plane, not "
                "the object. The cut has little margin below it."
            )

    if len(at_or_above) < 2:
        out["sensitivity_pct_per_mm"] = None
        out["trustworthy"] = None
        return out

    ref = max(max(r["footprint_mm"]) for r in at_or_above)
    worst = 0.0
    for a, b in zip(at_or_above, at_or_above[1:]):
        dz = abs(b["z_mm"] - a["z_mm"])
        if dz <= 1e-9:
            continue
        d = max(abs(b["footprint_mm"][i] - a["footprint_mm"][i]) for i in (0, 1))
        worst = max(worst, d / dz)
    pct = 100.0 * worst / ref if ref > 0.0 else float("inf")
    out["sensitivity_mm_per_mm"] = round(worst, 4)
    out["sensitivity_pct_per_mm"] = round(pct, 3)
    out["trustworthy"] = bool(pct <= FOOTPRINT_SENSITIVITY_WARN_PCT)
    return out


def remove_ground(mesh: trimesh.Trimesh, spec: BoardSpec,
                  z_tol: float = DEFAULT_Z_TOL_MM,
                  flatten_base: bool = True,
                  probe_sensitivity: bool = True) -> trimesh.Trimesh:
    """Cut away the board/table plane and keep the subject.

    In BOARD millimetres the printed target lies on z = 0, so everything the user
    cares about is above it. Slices at z = z_tol (capping the cut so the result
    stays closed), drops components whose centroid falls well outside the board
    footprint, then keeps the single largest remaining component.

    WHY flatten_base DEFAULTS TO TRUE: the cut has to sit slightly above the
    board or it eats the board plane itself, but that means the returned solid is
    z_tol mm shorter than the real object -- a systematic height error of exactly
    the tolerance, on every scan. Since the object genuinely rests on z = 0 the
    base is rebuilt down at z = 0, which restores true height and leaves the
    solid closed. It is rebuilt by extruding the cut ring along the local wall
    direction, NOT by collapsing it: see _extrude_base_to_ground. Pass
    flatten_base=False for an object that is not actually touching the board.

    Everything the cut removed, and how sensitive the recovered footprint is to
    where the cut was taken, is recorded in metadata['ground_cut'],
    metadata['base_extrusion'] and metadata['footprint_confidence'], and
    surfaced by stats().
    """
    if len(mesh.faces) == 0:
        raise MeshError("remove_ground: empty mesh")
    mesh = mesh.copy()
    # A reconstruction loaded with process=False has split vertices, and
    # slice_plane cannot cap a surface whose boundary is not topologically
    # connected. Merging first is what makes the cap watertight.
    mesh.merge_vertices()
    lo, hi = mesh.bounds[0][2], mesh.bounds[1][2]
    if hi <= z_tol:
        raise MeshError(
            f"remove_ground: nothing above z={z_tol} mm -- mesh z range is "
            f"[{lo:.2f}, {hi:.2f}] mm. The mesh is probably not in the BOARD frame, "
            "or the similarity transform is upside down."
        )

    best, capped, cap_error, n_parts, n_near = _cut_and_select(mesh, spec, z_tol, cap=True)
    out = best.copy()

    # What the cut actually threw away, measured on the input mesh.
    below = mesh.vertices[:, 2] <= z_tol
    below_faces = int(below[mesh.faces].all(axis=1).sum())
    below_v = mesh.vertices[below]
    ground_cut: dict[str, Any] = {
        "z_tol_mm": round(float(z_tol), 4),
        "input_z_range_mm": [round(float(lo), 4), round(float(hi), 4)],
        "faces_removed": below_faces,
        "faces_removed_frac": round(below_faces / max(len(mesh.faces), 1), 5),
        "removed_extent_mm": ([round(float(np.ptp(below_v[:, 0])), 3),
                               round(float(np.ptp(below_v[:, 1])), 3)]
                              if len(below_v) else [0.0, 0.0]),
        "kept_extent_mm": [round(float(v), 4) for v in out.extents[:2]],
    }
    if lo > z_tol:
        ground_cut["note"] = (
            f"the mesh starts at z={lo:.2f} mm, above the z={z_tol} mm cut: the cut "
            "removed nothing, and the subject does not reach the board. Object "
            "Capture truncates the object where it meets the support surface, so "
            "the missing height is real and is NOT reconstructed here."
        )
    elif ground_cut["removed_extent_mm"][0] > 3.0 * max(ground_cut["kept_extent_mm"][0], 1e-9):
        ground_cut["note"] = (
            "the geometry below the cut is far wider than the subject, i.e. it is "
            "the board/table plane, which is what the cut is for"
        )

    base_info: dict[str, Any] = {"applied": False}
    if flatten_base:
        out, base_info = _extrude_base_to_ground(out, z_tol, capped)
        out.metadata["base_flattened_mm"] = float(z_tol)

    out.metadata["capped"] = capped
    if not capped:
        out.metadata["cap_error"] = cap_error
    out.metadata["components_seen"] = n_parts
    out.metadata["components_on_board"] = n_near
    out.metadata["ground_cut"] = ground_cut
    out.metadata["base_extrusion"] = base_info
    out.metadata["footprint_confidence"] = (
        _footprint_sensitivity(mesh, spec, z_tol, flatten_base)
        if probe_sensitivity else {"probes": [], "trustworthy": None,
                                   "sensitivity_pct_per_mm": None}
    )
    return out


# --- repair ----------------------------------------------------------------

def make_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean degenerate/duplicate faces, fix winding and normals, fill holes.

    Does NOT guarantee success. The result carries metadata['watertight'] and
    metadata['winding_consistent'] so callers can report the truth instead of
    assuming a solid came out.
    """
    out = mesh.copy()
    out.merge_vertices()
    if len(out.faces):
        out.update_faces(out.nondegenerate_faces())
    if len(out.faces):
        out.update_faces(out.unique_faces())
    out.remove_infinite_values()
    out.remove_unreferenced_vertices()
    if len(out.faces) == 0:
        raise MeshError("make_watertight: every face was degenerate or duplicated")

    trimesh.repair.fix_winding(out)
    trimesh.repair.fix_inversion(out)

    for _ in range(3):
        if out.is_watertight:
            break
        if not out.fill_holes():
            break
        out.merge_vertices()
        out.remove_unreferenced_vertices()

    trimesh.repair.fix_winding(out)
    trimesh.repair.fix_normals(out)

    # manifold3d self-union resolves self-intersections and re-extracts a clean
    # boundary. Only legal on a mesh that is already a volume, so it is a polish
    # pass, not a repair pass.
    if out.is_volume:
        try:
            polished = trimesh.boolean.union([out], engine="manifold")
            if polished is not None and len(polished.faces) and polished.is_watertight:
                before = out.extents.copy()
                if np.abs(polished.extents - before).max() < DECIMATE_BBOX_TOL_MM:
                    # boolean returns a fresh mesh; carry the provenance the
                    # ground cut recorded, or stats() loses it silently.
                    polished.metadata.update(out.metadata)
                    out = polished
        except Exception:
            pass   # polish is optional; the trimesh-repaired mesh still stands

    out.metadata["watertight"] = bool(out.is_watertight)
    out.metadata["winding_consistent"] = bool(out.is_winding_consistent)
    return out


# --- print preparation -----------------------------------------------------

def _largest_hull_facet_transform(mesh: trimesh.Trimesh) -> np.ndarray:
    """Fallback orientation: put the biggest flat facet of the convex hull down."""
    hull = mesh.convex_hull
    areas = hull.area_faces
    if len(hull.facets):
        facet_areas = np.array([areas[f].sum() for f in hull.facets])
        normal = hull.facets_normal[int(np.argmax(facet_areas))]
        if facet_areas.max() < areas.max():
            normal = hull.face_normals[int(np.argmax(areas))]
    else:
        normal = hull.face_normals[int(np.argmax(areas))]
    return trimesh.geometry.align_vectors(np.asarray(normal, dtype=np.float64),
                                          np.array([0.0, 0.0, -1.0]))


def _align_footprint_yaw(mesh: trimesh.Trimesh) -> float:
    """Spin about Z until the footprint's minimum-area rectangle is axis aligned.

    compute_stable_poses() only constrains which face points down; the remaining
    yaw is arbitrary and varies with tiny numerical differences in the input. An
    arbitrary yaw inflates the axis-aligned bounding box (a 50x30 footprint spun
    37 degrees reports 57x46), which both misreports the part's size and wastes
    bed width. Returns the applied yaw in degrees.
    """
    try:
        T2, ext2 = trimesh.bounds.oriented_bounds_2D(mesh.vertices[:, :2])
    except Exception:
        return 0.0
    R2 = np.asarray(T2, dtype=np.float64)[:2, :2]
    if float(np.linalg.det(R2)) < 0.0:
        R2 = np.array([[1.0, 0.0], [0.0, -1.0]]) @ R2   # back to a pure rotation
    if ext2[0] < ext2[1]:                                # long side along X
        R2 = np.array([[0.0, -1.0], [1.0, 0.0]]) @ R2
    M = np.eye(4)
    M[:2, :2] = R2
    mesh.apply_transform(M)
    return float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0])))


def orient_for_print(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rest the object on its most stable face at z=0, centred on the bed origin."""
    if len(mesh.faces) == 0:
        raise MeshError("orient_for_print: empty mesh")
    out = mesh.copy()

    T = None
    if out.is_volume:
        try:
            transforms, probs = trimesh.poses.compute_stable_poses(out)
            if len(transforms):
                T = np.asarray(transforms[int(np.argmax(probs))], dtype=np.float64)
                out.metadata["stable_pose_probability"] = float(np.max(probs))
        except Exception:
            T = None
    if T is None:
        T = _largest_hull_facet_transform(out)
        out.metadata["stable_pose_probability"] = None
        out.metadata["orient_method"] = "largest_hull_facet"
    else:
        out.metadata["orient_method"] = "stable_pose"
    out.apply_transform(T)
    out.metadata["yaw_deg"] = _align_footprint_yaw(out)

    lo, hi = out.bounds
    out.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    return out


def decimate(mesh: trimesh.Trimesh, max_faces: int = 200_000) -> trimesh.Trimesh:
    """Reduce to at most max_faces triangles, refusing a reduction that moves the bbox."""
    if max_faces < 4:
        raise MeshError(f"decimate: max_faces={max_faces} is below a closed surface")
    n = len(mesh.faces)
    if n <= max_faces:
        return mesh.copy()

    before = mesh.extents.copy()
    try:
        out = mesh.simplify_quadric_decimation(face_count=int(max_faces))
    except Exception as exc:
        raise MeshError(
            f"decimate: quadric decimation failed ({exc}). "
            "`uv add fast-simplification` provides the backend."
        ) from exc
    if out is None or len(out.faces) == 0:
        raise MeshError("decimate: simplification returned an empty mesh")

    drift = float(np.abs(out.extents - before).max())
    if drift > DECIMATE_BBOX_TOL_MM:
        raise MeshError(
            f"decimate: simplifying {n} -> {len(out.faces)} faces moved the bounding "
            f"box by {drift:.3f} mm (limit {DECIMATE_BBOX_TOL_MM} mm). Raise max_faces."
        )
    out.metadata.update(mesh.metadata)
    out.metadata["decimated_from_faces"] = n
    out.metadata["decimate_bbox_drift_mm"] = drift
    return out


def stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Measured properties of a mesh assumed to be in millimetres."""
    lo, hi = mesh.bounds
    ext = mesh.extents
    watertight = bool(mesh.is_watertight)
    # The axis-aligned box depends on how orient_for_print() chose to rest the
    # part; the oriented box does not. For "how big is this object really" the
    # oriented box is the number to quote.
    try:
        obb = sorted((float(v) for v in mesh.bounding_box_oriented.primitive.extents),
                     reverse=True)
    except Exception:
        obb = None
    vol_cm3 = float(abs(mesh.volume) / 1000.0) if watertight else None
    bx, by = config.BED_MM
    fits = bool(ext[0] <= bx and ext[1] <= by and ext[2] <= config.BED_HEIGHT_MM)
    # Base provenance, recorded by remove_ground(). Absent when stats() is
    # called on a mesh that never went through the ground cut -- report None
    # rather than implying the base was checked.
    ground_cut = mesh.metadata.get("ground_cut")
    base_extrusion = mesh.metadata.get("base_extrusion")
    conf = mesh.metadata.get("footprint_confidence") or {}
    return {
        "bbox_mm": [round(float(v), 4) for v in ext],
        "obb_mm": None if obb is None else [round(v, 4) for v in obb],
        "bbox_min_mm": [round(float(v), 4) for v in lo],
        "bbox_max_mm": [round(float(v), 4) for v in hi],
        "volume_cm3": None if vol_cm3 is None else round(vol_cm3, 4),
        "surface_area_cm2": round(float(mesh.area) / 100.0, 4),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": watertight,
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "euler_number": int(mesh.euler_number),
        "bed_mm": [bx, by, config.BED_HEIGHT_MM],
        "fits_bed": fits,
        "bed_margin_mm": [round(float(bx - ext[0]), 3),
                          round(float(by - ext[1]), 3),
                          round(float(config.BED_HEIGHT_MM - ext[2]), 3)],
        # --- how much the base can be trusted -----------------------------
        "ground_cut": ground_cut,
        "base_extrusion": base_extrusion,
        "footprint_confidence": conf or None,
        "footprint_trustworthy": conf.get("trustworthy"),
        "footprint_sensitivity_pct_per_mm": conf.get("sensitivity_pct_per_mm"),
    }


# --- full chain ------------------------------------------------------------

def condition(raw_path: str | Path, res: ScaleResult, spec: BoardSpec,
              out_stl: str | Path, z_tol: float = DEFAULT_Z_TOL_MM,
              max_faces: int = 200_000, flatten_base: bool = True,
              allow_reject: bool = False) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Raw reconstruction -> printable metric STL. Returns (mesh, stats)."""
    raw = Path(raw_path).expanduser()
    out_stl = Path(out_stl).expanduser()
    out_stl.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"source": str(raw), "stl": str(out_stl), "steps": []}

    def step(name: str, m: trimesh.Trimesh, **extra: Any) -> None:
        report["steps"].append({
            "step": name,
            "faces": int(len(m.faces)),
            "bbox_mm": [round(float(v), 4) for v in m.extents],
            "watertight": bool(m.is_watertight),
            **extra,
        })

    m = load(raw)
    step("load", m, units="RECON (raw, unconverted)")
    report["usd_metersPerUnit"] = m.metadata.get("usd_metersPerUnit")
    report["usd_upAxis"] = m.metadata.get("usd_upAxis")

    m = apply_scale(m, res, allow_reject=allow_reject)
    step("apply_scale", m, scale=float(res.scale), rms_mm=float(res.rms_mm),
         quality=res.quality)

    m = remove_ground(m, spec, z_tol=z_tol, flatten_base=flatten_base)
    # Held separately: every downstream step rebuilds the mesh, and a repair or
    # boolean that drops metadata must not be able to drop the base provenance
    # with it. These are re-injected into the report below.
    base_meta = {k: m.metadata.get(k)
                 for k in ("ground_cut", "base_extrusion", "footprint_confidence")}
    step("remove_ground", m,
         components_seen=m.metadata.get("components_seen"),
         components_on_board=m.metadata.get("components_on_board"),
         base_flattened_mm=m.metadata.get("base_flattened_mm"),
         base_footprint_gain_mm=(base_meta["base_extrusion"] or {}).get("footprint_gain_mm"),
         capped=m.metadata.get("capped"), cap_error=m.metadata.get("cap_error"))

    m = make_watertight(m)
    step("make_watertight", m, winding_consistent=bool(m.is_winding_consistent))

    m = decimate(m, max_faces=max_faces)
    step("decimate", m, drift_mm=m.metadata.get("decimate_bbox_drift_mm"))

    if not m.is_watertight:
        m = make_watertight(m)
        step("make_watertight(2)", m)

    m = orient_for_print(m)
    step("orient_for_print", m, method=m.metadata.get("orient_method"),
         yaw_deg=m.metadata.get("yaw_deg"),
         stable_pose_probability=m.metadata.get("stable_pose_probability"))

    m.export(out_stl)
    if not out_stl.exists() or out_stl.stat().st_size == 0:
        raise MeshError(f"condition: STL export produced nothing at {out_stl}")

    report.update(stats(m))
    report.update(base_meta)
    conf = base_meta["footprint_confidence"] or {}
    report["footprint_trustworthy"] = conf.get("trustworthy")
    report["footprint_sensitivity_pct_per_mm"] = conf.get("sensitivity_pct_per_mm")
    report["stl_bytes"] = out_stl.stat().st_size

    gc = base_meta["ground_cut"] or {}
    if conf.get("trustworthy") is False:
        report.setdefault("warnings", []).append(
            f"FOOTPRINT NOT TRUSTWORTHY: moving the z={z_tol} mm ground cut moves the "
            f"recovered footprint by {conf.get('sensitivity_mm_per_mm')} mm per mm of "
            f"cut height ({conf.get('sensitivity_pct_per_mm')}%/mm, limit "
            f"{FOOTPRINT_SENSITIVITY_WARN_PCT}%/mm). The board plane and/or Object "
            f"Capture's truncation of the contact patch is reaching into the base -- "
            f"measure the base with calipers before trusting it. Probes: "
            f"{conf.get('probes')}"
        )
    if conf.get("leak_note"):
        report.setdefault("warnings", []).append(
            f"ground cut has little margin: {conf['leak_note']}"
        )
    be = base_meta["base_extrusion"] or {}
    if be.get("clamped_ring_vertices"):
        report.setdefault("warnings", []).append(
            f"{be['clamped_ring_vertices']} of {be['ring_vertices']} base ring vertices "
            f"sit on a near-horizontal wall and had their extrusion clamped at slope "
            f"{BASE_MAX_TAPER_SLOPE}; the footprint under those is a floor, not a "
            "measurement"
        )
    if gc.get("note") and "does not reach the board" in gc["note"]:
        report.setdefault("warnings", []).append(
            f"the reconstruction's lowest point is z={gc['input_z_range_mm'][0]} mm, "
            f"above the z={z_tol} mm ground cut: the object never reaches the board, so "
            "the reported height is short by an unknown amount (Object Capture truncates "
            "the contact patch). No geometry has been invented to fill it."
        )
    if not report["watertight"]:
        report.setdefault("warnings", []).append(
            "mesh is NOT watertight after conditioning; the slicer may produce a "
            "hollow or malformed part"
        )
    if not report["fits_bed"]:
        report.setdefault("warnings", []).append(
            f"part {report['bbox_mm']} mm does not fit the "
            f"{config.BED_MM[0]}x{config.BED_MM[1]}x{config.BED_HEIGHT_MM} mm bed"
        )
    return m, report


# --- self-test -------------------------------------------------------------

def _truth_scene() -> tuple[trimesh.Trimesh, np.ndarray]:
    """BOARD-frame ground truth: board+table slab, the 50x30x20 subject, and debris.

    The far debris lump is deliberately *larger* than the subject, so if the
    board-footprint filter in remove_ground() ever stops working the self-test
    fails loudly with a 40x40x40 bbox instead of quietly passing.
    """
    spec = BoardSpec()
    cx, cy = spec.width_mm / 2.0, spec.height_mm / 2.0

    slab = trimesh.creation.box(extents=[400.0, 340.0, 4.0])
    slab.apply_translation([cx, cy, -2.0])

    subject = trimesh.creation.box(extents=[50.0, 30.0, 20.0])
    subject.apply_translation([cx, cy, 10.0])

    floater = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    floater.apply_translation([cx - 60.0, cy - 40.0, 35.0])

    far_debris = trimesh.creation.box(extents=[40.0, 40.0, 40.0])
    far_debris.apply_translation([520.0, 480.0, 20.0])

    scene = trimesh.util.concatenate([slab, subject, floater, far_debris])
    return scene, np.array([50.0, 30.0, 20.0])


def _taper_scene(taper_deg: float, r0: float = 20.0, h: float = 15.0,
                 sections: int = 256) -> trimesh.Trimesh:
    """A truncated cone of exactly known base diameter, sitting on a board slab.

    r(z) = r0 - z*tan(taper), so the base diameter at z=0 is exactly 2*r0 and
    the polygon vertices lie exactly on that circle. This is the shape that
    exposed the flatten_base bug: the height always came out right, so nothing
    in the report flagged that the footprint was z_tol*tan(taper) small per side.
    """
    spec = BoardSpec()
    cx, cy = spec.width_mm / 2.0, spec.height_mm / 2.0
    slope = np.tan(np.deg2rad(taper_deg))
    ang = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    zs = np.arange(0.0, h + 1e-9, 0.25)
    verts = [[(r0 - z * slope) * np.cos(a), (r0 - z * slope) * np.sin(a), z]
             for z in zs for a in ang]
    faces: list[list[int]] = []
    for iz in range(len(zs) - 1):
        for i in range(sections):
            j = (i + 1) % sections
            a, b = iz * sections + i, iz * sections + j
            c, d = (iz + 1) * sections + i, (iz + 1) * sections + j
            faces += [[a, b, d], [a, d, c]]
    bot, top = len(verts), len(verts) + 1
    verts += [[0.0, 0.0, 0.0], [0.0, 0.0, h]]
    for i in range(sections):
        j = (i + 1) % sections
        faces += [[bot, j, i],
                  [top, (len(zs) - 1) * sections + i, (len(zs) - 1) * sections + j]]
    cone = trimesh.Trimesh(vertices=np.array(verts, dtype=np.float64),
                           faces=np.array(faces, dtype=np.int64), process=True)
    cone.apply_translation([cx, cy, 0.0])
    slab = trimesh.creation.box(extents=[400.0, 340.0, 4.0])
    slab.apply_translation([cx, cy, -2.0])
    return trimesh.util.concatenate([cone, slab])


def _base_diameter(m: trimesh.Trimesh) -> float:
    """2x the max radius of the ring on the solid's own lowest plane."""
    v = m.vertices
    ring = v[v[:, 2] <= v[:, 2].min() + 1e-4]
    c = np.array([BoardSpec().width_mm / 2.0, BoardSpec().height_mm / 2.0])
    return 2.0 * float(np.linalg.norm(ring[:, :2] - c, axis=1).max())


def _fake_recon(truth: trimesh.Trimesh, s: float, R: np.ndarray,
                t: np.ndarray) -> trimesh.Trimesh:
    """Push BOARD-frame truth backwards through the similarity into RECON."""
    inv = np.eye(4)
    inv[:3, :3] = R.T / s
    inv[:3, 3] = -(R.T @ t) / s
    out = truth.copy()
    out.apply_transform(inv)
    return out


_BLENDER_TO_USDZ = r'''
import bpy, sys
argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=src, forward_axis="Y", up_axis="Z", global_scale=1.0)
bpy.ops.wm.usd_export(filepath=dst, export_materials=False, export_uvmaps=False,
                      convert_orientation=False)   # Z-up stage: stored points == input
print("scan2print: wrote", dst)
'''


def _selftest() -> int:
    np.set_printoptions(suppress=True)
    ok = True
    spec = BoardSpec()
    truth, want = _truth_scene()

    # A deliberately awkward similarity: big scale factor, off-axis rotation,
    # large translation.
    s_true = 250.0
    axis = np.array([0.37, -0.62, 0.69]); axis /= np.linalg.norm(axis)
    R_true = trimesh.transformations.rotation_matrix(np.deg2rad(41.0), axis)[:3, :3]
    t_true = np.array([-88.0, 141.0, 27.5])
    res = ScaleResult(scale=s_true, rms_mm=0.42, n_cameras=64,
                      R=R_true.tolist(), t=t_true.tolist(),
                      inlier_frac=0.97, quality="good", notes=[])

    recon = _fake_recon(truth, s_true, R_true, t_true)
    print("=" * 74)
    print("SYNTHETIC SCALE-SURVIVAL TEST")
    print("=" * 74)
    print(f"truth subject bbox      : {want} mm")
    print(f"truth scene bbox        : {np.round(truth.extents, 3)} mm")
    print(f"fake RECON scene bbox   : {np.round(recon.extents, 6)} (arbitrary units)")
    print(f"similarity              : scale={s_true}  |t|={np.linalg.norm(t_true):.2f} mm"
          f"  det(R)={np.linalg.det(R_true):+.6f}")

    tmp = Path(tempfile.mkdtemp(prefix="scan2print_mesh_"))
    cases: list[tuple[str, Path]] = []

    ply = tmp / "recon.ply"
    recon.export(ply)
    cases.append(("PLY (trimesh native)", ply))

    usdz = tmp / "recon.usdz"
    if config.BLENDER_BIN.exists():
        script = tmp / "to_usdz.py"
        script.write_text(_BLENDER_TO_USDZ)
        try:
            _run([str(config.BLENDER_BIN), "--background", "--factory-startup",
                  "--python", str(script), "--", str(ply), str(usdz)],
                 BLENDER_TIMEOUT_S, "blender PLY->USDZ")
            cases.append(("USDZ (Z-up stage, via Blender)", usdz))
        except MeshError as exc:
            print(f"\n!! could not author a test USDZ: {exc}")
            ok = False
    else:
        print("\n!! Blender missing; USDZ path not exercised")
        ok = False

    for label, path in cases:
        print("\n" + "-" * 74)
        print(f"CASE: {label}   ({path.name}, {path.stat().st_size} bytes)")
        print("-" * 74)
        if path.suffix.lower() in USD_EXTS:
            print(f"  stage metadata        : {usd_layer_metadata(path)}")
        try:
            raw = load(path)
            err_raw = np.abs(np.sort(raw.extents) - np.sort(recon.extents)).max()
            print(f"  loaded RECON extents  : {np.round(raw.extents, 6)}"
                  f"   (max err vs authored: {err_raw:.3e})")
            if err_raw > 1e-4 * max(recon.extents):
                print("  !! LOAD CHANGED THE GEOMETRY -- units or axes were converted")
                ok = False

            out_stl = tmp / (path.stem + "_conditioned.stl")
            m, rep = condition(path, res, spec, out_stl)
        except Exception as exc:
            print(f"  !! FAILED: {type(exc).__name__}: {exc}")
            ok = False
            continue

        for st in rep["steps"]:
            print(f"    {st['step']:<20} faces={st['faces']:>7}  "
                  f"bbox={np.round(st['bbox_mm'], 3)}  watertight={st['watertight']}")

        got = np.sort(np.array(rep["bbox_mm"], dtype=float))[::-1]
        exp = np.sort(want)[::-1]
        err = np.abs(got - exp)
        print(f"  recovered bbox (sorted): {got} mm")
        print(f"  expected  bbox (sorted): {exp} mm")
        print(f"  ABSOLUTE ERROR         : {np.round(err, 6)} mm   (max {err.max():.6f} mm)")
        print(f"  RELATIVE ERROR         : {np.round(100.0 * err / exp, 6)} %"
              f"   (max {(100.0 * err / exp).max():.6f} %)")
        print(f"  oriented bbox (rot-inv): {rep['obb_mm']} mm")
        print(f"  volume                 : {rep['volume_cm3']} cm3  "
              f"(truth {50 * 30 * 20 / 1000.0} cm3)")
        print(f"  watertight={rep['watertight']}  winding_consistent="
              f"{rep['winding_consistent']}  is_volume={rep['is_volume']}  "
              f"euler={rep['euler_number']}")
        print(f"  fits bed {rep['bed_mm']}: {rep['fits_bed']}  margin={rep['bed_margin_mm']}")
        print(f"  rests on z=0           : min_z={rep['bbox_min_mm'][2]}  "
              f"centred xy={rep['bbox_min_mm'][0] + rep['bbox_max_mm'][0]:.6f}, "
              f"{rep['bbox_min_mm'][1] + rep['bbox_max_mm'][1]:.6f}")
        if rep.get("warnings"):
            print(f"  warnings: {rep['warnings']}")

        if err.max() > 0.01:
            print("  !! DIMENSIONAL ERROR EXCEEDS 0.01 mm")
            ok = False
        if not rep["watertight"]:
            print("  !! NOT WATERTIGHT")
            ok = False

        back = load(out_stl)
        berr = np.abs(np.sort(back.extents)[::-1] - exp)
        print(f"  STL reload             : {np.round(np.sort(back.extents)[::-1], 6)} mm"
              f"  (max err {berr.max():.6f} mm, {rep['stl_bytes']} bytes)")
        if berr.max() > 0.01:
            print("  !! STL ROUND TRIP LOST DIMENSIONS")
            ok = False

    # --- Y-up stage: load() must return the literal stored points ----------
    print("\n" + "-" * 74)
    print("USD Y-UP / metersPerUnit HANDLING (exact stored-point recovery)")
    print("-" * 74)
    yup_pts = [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0]]
    yup = tmp / "yup.usda"
    yup.write_text(
        '#usda 1.0\n(\n    defaultPrim = "obj"\n    metersPerUnit = 0.01\n'
        '    upAxis = "Y"\n)\n\ndef Mesh "obj"\n{\n'
        "    int[] faceVertexCounts = [3, 3, 3, 3]\n"
        "    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]\n"
        "    point3f[] points = [(0, 0, 0), (7, 0, 0), (0, 5, 0), (0, 0, 3)]\n}\n")
    exp_pts = np.array(sorted(yup_pts))
    print(f"  stage metadata         : {usd_layer_metadata(yup)}")
    print(f"  points authored in file: {yup_pts}")
    got_pts = np.array(sorted(load(yup).vertices.tolist()))
    perr = float(np.abs(got_pts - exp_pts).max())
    print(f"  points returned by load: {np.round(got_pts, 6).tolist()}")
    print(f"  max coordinate error   : {perr:.3e}")
    if perr > 1e-5:
        print("  !! load() did not return raw stage coordinates")
        ok = False
    else:
        print("  OK: Blender's Y-up->Z-up rotation undone, metersPerUnit=0.01 NOT applied")

    fb = _usda_fallback_mesh(yup)
    ferr = float(np.abs(np.array(sorted(fb.vertices.tolist())) - exp_pts).max())
    print(f"  usdcat fallback reader : max coordinate error {ferr:.3e}")
    if ferr > 1e-5:
        print("  !! USDA fallback reader disagrees with the Blender route")
        ok = False

    # --- component / ground-plane isolation checks -------------------------
    print("\n" + "-" * 74)
    print("GROUND REMOVAL + COMPONENT SELECTION")
    print("-" * 74)
    board_frame = truth.copy()
    print(f"  input scene z range    : {np.round(board_frame.bounds[:, 2], 3)} mm, "
          f"{len(board_frame.split(only_watertight=False))} components")
    kept = remove_ground(board_frame, spec, z_tol=1.0)
    print(f"  after remove_ground    : bbox={np.round(kept.extents, 4)} mm, "
          f"components_seen={kept.metadata['components_seen']}, "
          f"on_board={kept.metadata['components_on_board']}")
    if np.abs(np.sort(kept.extents)[::-1] - np.array([50.0, 30.0, 20.0])).max() > 1e-6:
        print("  !! wrong component survived / base not flattened "
              "(expected the 50x30x20 subject)")
        ok = False
    else:
        print("  OK: slab, near floater and the larger off-board debris all removed")

    sunk = truth.copy()
    sunk.apply_translation([0.0, 0.0, -100.0])   # whole scene below the board plane
    try:
        remove_ground(sunk, spec, z_tol=1.0)
        print("  !! mesh entirely below the board was accepted instead of raising")
        ok = False
    except MeshError as exc:
        print(f"  OK: below-board mesh rejected -> {str(exc).splitlines()[0][:110]}")

    # --- tapered base: the footprint, not the cross-section 1 mm up ---------
    print("\n" + "-" * 74)
    print("TAPERED BASE (truncated cones, true base diameter 40.000 mm)")
    print("-" * 74)
    print("  collapsing the cut ring to z=0 reproduces the cross-section at z_tol as "
          "the\n  footprint, which undersizes every taper by z_tol*tan(taper) per side. "
          "The\n  height is unaffected, so nothing else in the report ever notices.")
    for z_tol in (1.0, DEFAULT_Z_TOL_MM):
        print(f"\n  z_tol = {z_tol} mm")
        print(f"    {'taper':>6} {'collapsed':>10} {'extruded':>10} {'err':>9} "
              f"{'height':>8} {'taper seen':>11} {'sens %/mm':>10}")
        for tp in (0.0, 15.0, 30.0, 45.0):
            got = remove_ground(_taper_scene(tp), spec, z_tol=z_tol)
            be = got.metadata["base_extrusion"]
            conf = got.metadata["footprint_confidence"]
            dia = _base_diameter(got)
            old = 40.0 - 2.0 * z_tol * np.tan(np.deg2rad(tp))   # the old behaviour
            print(f"    {tp:>5.0f}d {old:>10.3f} {dia:>10.3f} {dia - 40.0:>+9.4f} "
                  f"{got.extents[2]:>8.3f} {be['taper_deg_median']:>10.2f}d "
                  f"{conf['sensitivity_pct_per_mm']:>10}")
            if abs(dia - 40.0) > 0.01:
                print("    !! FOOTPRINT ERROR EXCEEDS 0.01 mm")
                ok = False
            if abs(got.extents[2] - 15.0) > 0.01:
                print("    !! HEIGHT ERROR EXCEEDS 0.01 mm")
                ok = False
            if not got.is_volume:
                print("    !! extruded base is not a closed volume")
                ok = False

    # A sphere touching the board has no real footprint at all, and the answer
    # genuinely depends on where the cut is taken. It must say so.
    print("\n  sphere r=20 mm tangent to the board (true footprint: a point)")
    ball = trimesh.creation.icosphere(subdivisions=5, radius=20.0)
    ball.apply_translation([spec.width_mm / 2.0, spec.height_mm / 2.0, 20.0])
    slab = trimesh.creation.box(extents=[400.0, 340.0, 4.0])
    slab.apply_translation([spec.width_mm / 2.0, spec.height_mm / 2.0, -2.0])
    got = remove_ground(trimesh.util.concatenate([ball, slab]), spec)
    conf = got.metadata["footprint_confidence"]
    print(f"    bbox {np.round(got.extents, 4)} mm  base dia "
          f"{_base_diameter(got):.3f} mm  is_volume={got.is_volume}")
    print(f"    footprint sensitivity {conf['sensitivity_pct_per_mm']} %/mm  "
          f"trustworthy={conf['trustworthy']}  "
          f"clamped {got.metadata['base_extrusion']['clamped_ring_vertices']}"
          f"/{got.metadata['base_extrusion']['ring_vertices']} ring vertices")
    if np.abs(np.sort(got.extents) - 40.0).max() > 0.05:
        print("    !! sphere bbox is wrong")
        ok = False
    if conf["trustworthy"] is not False:
        print("    !! an unresolvable footprint was reported as trustworthy")
        ok = False
    else:
        print("    OK: unresolvable footprint reported as NOT trustworthy")

    # --- reflection guard ---------------------------------------------------
    bad = ScaleResult(scale=s_true, rms_mm=0.4, n_cameras=64,
                      R=(R_true @ np.diag([1.0, 1.0, -1.0])).tolist(),
                      t=t_true.tolist(), inlier_frac=0.97, quality="good", notes=[])
    try:
        apply_scale(truth, bad)
        print("  !! mirrored ScaleResult was accepted")
        ok = False
    except MeshError as exc:
        print(f"  OK: reflection rejected -> {str(exc).splitlines()[0][:100]}")

    rej = ScaleResult(scale=s_true, rms_mm=99.0, n_cameras=4, R=R_true.tolist(),
                      t=t_true.tolist(), inlier_frac=0.2, quality="reject",
                      notes=["residual far too high"])
    try:
        apply_scale(truth, rej)
        print("  !! quality='reject' was accepted")
        ok = False
    except MeshError as exc:
        print(f"  OK: quality='reject' refused -> {str(exc).splitlines()[0][:100]}")

    # --- decimation ---------------------------------------------------------
    print("\n" + "-" * 74)
    print("DECIMATION")
    print("-" * 74)
    dense = trimesh.creation.icosphere(subdivisions=6, radius=25.0)
    small = decimate(dense, max_faces=20_000)
    print(f"  {len(dense.faces)} -> {len(small.faces)} faces, "
          f"bbox {np.round(dense.extents, 4)} -> {np.round(small.extents, 4)} mm, "
          f"drift {small.metadata['decimate_bbox_drift_mm']:.4f} mm, "
          f"watertight={small.is_watertight}")
    if len(small.faces) > 20_000:
        print("  !! decimation did not reach the target")
        ok = False

    print("\n" + "=" * 74)
    print("RESULT:", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    print("artifacts kept in", tmp)
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
