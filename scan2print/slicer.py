"""Slice an STL with the installed Orca-Flashforge CLI.

WHY THIS FILE IS LONGER THAN A subprocess.run() WRAPPER
-------------------------------------------------------
Orca's bundled Flashforge profiles are stored as a PrusaSlicer-style
inheritance tree ("Flashforge PLA Basic" inherits "Flashforge Generic PLA"
inherits "fdm_filament_pla" inherits "fdm_filament_common"). The GUI resolves
that tree. The CLI does NOT -- verified on this machine, Orca-Flashforge
1.6.2: handing it a leaf preset silently slices with Orca's built-in defaults
for every key the leaf file does not literally contain.

The damage is not cosmetic. Passing the leaf preset unflattened gave
15% infill -> 20%, no_brim -> auto_brim, outer wall 200 -> 60 mm/s,
filament density 1.24 -> 0 (so the gram estimate came out as 0.00 g), and a
29m41s time estimate for a job that actually takes 11m43s.

So this module flattens each inheritance chain itself and writes the merged
config to `<out_dir>/.presets/`, which doubles as an audit trail: the exact
settings that produced a given gcode sit next to it.

Three further CLI quirks, all found the hard way:
  * leaf filament presets have no "type" key -- the CLI rejects them with
    'unknown config type ... in load-settings'. It has to be added.
  * a merged config with no "from" key is rejected with "'s from unsupported".
    It must stay "system".
  * "compatible_printers" must survive the merge or the CLI refuses with
    'process not compatible with printer'.

Verified working command shape (see slice_stl.command on any SliceResult):
  Orca-Flashforge --slice 0 --arrange 1 --ensure-on-bed \\
    --load-settings "<flat machine>.json;<flat process>.json" \\
    --load-filaments "<flat filament>.json" \\
    --outputdir <dir> <model.stl>
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config

# Keys that describe a preset's place in the tree rather than its settings.
# They must not survive into the flattened file (except "from", which the CLI
# validates, and "compatible_printers", which it checks).
_DROP_KEYS = ("inherits", "instantiation")

_PRESET_TYPE = {"machine": "machine", "process": "process", "filament": "filament"}

# Orca logs these at [error] level when driven headless. They concern the GUI's
# OpenGL preview and the 3mf thumbnail, never the toolpaths, so they are noise
# for us. Anything else on the error stream is worth showing the user.
_BENIGN_ERRORS = (
    "exclude triangles",
    "fragment shader",
    "Unable to load shaders",
    "shader for rendering thumbnail",
)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class SliceResult:
    """Everything the demo needs to talk about a slice.

    Implements os.PathLike, so `Path(result)` / `open(result)` give the gcode
    and callers that only want the path can treat it as one.
    """

    gcode: Path
    print_time_s: float
    print_time_str: str
    filament_mm: float
    filament_cm3: float
    filament_g: float
    layers: int
    max_z_mm: float
    machine: str
    process: str
    filament: str
    command: list[str]
    preset_dir: Path
    threemf: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def __fspath__(self) -> str:
        return str(self.gcode)

    def summary(self) -> str:
        g = f"{self.filament_g:.2f} g" if self.filament_g > 0 else "n/a"
        return (
            f"{self.gcode.name}  {self.gcode.stat().st_size / 1024:.0f} KiB\n"
            f"  printer : {self.machine}\n"
            f"  process : {self.process}\n"
            f"  filament: {self.filament}\n"
            f"  time    : {self.print_time_str} ({self.print_time_s:.0f} s)\n"
            f"  material: {self.filament_mm:.1f} mm / {self.filament_cm3:.2f} cm3 / {g}\n"
            f"  layers  : {self.layers} up to Z={self.max_z_mm:.2f} mm"
        )


class SliceError(RuntimeError):
    """Orca refused to slice. Carries the real stdout/stderr."""


# --------------------------------------------------------------------------
# preset discovery + flattening
# --------------------------------------------------------------------------
def _preset_dir(kind: str) -> Path:
    if kind not in _PRESET_TYPE:
        raise ValueError(f"unknown preset kind {kind!r}; expected one of {sorted(_PRESET_TYPE)}")
    d = config.ORCA_SYSTEM / kind
    if not d.is_dir():
        raise FileNotFoundError(
            f"Orca system presets not found at {d}. Is Orca-Flashforge installed and "
            f"has it been launched once? (config.ORCA_SYSTEM = {config.ORCA_SYSTEM})"
        )
    return d


def index_presets(kind: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Map preset name -> (path, raw json) for one preset kind.

    Indexed by both the "name" field and the filename stem, because Orca's own
    `inherits` strings sometimes reference one and sometimes the other.
    """
    out: dict[str, tuple[Path, dict[str, Any]]] = {}
    for f in sorted(_preset_dir(kind).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        for key in (d.get("name"), f.stem):
            if key and key not in out:
                out[key] = (f, d)
    return out


def _resolve_chain(
    idx: dict[str, tuple[Path, dict[str, Any]]], name: str, _seen: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Root-first list of preset dicts from `name` up through its ancestors."""
    if name not in idx:
        near = [n for n in idx if name.lower() in n.lower()][:5]
        raise KeyError(
            f"preset {name!r} not found under {config.ORCA_SYSTEM}"
            + (f"; did you mean one of {near}?" if near else "")
        )
    d = idx[name][1]
    parent = d.get("inherits")
    if parent and parent not in _seen:
        return _resolve_chain(idx, parent, _seen + (name,)) + [d]
    return [d]


def flatten_preset(kind: str, name: str) -> dict[str, Any]:
    """Collapse a preset's whole `inherits` chain into one self-contained dict."""
    merged: dict[str, Any] = {}
    for d in _resolve_chain(index_presets(kind), name):
        merged.update(d)
    for k in _DROP_KEYS:
        merged.pop(k, None)
    merged["type"] = _PRESET_TYPE[kind]
    merged["name"] = name
    merged["from"] = "system"   # the CLI rejects anything else, including absent
    return merged


def materialize_preset(kind: str, name: str, out_dir: str | Path) -> Path:
    """Write a flattened preset to disk and return its path."""
    d = flatten_preset(kind, name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    p = out / f"{kind}__{safe}.json"
    p.write_text(json.dumps(d, indent=1) + "\n")
    return p


def _flat_get(kind: str, name: str, key: str, default: Any = None) -> Any:
    try:
        return flatten_preset(kind, name).get(key, default)
    except KeyError:
        return default


def machine_name_of(machine_preset: str | Path) -> str:
    """The preset name Orca matches `compatible_printers` against."""
    p = Path(machine_preset)
    if p.exists():
        d = json.loads(p.read_text())
        return d.get("name") or p.stem
    return str(machine_preset)


def _compatible(kind: str, name: str, machine: str) -> bool:
    cp = _flat_get(kind, name, "compatible_printers") or []
    return any(machine == c or machine in c for c in cp)


def find_presets(kind: str, machine: str, instantiable_only: bool = True) -> list[str]:
    """Names of `kind` presets this machine can actually use."""
    names = []
    for f in sorted(_preset_dir(kind).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        name = d.get("name") or f.stem
        if name.startswith("fdm_"):
            continue
        flat = None
        if instantiable_only:
            try:
                flat = flatten_preset(kind, name)
            except KeyError:
                continue
            if str(flat.get("instantiation", d.get("instantiation", "true"))).lower() != "true":
                # instantiation lives on the leaf, and the merge dropped it --
                # read it back off the raw leaf instead.
                if str(d.get("instantiation", "true")).lower() != "true":
                    continue
        if _compatible(kind, name, machine):
            names.append(name)
    return sorted(set(names))


def pick_process(machine: str, layer_mm: float | None = None) -> str:
    """Default process preset for this machine, or the one closest to layer_mm."""
    if layer_mm is None:
        mp = config.MACHINE_PRESET
        if Path(mp).exists():
            dflt = json.loads(Path(mp).read_text()).get("default_print_profile")
            if dflt:
                return str(dflt)
    cands = find_presets("process", machine)
    if not cands:
        raise SliceError(f"no Orca process presets are compatible with {machine!r}")
    if layer_mm is None:
        return cands[0]
    scored = []
    for n in cands:
        lh = _flat_get("process", n, "layer_height")
        try:
            scored.append((abs(float(lh) - layer_mm), n))
        except (TypeError, ValueError):
            continue
    if not scored:
        raise SliceError(f"none of {cands} declare a layer_height")
    return min(scored)[1]


def pick_filament(machine: str, material: str = "PLA") -> str:
    """A concrete filament preset of the requested material for this machine."""
    want = material.upper()
    cands = []
    for n in find_presets("filament", machine):
        ft = _flat_get("filament", n, "filament_type")
        if isinstance(ft, list):
            ft = ft[0] if ft else None
        if ft and str(ft).upper() == want:
            cands.append(n)
    if not cands:
        avail = sorted(
            {
                (lambda t: t[0] if isinstance(t, list) else t)(
                    _flat_get("filament", n, "filament_type")
                )
                for n in find_presets("filament", machine)
            }
            - {None}
        )
        raise SliceError(
            f"no {material!r} filament preset is compatible with {machine!r}. "
            f"Available types: {avail}"
        )
    # "Basic" is Flashforge's plain, fully-specified stock profile; prefer it,
    # then the shortest name, so the choice is deterministic run to run.
    cands.sort(key=lambda n: (0 if "basic" in n.lower() else 1, len(n), n))
    return cands[0]


# --------------------------------------------------------------------------
# gcode header parsing
# --------------------------------------------------------------------------
_TIME_UNITS = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def parse_time_string(s: str) -> float:
    """'1h 2m 3s' / '18m 49s' / '45s' -> seconds. Returns 0.0 if unparseable."""
    total = 0.0
    for value, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*([dhms])", s):
        total += float(value) * _TIME_UNITS[unit]
    return total


def parse_gcode_stats(path: str | Path) -> dict[str, Any]:
    """Pull the estimate block out of an Orca gcode file.

    Orca writes the summary near the top (right after HEADER_BLOCK_END) but
    some flavours append it instead, so both ends are scanned.
    """
    p = Path(path)
    text_head: list[str] = []
    with open(p, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 4000 or line.startswith("; CONFIG_BLOCK_START"):
                break
            text_head.append(line)
    lines = text_head
    if not any("estimated printing time" in ln for ln in lines):
        with open(p, "rb") as fh:
            fh.seek(max(0, p.stat().st_size - 200_000))
            lines = lines + fh.read().decode("utf-8", "replace").splitlines(keepends=True)

    stats: dict[str, Any] = {
        "print_time_s": 0.0,
        "print_time_str": "",
        "filament_mm": 0.0,
        "filament_cm3": 0.0,
        "filament_g": 0.0,
        "layers": 0,
        "max_z_mm": 0.0,
    }
    pats = [
        (r";\s*estimated printing time \(normal mode\)\s*=\s*(.+)", "print_time_str"),
        (r";\s*total filament used \[g\]\s*=\s*([0-9.]+)", "filament_g"),
        (r";\s*filament used \[g\]\s*=\s*([0-9.]+)", "filament_g"),
        (r";\s*filament used \[mm\]\s*=\s*([0-9.]+)", "filament_mm"),
        (r";\s*filament used \[cm3\]\s*=\s*([0-9.]+)", "filament_cm3"),
        (r";\s*total layers? count\s*=\s*([0-9]+)", "layers"),
        (r";\s*total layer number:\s*([0-9]+)", "layers"),
        (r";\s*max_z_height:\s*([0-9.]+)", "max_z_mm"),
    ]
    for ln in lines:
        for pat, key in pats:
            m = re.match(pat, ln)
            if not m:
                continue
            v = m.group(1).strip()
            if key == "print_time_str":
                stats["print_time_str"] = v
                stats["print_time_s"] = parse_time_string(v)
            elif key == "layers":
                stats["layers"] = max(stats["layers"], int(v))
            else:
                stats[key] = max(stats[key], float(v))
    return stats


# --------------------------------------------------------------------------
# the slice itself
# --------------------------------------------------------------------------
def _preflight_fits_bed(stl: Path) -> list[str]:
    """Cheap sanity check so an oversized part fails here, not 40 s into Orca."""
    warnings: list[str] = []
    try:
        import trimesh

        m = trimesh.load(stl, force="mesh")
        ext = sorted(float(v) for v in m.extents)
    except Exception as e:  # a load failure is Orca's problem to report properly
        return [f"could not pre-check model extents: {type(e).__name__}: {e}"]
    bed = sorted([config.BED_MM[0], config.BED_MM[1], config.BED_HEIGHT_MM])
    if any(e > b for e, b in zip(ext, bed)):
        raise SliceError(
            f"{stl.name} is {ext[2]:.1f} x {ext[1]:.1f} x {ext[0]:.1f} mm and will not fit the "
            f"{config.BED_MM[0]:.0f} x {config.BED_MM[1]:.0f} x {config.BED_HEIGHT_MM:.0f} mm "
            f"{config.PRINTER_MODEL} bed in any orientation. Split the model or scale it down "
            f"-- scan2print will not silently shrink a metric part."
        )
    if ext[2] > 0.95 * bed[2]:
        warnings.append(f"model nearly fills the bed ({ext[2]:.1f} mm vs {bed[2]:.0f} mm)")
    return warnings


def slice_stl(
    stl: str | Path,
    out_dir: str | Path,
    machine_preset: str | Path = config.MACHINE_PRESET,
    process: str | None = None,
    filament: str | None = None,
    extra_args: Sequence[str] | None = None,
    *,
    material: str = "PLA",
    layer_mm: float | None = None,
    arrange: bool = True,
    orient: bool = False,
    ensure_on_bed: bool = True,
    export_3mf: bool = False,
    rename_to_model: bool = True,
    datadir: str | Path | None = None,
    timeout_s: float = 900.0,
) -> SliceResult:
    """Slice `stl` and return the produced gcode plus its estimates.

    `process` and `filament` are Orca preset NAMES (not paths); leave them
    None to take the machine's declared default process and a stock preset of
    `material`. The result is os.PathLike -- Path(result) is the gcode.
    """
    stl = Path(stl).expanduser().resolve()
    if not stl.exists():
        raise FileNotFoundError(f"model not found: {stl}")
    if not config.ORCA_BIN.exists():
        raise SliceError(
            f"Orca-Flashforge CLI not found at {config.ORCA_BIN}. "
            f"config.check_tools() reports: {config.check_tools()}"
        )

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = _preflight_fits_bed(stl)

    machine = machine_name_of(machine_preset)
    process = process or pick_process(machine, layer_mm)
    filament = filament or pick_filament(machine, material)

    presets = out_dir / ".presets"
    m_json = materialize_preset("machine", machine, presets)
    p_json = materialize_preset("process", process, presets)
    f_json = materialize_preset("filament", filament, presets)

    cmd: list[str] = [str(config.ORCA_BIN), "--slice", "0"]
    if datadir:
        cmd += ["--datadir", str(datadir)]
    if arrange:
        cmd += ["--arrange", "1"]
    if orient:
        cmd += ["--orient", "1"]
    if ensure_on_bed:
        cmd += ["--ensure-on-bed"]
    cmd += [
        "--load-settings", f"{m_json};{p_json}",
        "--load-filaments", str(f_json),
        "--outputdir", str(out_dir),
    ]
    if export_3mf:
        # Orca prepends --outputdir to this argument, so an absolute path here
        # produces a doubled path and "Unable to open the file". Bare name only.
        cmd += ["--export-3mf", f"{stl.stem}.3mf"]
    cmd += list(extra_args or [])
    cmd.append(str(stl))

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise SliceError(f"Orca timed out after {timeout_s:.0f}s\ncommand: {_fmt(cmd)}") from e
    blob = (proc.stdout or "") + (proc.stderr or "")

    produced = _find_new_gcode(out_dir, t0)
    if proc.returncode != 0 or "run found error" in blob or not produced:
        raise SliceError(
            f"Orca-Flashforge failed to slice {stl.name} (exit {proc.returncode}).\n"
            f"command: {_fmt(cmd)}\n"
            f"--- orca stdout ---\n{proc.stdout}\n"
            f"--- orca stderr ---\n{proc.stderr}"
        )
    gcode = produced[0]

    # Orca names its output plate_N.gcode. Give the single-plate case the
    # model's name so the file that reaches the printer is identifiable.
    if rename_to_model and len(produced) == 1 and gcode.suffix == ".gcode":
        target = gcode.with_name(f"{stl.stem}.gcode")
        if target != gcode:
            gcode = Path(os.replace(gcode, target) or target)

    # Orca logs recoverable complaints on the same stream; surface the real
    # ones without failing, since the gcode itself is valid.
    for ln in blob.splitlines():
        if "[error]" in ln and not any(n in ln for n in _BENIGN_ERRORS):
            warnings.append(ln.strip())

    stats = parse_gcode_stats(gcode)
    if stats["print_time_s"] <= 0:
        warnings.append("could not parse a print-time estimate out of the gcode header")

    threemf = out_dir / f"{stl.stem}.3mf"
    return SliceResult(
        gcode=gcode,
        print_time_s=stats["print_time_s"],
        print_time_str=stats["print_time_str"],
        filament_mm=stats["filament_mm"],
        filament_cm3=stats["filament_cm3"],
        filament_g=stats["filament_g"],
        layers=stats["layers"],
        max_z_mm=stats["max_z_mm"],
        machine=machine,
        process=process,
        filament=filament,
        command=cmd,
        preset_dir=presets,
        threemf=threemf if (export_3mf and threemf.exists()) else None,
        warnings=warnings,
    )


def _find_new_gcode(out_dir: Path, since: float) -> list[Path]:
    """Orca names its output itself (plate_1.gcode here); never assume a name."""
    hits: list[Path] = []
    for pat in ("*.gcode", "*.gcode.3mf", "*.gco", "*.g"):
        for f in glob.glob(str(out_dir / "**" / pat), recursive=True):
            p = Path(f)
            if p.is_file() and p.stat().st_mtime >= since - 2.0:
                hits.append(p)
    hits.sort(key=lambda p: (-p.stat().st_mtime, str(p)))
    return hits


def _fmt(cmd: Iterable[str]) -> str:
    import shlex

    return " ".join(shlex.quote(c) for c in cmd)


def command_line(result: SliceResult) -> str:
    """The exact shell command that produced this gcode, copy-pasteable."""
    return _fmt(result.command)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="scan2print.slicer")
    ap.add_argument("stl", nargs="?", help="model to slice")
    ap.add_argument("-o", "--out", default=str(config.OUT_DIR / "gcode"))
    ap.add_argument("--material", default="PLA")
    ap.add_argument("--layer", type=float, default=None, help="preferred layer height mm")
    ap.add_argument("--process", default=None)
    ap.add_argument("--filament", default=None)
    ap.add_argument("--list", action="store_true", help="list compatible presets and exit")
    a = ap.parse_args(argv)

    machine = machine_name_of(config.MACHINE_PRESET)
    if a.list:
        print(f"machine : {machine}")
        print("process :")
        for n in find_presets("process", machine):
            print(f"    {n}   (layer {_flat_get('process', n, 'layer_height')})")
        print("filament:")
        for n in find_presets("filament", machine):
            ft = _flat_get("filament", n, "filament_type")
            print(f"    {n}   ({ft[0] if isinstance(ft, list) else ft})")
        return 0
    if not a.stl:
        ap.error("stl is required unless --list is given")

    r = slice_stl(
        a.stl, a.out, process=a.process, filament=a.filament,
        material=a.material, layer_mm=a.layer,
    )
    print(command_line(r))
    print()
    print(r.summary())
    for w in r.warnings:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
