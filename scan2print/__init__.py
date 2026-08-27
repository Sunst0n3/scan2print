"""scan2print -- phone video of an object on a ChArUco board -> true-millimetre
printable mesh -> sliced gcode for a Flashforge Adventurer 5M Pro.

Submodules are imported lazily: `import scan2print` stays cheap, and pulling in
`scan2print.mesh` (trimesh) or `scan2print.charuco` (cv2) only happens when
something actually asks for them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

_SUBMODULES = (
    "config", "types", "charuco", "frames", "scale",
    "mesh", "compensate", "slicer", "synth", "verify", "cli",
)

__all__ = list(_SUBMODULES) + ["__version__"]


def __getattr__(name: str):
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # keeps editors/type-checkers honest without the runtime cost
    from . import (  # noqa: F401
        charuco, cli, compensate, config, frames, mesh, scale, slicer, synth,
        types, verify,
    )
