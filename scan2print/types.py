"""Shared data contracts for the scan2print pipeline.

Coordinate frames used throughout:
  BOARD  -- right-handed, millimetres, origin at the ChArUco board's top-left
            inner corner, +X along the board's long axis, +Y along the short
            axis, +Z out of the board towards the camera. This is the metric
            ground truth frame and is THE authoritative convention.

            NOTE: OpenCV's own ChArUco object-point frame is NOT this frame --
            it puts +Z INTO the page, so raw solvePnP camera centres come back
            with negative Z. charuco.cv_points_to_board() performs the flip.
            Anything producing BOARD-frame coordinates must go through that
            helper rather than using OpenCV output directly; a sign error here
            mirrors the model, which still fits well and prints backwards.
  RECON  -- Object Capture's arbitrary-scale reconstruction frame. Related to
            BOARD by a similarity transform (scale + rotation + translation),
            which is exactly what scale.solve_scale() recovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json
import numpy as np

# Max fractional difference between x and y rescale factors before we refuse.
ASPECT_TOL = 0.01


@dataclass
class BoardSpec:
    """Physical description of the printed ChArUco target.

    square_mm is the single most important number in this program: every
    dimension the pipeline reports is ultimately derived from it. It must
    match the *printed* square size measured with calipers, not the nominal
    size requested at generation time -- printers scale pages.
    """

    squares_x: int = 7
    squares_y: int = 5
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dict_name: str = "DICT_5X5_1000"

    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_mm

    @property
    def n_corners(self) -> int:
        """Number of interior chessboard corners the detector can return."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "BoardSpec":
        return BoardSpec(**{k: v for k, v in d.items() if k in BoardSpec.__dataclass_fields__})


@dataclass
class Intrinsics:
    """Pinhole intrinsics + Brown-Conrady distortion for one camera at one
    resolution. Valid only for the resolution it was calibrated at; scale fx,
    fy, cx, cy proportionally if frames are resized."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=lambda: [0.0] * 5)
    width: int = 0
    height: int = 0
    rms_px: float = 0.0
    source: str = "unknown"      # "charuco-calibration" | "objectcapture" | "unknown"

    @property
    def is_calibrated(self) -> bool:
        """True only for intrinsics we actually solved for ourselves.

        Object Capture's self-reported intrinsics are usable but measured
        ~0.33% off on ground-truth renders, which propagates 1:1 into every
        reported dimension. Callers should cap reported quality when this is
        False rather than implying a measurement was made.
        """
        return self.source == "charuco-calibration" and self.rms_px > 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def D(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64).reshape(-1, 1)

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Return intrinsics rescaled for a different frame resolution."""
        if self.width == 0 or self.height == 0:
            raise ValueError("cannot rescale intrinsics with unknown source resolution")
        sx, sy = width / self.width, height / self.height
        # A single optical path cannot legitimately rescale by different
        # factors in x and y. When this fires it almost always means stills
        # calibration (4:3) is being applied to video (16:9) -- which silently
        # destroys undistortion and drops every board pose, with an error
        # message that otherwise blames the user's shooting technique.
        if abs(sx / sy - 1.0) > ASPECT_TOL:
            raise ValueError(
                f"refusing to rescale intrinsics non-uniformly: calibrated at "
                f"{self.width}x{self.height} (aspect {self.width / self.height:.4f}) "
                f"but frames are {width}x{height} (aspect {width / height:.4f}). "
                f"Recalibrate at the aspect ratio you actually shoot at -- phone "
                f"stills and phone video usually differ."
            )
        # Provenance must survive the rescale. Dropping it silently demotes a
        # real calibration to "unknown", which caps run quality at marginal and
        # tells the user to run a calibration they already ran -- rescaling to
        # the extraction resolution is the normal path, not an exception.
        return Intrinsics(
            fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
            dist=list(self.dist), width=width, height=height, rms_px=self.rms_px,
            source=self.source,
        )

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path) -> "Intrinsics":
        with open(path) as f:
            d = json.load(f)
        known = Intrinsics.__dataclass_fields__
        return Intrinsics(**{k: v for k, v in d.items() if k in known})


@dataclass
class FramePose:
    """Metric camera pose for one frame, recovered from the ChArUco board.

    R and t map BOARD points into camera coordinates: x_cam = R @ x_board + t.
    `center` is the camera position expressed in BOARD millimetres, i.e.
    -R.T @ t, which is what the scale solve actually consumes.
    """

    name: str
    R: list[list[float]]
    t: list[float]
    center: list[float]
    n_corners: int
    reproj_rms_px: float

    @property
    def R_np(self) -> np.ndarray:
        return np.array(self.R, dtype=np.float64).reshape(3, 3)

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float64).reshape(3)


@dataclass
class ReconPose:
    """Camera pose as reported by Object Capture, in the arbitrary-scale
    RECON frame. `center` is the camera position in RECON units."""

    name: str
    transform: list[float]        # 16 floats, column-major camera-to-world
    center: list[float]

    @property
    def center_np(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float64).reshape(3)


@dataclass
class ScaleResult:
    """Output of the similarity fit between RECON and BOARD camera tracks.

    Multiplying RECON geometry by `scale` converts it to millimetres.
    `rms_mm` is the residual of the fit and is the pipeline's primary quality
    signal -- a low residual means the two independently-derived camera tracks
    agree, which is strong evidence the scale is right.
    """

    scale: float
    rms_mm: float
    n_cameras: int
    R: list[list[float]]
    t: list[float]
    inlier_frac: float
    quality: str                  # "good" | "marginal" | "reject"
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ScaleResult":
        return ScaleResult(**d)
