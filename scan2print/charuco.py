"""ChArUco target generation, detection, and metric camera pose.

This module is the origin of every real-world dimension the pipeline reports,
so it is deliberately paranoid: it refuses to return a pose it cannot justify,
and it burns physical-verification aids into the printed target.

Coordinate frames
-----------------
OpenCV's raw ``CharucoBoard`` frame is NOT the scan2print BOARD frame, and the
difference is not cosmetic. Two facts, both measured on OpenCV 5.0.0 rather
than remembered:

1. The origin sits at the board's *outer* top-left corner, so the first
   interior chessboard corner is at ``(square_mm, square_mm, 0)``.
2. In the image produced by ``generateImage``, +X runs right and +Y runs
   *down* the page. Since the frame is right-handed, +Z therefore points
   *into* the paper, away from whoever is looking at it. Feeding a real
   fronto-parallel photo of the board through ``solvePnP`` puts the camera
   centre at ``z = -250`` mm, i.e. behind the page, which confirms it.

``types.py`` specifies BOARD as right-handed, origin at the top-left *interior*
corner, +X along the long axis, +Z out of the board *towards the camera*. So
every object point here is mapped

    p_board = diag(1, -1, -1) @ (p_opencv - (square_mm, square_mm, 0))

which is a translation plus a 180-degree rotation about X: distances, scale and
handedness are all preserved, and cameras viewing the printed face now get a
positive BOARD z as the contract requires. +Y consequently runs *up* the page,
so the board body occupies y <= 0. Getting this wrong is silent and vicious --
it mirrors the reconstruction rather than erroring.

OpenCV 5 API notes
------------------
The legacy free functions (``interpolateCornersCharuco``,
``estimatePoseCharucoBoard``, ``calibrateCameraCharuco``) no longer exist. The
object-oriented path used here is ``CharucoDetector.detectBoard`` ->
``CharucoBoard.matchImagePoints`` -> ``cv2.solvePnP`` / ``cv2.calibrateCamera``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scan2print import config
from scan2print.types import BoardSpec, FramePose, Intrinsics

__all__ = [
    "make_board",
    "board_object_points",
    "board_center_mm",
    "cv_points_to_board",
    "render_target",
    "detect",
    "pose_for_frame",
    "calibrate_camera",
    "poses_for_dir",
]

MM_PER_INCH = 25.4

# Minimum number of usable board views before a full intrinsics calibration is
# trustworthy. Fewer than this and focal length / principal point are badly
# under-constrained, which would silently poison every downstream measurement.
MIN_CALIB_VIEWS = 6

# A whole-set calibration RMS worse than this means the input images are not
# actually of this board (wrong square size, wrong dictionary, motion blur).
# Better to stop than to hand back a plausible-looking but wrong K.
CALIB_HARD_FAIL_RMS_PX = 3.0

# Printed verification aids.
RULER_MM = 100.0          # horizontal caliper bar
VRULER_MM = 50.0          # vertical bar, catches anisotropic printer scaling
RULER_TICK_MM = 10.0

# Paper sizes (w_mm, h_mm) and the per-edge margin most consumer printers
# cannot print inside.
_PAPER = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
}
_PRINTER_UNPRINTABLE_MM = 5.0

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heic")

# CharucoBoard / CharucoDetector construction is not free; cache per spec.
_board_cache: dict[tuple, cv2.aruco.CharucoBoard] = {}
_detector_cache: dict[tuple, cv2.aruco.CharucoDetector] = {}


def _spec_key(spec: BoardSpec) -> tuple:
    return (
        int(spec.squares_x),
        int(spec.squares_y),
        float(spec.square_mm),
        float(spec.marker_mm),
        str(spec.dict_name),
    )


def _validate(spec: BoardSpec) -> None:
    if spec.squares_x < 2 or spec.squares_y < 2:
        raise ValueError(f"board must be at least 2x2 squares, got {spec.squares_x}x{spec.squares_y}")
    if spec.square_mm <= 0 or spec.marker_mm <= 0:
        raise ValueError(f"square_mm and marker_mm must be positive, got {spec.square_mm}/{spec.marker_mm}")
    if spec.marker_mm >= spec.square_mm:
        raise ValueError(
            f"marker_mm ({spec.marker_mm}) must be smaller than square_mm ({spec.square_mm}); "
            "aruco markers have to fit inside a chessboard square with a white quiet zone"
        )
    if not hasattr(cv2.aruco, spec.dict_name):
        names = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        raise ValueError(f"unknown aruco dictionary {spec.dict_name!r}; available: {names}")
    n_markers = (spec.squares_x * spec.squares_y) // 2
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dict_name))
    available = int(dictionary.bytesList.shape[0])
    if n_markers > available:
        raise ValueError(
            f"{spec.dict_name} holds {available} markers but a {spec.squares_x}x{spec.squares_y} "
            f"board needs {n_markers}"
        )


def make_board(spec: BoardSpec) -> cv2.aruco.CharucoBoard:
    """Build (and cache) the OpenCV board object for a physical target.

    Board units are millimetres, so every object point coming out of it is
    already metric.
    """
    key = _spec_key(spec)
    board = _board_cache.get(key)
    if board is None:
        _validate(spec)
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dict_name))
        board = cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y), float(spec.square_mm), float(spec.marker_mm), dictionary
        )
        _board_cache[key] = board
    return board


# diag(1, -1, -1): 180 degrees about X. Proper rotation (det = +1), so it
# preserves handedness, distances and scale.
_FLIP = np.array([1.0, -1.0, -1.0])


def cv_points_to_board(pts_cv: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """Map OpenCV board object points into the scan2print BOARD frame.

    See the module docstring: translate the origin to the top-left interior
    corner, then rotate 180 degrees about X so +Z points out of the printed
    face towards the camera.
    """
    pts = np.asarray(pts_cv, dtype=np.float64).reshape(-1, 3)
    pts = pts - np.array([spec.square_mm, spec.square_mm, 0.0])
    return pts * _FLIP + 0.0          # + 0.0 normalises IEEE negative zeros


def board_object_points(spec: BoardSpec) -> np.ndarray:
    """All interior chessboard corners in the scan2print BOARD frame, (N, 3) mm.

    Corner i corresponds to charuco id i; corner 0 is exactly the origin and
    the board extends towards +X and -Y.
    """
    return cv_points_to_board(make_board(spec).getChessboardCorners(), spec)


def board_center_mm(spec: BoardSpec) -> np.ndarray:
    """Centre of the interior corner grid in BOARD mm -- a convenient aim point."""
    return board_object_points(spec).mean(axis=0)


def _detector(spec: BoardSpec) -> cv2.aruco.CharucoDetector:
    key = _spec_key(spec)
    det = _detector_cache.get(key)
    if det is None:
        board = make_board(spec)
        det_params = cv2.aruco.DetectorParameters()
        # Sub-pixel marker refinement; the charuco corners themselves are
        # refined separately by the charuco detector, but better marker
        # corners means better interpolation seeds.
        det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det_params.cornerRefinementWinSize = 5
        det_params.cornerRefinementMaxIterations = 50
        det_params.cornerRefinementMinAccuracy = 0.01
        ch_params = cv2.aruco.CharucoParameters()
        ch_params.tryRefineMarkers = True
        ch_params.minMarkers = 2
        det = cv2.aruco.CharucoDetector(board, ch_params, det_params)
        _detector_cache[key] = det
    return det


# --------------------------------------------------------------------------
# target rendering
# --------------------------------------------------------------------------

def _font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size_px)
            except OSError:
                continue
    return ImageFont.load_default(size=size_px)


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    length_px: int,
    thickness_px: int,
    seg_px: float,
    n_seg: int,
    font: ImageFont.FreeTypeFont,
    tick_mm: float,
    vertical: bool = False,
) -> None:
    """Alternating black/white segments with an exact inked extent.

    The inked extent is exactly ``length_px`` pixels (x0 .. x0+length_px-1
    inclusive), so a caliper laid across the outside edges reads the intended
    physical length.
    """
    if vertical:
        draw.rectangle([x0, y0, x0 + thickness_px - 1, y0 + length_px - 1], outline=0, width=2)
        for i in range(n_seg):
            if i % 2 == 0:
                continue
            a = y0 + int(round(i * seg_px))
            b = y0 + int(round((i + 1) * seg_px)) - 1
            draw.rectangle([x0, a, x0 + thickness_px - 1, min(b, y0 + length_px - 1)], fill=0)
        for i in range(n_seg + 1):
            y = y0 + min(int(round(i * seg_px)), length_px - 1)
            draw.line([x0 + thickness_px + 2, y, x0 + thickness_px + 10, y], fill=0, width=2)
            draw.text((x0 + thickness_px + 14, y), f"{i * tick_mm:g}", font=font, fill=0, anchor="lm")
    else:
        draw.rectangle([x0, y0, x0 + length_px - 1, y0 + thickness_px - 1], outline=0, width=2)
        for i in range(n_seg):
            if i % 2 == 0:
                continue
            a = x0 + int(round(i * seg_px))
            b = x0 + int(round((i + 1) * seg_px)) - 1
            draw.rectangle([a, y0, min(b, x0 + length_px - 1), y0 + thickness_px - 1], fill=0)
        for i in range(n_seg + 1):
            x = x0 + min(int(round(i * seg_px)), length_px - 1)
            draw.line([x, y0 + thickness_px + 2, x, y0 + thickness_px + 10], fill=0, width=2)
            draw.text((x, y0 + thickness_px + 14), f"{i * tick_mm:g}", font=font, fill=0, anchor="mt")


def render_target(
    spec: BoardSpec,
    out_png: Path,
    dpi: int = 300,
    *,
    paper: str = "auto",
    margin_mm: float = 10.0,
    write_pdf: bool = True,
) -> Path:
    """Render a print-ready ChArUco target at true physical size.

    The page carries a caption stating the nominal square size, a 100 mm
    horizontal caliper bar and a 50 mm vertical bar. Consumer printers apply
    "fit to page" by default; if either bar does not measure its stated length
    the print is scaled and every downstream millimetre would be wrong.

    ``paper`` is "auto" (page grows to fit the content) or one of
    ``letter`` / ``a4`` / ``a3``, in which case the content is centred and an
    error is raised if it does not fit inside the printable area.

    Writes ``out_png``, a same-basename ``.pdf`` at true physical size, and a
    ``.json`` sidecar recording the exact rendered geometry. Returns the PNG
    path.
    """
    _validate(spec)
    out_png = Path(out_png)
    if dpi < 72:
        raise ValueError(f"dpi={dpi} is too coarse for a metrology target")
    ppmm = dpi / MM_PER_INCH

    # Integer pixels per square keeps every square identical; the residual
    # quantisation is reported honestly rather than hidden.
    px_per_square = int(round(spec.square_mm * ppmm))
    if px_per_square < 20:
        raise ValueError(
            f"square_mm={spec.square_mm} at {dpi} dpi is only {px_per_square} px per square; "
            "raise dpi or square_mm"
        )
    eff_square_mm = px_per_square / ppmm
    board_w_px = spec.squares_x * px_per_square
    board_h_px = spec.squares_y * px_per_square

    board_img = make_board(spec).generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)
    if board_img.shape[:2] != (board_h_px, board_w_px):
        raise RuntimeError(
            f"generateImage returned {board_img.shape[:2]}, expected {(board_h_px, board_w_px)}"
        )

    # --- typography / element sizes, all derived from dpi so the page looks
    # identical at any resolution.
    def px(mm: float) -> int:
        return int(round(mm * ppmm))

    f_title = _font(px(5.0), bold=True)
    f_body = _font(px(3.4))
    f_small = _font(px(2.6))
    f_tick = _font(px(2.4))

    ruler_px = int(round(RULER_MM * ppmm))
    vruler_px = int(round(VRULER_MM * ppmm))
    ruler_thick = px(6.0)
    vruler_thick = px(6.0)
    tick_room = px(6.0)
    gap = px(6.0)
    margin_px = px(margin_mm)

    title = "scan2print ChArUco target  --  PRINT AT 100%, DO NOT SCALE"
    head_lines = [
        f"nominal square {spec.square_mm:.2f} mm   marker {spec.marker_mm:.2f} mm   "
        f"{spec.squares_x} x {spec.squares_y}   {spec.dict_name}",
        f"rendered square {eff_square_mm:.4f} mm at {dpi} dpi ({px_per_square} px)   "
        f"board {spec.squares_x * eff_square_mm:.2f} x {spec.squares_y * eff_square_mm:.2f} mm",
    ]
    foot_lines = [
        "CHECK BEFORE SCANNING: measure the bars above with calipers. The horizontal bar is "
        f"{RULER_MM:.1f} mm and the vertical bar is {VRULER_MM:.1f} mm, outer edge to outer edge.",
        "If they do not match, your printer rescaled the page. Reprint with scaling set to 100% / "
        "Actual Size, or measure one chessboard square directly and",
        "set BoardSpec.square_mm to that measured value -- the pipeline trusts square_mm absolutely "
        "and every reported dimension scales linearly with it.",
        f"(rendered lengths, before printer error: bar {ruler_px / ppmm:.3f} mm, "
        f"vbar {vruler_px / ppmm:.3f} mm, square {eff_square_mm:.4f} mm)",
    ]

    probe = Image.new("L", (8, 8), 255)
    pd = ImageDraw.Draw(probe)

    def text_w(s: str, font: ImageFont.FreeTypeFont) -> int:
        return int(round(pd.textlength(s, font=font)))

    def wrap(s: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
        """Greedy word wrap. Text must never widen the page -- the board and
        the caliper bars decide the physical size, captions fit around them."""
        out: list[str] = []
        line = ""
        for word in s.split():
            trial = f"{line} {word}".strip()
            if line and text_w(trial, font) > max_w:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
        return out

    vbar_label = f"{VRULER_MM:.1f} mm"
    vbar_block_px = vruler_thick + px(4.0) + max(
        text_w("100", f_tick), text_w(vbar_label, f_small)
    ) + px(3.0)
    ruler_row_px = ruler_px + gap * 2 + vbar_block_px

    content_w = max(board_w_px, ruler_row_px)
    title_lines = wrap(title, f_title, content_w)
    head_lines = [ln for s in head_lines for ln in wrap(s, f_body, content_w)]
    foot_lines = [ln for s in foot_lines for ln in wrap(s, f_small, content_w)]

    line_h_title = px(6.5)
    line_h_body = px(4.6)
    line_h_small = px(3.6)
    head_h = len(title_lines) * line_h_title + len(head_lines) * line_h_body
    ruler_label_h = line_h_small
    ruler_row_h = ruler_label_h + max(ruler_thick + tick_room, vruler_px + px(2.0) + line_h_small)
    foot_h = len(foot_lines) * line_h_small
    content_h = head_h + gap + board_h_px + gap + ruler_row_h + gap + foot_h

    if paper == "auto":
        page_w = content_w + 2 * margin_px
        page_h = content_h + 2 * margin_px
        off_x = margin_px
        off_y = margin_px
        # A page bigger than the paper in the tray is the single most likely
        # way this target gets silently scaled, so say so loudly.
        pw_mm, ph_mm = page_w / ppmm, page_h / ppmm

        def _fits(name: str) -> bool:
            w_mm, h_mm = _PAPER[name]
            return (pw_mm <= w_mm - 2 * _PRINTER_UNPRINTABLE_MM
                    and ph_mm <= h_mm - 2 * _PRINTER_UNPRINTABLE_MM)

        # Letter and A4 are the only sizes a home printer is assumed to hold.
        if not (_fits("letter") or _fits("a4")):
            usable = min(_PAPER["letter"][0], _PAPER["a4"][0]) - 2 * _PRINTER_UNPRINTABLE_MM - 2 * margin_mm
            max_square = usable / spec.squares_x
            extra = " It does fit A3." if _fits("a3") else ""
            print(
                f"[charuco] WARNING: this page is {pw_mm:.1f} x {ph_mm:.1f} mm, which does not fit "
                f"Letter or A4 at 100% scale.{extra} On Letter/A4 the print dialog will fall back to "
                f"'fit to page', silently rescaling the board and corrupting every downstream "
                f"measurement. Either print on larger stock, or rebuild the target with "
                f"square_mm <= {max_square:.1f} (currently {spec.square_mm:g}) for a "
                f"{spec.squares_x}x{spec.squares_y} board.",
                file=sys.stderr,
            )
    else:
        key = paper.lower()
        if key not in _PAPER:
            raise ValueError(f"unknown paper {paper!r}; use 'auto' or one of {sorted(_PAPER)}")
        pw_mm, ph_mm = _PAPER[key]
        page_w, page_h = px(pw_mm), px(ph_mm)
        printable_w = px(pw_mm - 2 * _PRINTER_UNPRINTABLE_MM)
        printable_h = px(ph_mm - 2 * _PRINTER_UNPRINTABLE_MM)
        if content_w > printable_w or content_h > printable_h:
            raise ValueError(
                f"target content is {content_w / ppmm:.1f} x {content_h / ppmm:.1f} mm but the "
                f"printable area of {key} is {printable_w / ppmm:.1f} x {printable_h / ppmm:.1f} mm. "
                f"Reduce square_mm (currently {spec.square_mm} mm) or the board size, or use "
                f"paper='a3'/'auto'."
            )
        off_x = (page_w - content_w) // 2
        off_y = (page_h - content_h) // 2

    page = Image.new("L", (page_w, page_h), 255)
    draw = ImageDraw.Draw(page)

    y = off_y
    for line in title_lines:
        draw.text((off_x, y), line, font=f_title, fill=0)
        y += line_h_title
    for line in head_lines:
        draw.text((off_x, y), line, font=f_body, fill=0)
        y += line_h_body
    y += gap

    board_x = off_x + (content_w - board_w_px) // 2
    page.paste(Image.fromarray(board_img), (board_x, y))
    # Hairline frame just outside the board so the printed extent is visible.
    draw.rectangle(
        [board_x - 2, y - 2, board_x + board_w_px + 1, y + board_h_px + 1], outline=0, width=1
    )
    y += board_h_px + gap

    # Labels sit above the bars: printed over the alternating segments they
    # become unreadable exactly where the user needs to read them.
    draw.text((off_x, y), f"{RULER_MM:.1f} mm  <-- measure this", font=f_small, fill=0)
    _draw_scale_bar(
        draw, off_x, y + ruler_label_h, ruler_px, ruler_thick,
        RULER_TICK_MM * ppmm, int(round(RULER_MM / RULER_TICK_MM)), f_tick, RULER_TICK_MM,
    )
    vx = off_x + ruler_px + gap * 2
    draw.text((vx, y), vbar_label, font=f_small, fill=0)
    _draw_scale_bar(
        draw, vx, y + ruler_label_h, vruler_px, vruler_thick,
        RULER_TICK_MM * ppmm, int(round(VRULER_MM / RULER_TICK_MM)), f_tick, RULER_TICK_MM,
        vertical=True,
    )
    y += ruler_row_h + gap

    for line in foot_lines:
        draw.text((off_x, y), line, font=f_small, fill=0)
        y += line_h_small

    out_png.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_png, dpi=(dpi, dpi))

    if write_pdf:
        # Pillow writes the PDF MediaBox as pixels * 72/dpi, i.e. true
        # physical size, as long as `resolution` matches the raster dpi.
        page.convert("L").save(out_png.with_suffix(".pdf"), "PDF", resolution=float(dpi))

    sidecar = {
        "spec": spec.to_json(),
        "dpi": dpi,
        "px_per_square": px_per_square,
        "effective_square_mm": eff_square_mm,
        "effective_board_mm": [spec.squares_x * eff_square_mm, spec.squares_y * eff_square_mm],
        "page_px": [page_w, page_h],
        "page_mm": [page_w / ppmm, page_h / ppmm],
        "ruler_mm_rendered": ruler_px / ppmm,
        "vruler_mm_rendered": vruler_px / ppmm,
        "paper": paper,
    }
    out_png.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return out_png


# --------------------------------------------------------------------------
# detection + pose
# --------------------------------------------------------------------------

def _as_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale or 3-channel image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {arr.dtype}")
    return np.ascontiguousarray(arr)


def detect(gray: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray, int]:
    """Detect interior ChArUco corners.

    Returns ``(charuco_corners, charuco_ids, n)`` where corners is
    ``(n, 1, 2)`` float32 pixel coordinates and ids is ``(n, 1)`` int32.
    Empty arrays and ``n == 0`` when the board is not visible.
    """
    gray = _as_gray(gray)
    ch_corners, ch_ids, _mk_corners, _mk_ids = _detector(spec).detectBoard(gray)
    if ch_corners is None or ch_ids is None or len(ch_ids) == 0:
        return (
            np.empty((0, 1, 2), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32),
            0,
        )
    ch_corners = np.asarray(ch_corners, dtype=np.float32).reshape(-1, 1, 2)
    ch_ids = np.asarray(ch_ids, dtype=np.int32).reshape(-1, 1)
    return ch_corners, ch_ids, int(len(ch_ids))


def _solve_pnp(
    objp: np.ndarray, imgp: np.ndarray, K: np.ndarray, D: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """PnP for a planar target.

    ITERATIVE alone can settle into the mirror solution on planar point sets,
    so IPPE (which is built for planar targets and returns the two candidate
    poses ranked) seeds the iterative refinement. Falls back to plain
    ITERATIVE if IPPE is unhappy.
    """
    objp = np.ascontiguousarray(objp, dtype=np.float64).reshape(-1, 1, 3)
    imgp = np.ascontiguousarray(imgp, dtype=np.float64).reshape(-1, 1, 2)
    rvec = tvec = None
    if len(objp) >= 4:
        try:
            ok, rv, tv = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                rvec, tvec = rv, tv
        except cv2.error:
            rvec = tvec = None
    if rvec is None:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("cv2.solvePnP failed to converge")
    else:
        ok, rvec, tvec = cv2.solvePnP(
            objp, imgp, K, D, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError("cv2.solvePnP refinement failed to converge")
    return np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1)


def _reproj_rms_px(
    objp: np.ndarray, imgp: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
    K: np.ndarray, D: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(
        np.ascontiguousarray(objp, dtype=np.float64).reshape(-1, 1, 3), rvec, tvec, K, D
    )
    err = proj.reshape(-1, 2) - np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def pose_for_frame(
    gray: np.ndarray, spec: BoardSpec, intr: Intrinsics, name: str = ""
) -> FramePose | None:
    """Metric camera pose for one frame, or None if it cannot be trusted.

    Rejects the frame when fewer than ``config.MIN_CHARUCO_CORNERS`` corners
    are found, when the detected corners are collinear (a degenerate,
    unsolvable configuration), or when the reprojection RMS exceeds
    ``config.MAX_REPROJ_RMS_PX``.
    """
    corners, ids, n = detect(gray, spec)
    if n < config.MIN_CHARUCO_CORNERS:
        return None

    board = make_board(spec)
    if board.checkCharucoCornersCollinear(ids):
        return None

    objp, imgp = board.matchImagePoints(corners, ids)
    if objp is None or imgp is None or len(objp) < config.MIN_CHARUCO_CORNERS:
        return None
    objp = cv_points_to_board(objp, spec)
    imgp = np.asarray(imgp, dtype=np.float64).reshape(-1, 2)

    K, D = intr.K, intr.D
    try:
        rvec, tvec = _solve_pnp(objp, imgp, K, D)
    except (cv2.error, RuntimeError):
        return None

    rms = _reproj_rms_px(objp, imgp, rvec, tvec, K, D)
    if not np.isfinite(rms) or rms > config.MAX_REPROJ_RMS_PX:
        return None

    R, _ = cv2.Rodrigues(rvec)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = tvec.reshape(3)
    center = -R.T @ t
    return FramePose(
        name=name,
        R=R.tolist(),
        t=t.tolist(),
        center=center.tolist(),
        n_corners=int(len(objp)),
        reproj_rms_px=rms,
    )


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibrate_camera(
    image_paths: list[Path], spec: BoardSpec, *, flags: int = 0
) -> Intrinsics:
    """Full intrinsics calibration from ChArUco board views.

    ``flags`` is passed straight to ``cv2.calibrateCamera``;
    ``cv2.CALIB_FIX_K3`` is often worth setting for phone cameras, whose mild
    distortion lets k3 soak up noise.

    Raises RuntimeError with a specific message when the input cannot support
    a trustworthy calibration -- a wrong K would corrupt every pose and hence
    every millimetre the pipeline reports.
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise RuntimeError("calibrate_camera got an empty image list")

    board = make_board(spec)
    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    used: list[str] = []
    rejected: list[str] = []
    size_wh: tuple[int, int] | None = None

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rejected.append(f"{p.name}: unreadable")
            continue
        h, w = img.shape[:2]
        if size_wh is None:
            size_wh = (w, h)
        elif (w, h) != size_wh:
            raise RuntimeError(
                f"{p.name} is {w}x{h} but earlier images are {size_wh[0]}x{size_wh[1]}; "
                "intrinsics are only valid for a single resolution -- calibrate each separately"
            )
        corners, ids, n = detect(img, spec)
        if n < config.MIN_CHARUCO_CORNERS:
            rejected.append(f"{p.name}: only {n} corners")
            continue
        if board.checkCharucoCornersCollinear(ids):
            rejected.append(f"{p.name}: collinear corners")
            continue
        o, i = board.matchImagePoints(corners, ids)
        if o is None or i is None or len(o) < config.MIN_CHARUCO_CORNERS:
            rejected.append(f"{p.name}: matchImagePoints returned too few points")
            continue
        obj_pts.append(cv_points_to_board(o, spec).astype(np.float32).reshape(-1, 1, 3))
        img_pts.append(np.asarray(i, dtype=np.float32).reshape(-1, 1, 2))
        used.append(p.name)

    if len(used) < MIN_CALIB_VIEWS:
        raise RuntimeError(
            f"only {len(used)} of {len(paths)} images produced a usable board view "
            f"(need {MIN_CALIB_VIEWS}). Rejections: {rejected[:12]}"
        )
    assert size_wh is not None

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, size_wh, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    if not np.isfinite(rms):
        raise RuntimeError("cv2.calibrateCamera returned a non-finite RMS")
    if rms > CALIB_HARD_FAIL_RMS_PX:
        raise RuntimeError(
            f"calibration RMS {rms:.3f} px exceeds {CALIB_HARD_FAIL_RMS_PX} px over {len(used)} views. "
            "The images are probably blurred, or square_mm/dict_name do not match the printed board."
        )
    if rms > config.MAX_REPROJ_RMS_PX:
        print(
            f"[charuco] WARNING: calibration RMS {rms:.3f} px is above the per-frame gate "
            f"({config.MAX_REPROJ_RMS_PX} px); poses will mostly be rejected downstream.",
            file=sys.stderr,
        )

    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    return Intrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        dist=[float(v) for v in dist[:5]] + [0.0] * max(0, 5 - len(dist)),
        width=int(size_wh[0]), height=int(size_wh[1]), rms_px=float(rms),
    )


def poses_for_dir(
    image_dir: Path, spec: BoardSpec, intr: Intrinsics
) -> dict[str, FramePose]:
    """Metric poses for every image in a directory, keyed by filename stem.

    Frames whose pose fails the quality gates are simply absent from the
    result -- the count is the caller's quality signal.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise RuntimeError(f"{image_dir} is not a directory")
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not files:
        raise RuntimeError(f"no images found in {image_dir} (looked for {_IMAGE_EXTS})")

    poses: dict[str, FramePose] = {}
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if intr.width and (img.shape[1], img.shape[0]) != (intr.width, intr.height):
            raise RuntimeError(
                f"{p.name} is {img.shape[1]}x{img.shape[0]} but intrinsics were calibrated at "
                f"{intr.width}x{intr.height}; use Intrinsics.scaled_to() first"
            )
        pose = pose_for_frame(img, spec, intr, name=p.stem)
        if pose is not None:
            poses[p.stem] = pose
    return poses


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _look_at(eye_mm: np.ndarray, target_mm: np.ndarray, up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth BOARD->camera rotation/translation for a camera at
    `eye_mm` looking at `target_mm` (both in BOARD mm).

    Camera axes follow OpenCV: x right, y down, z forward. `up` is the BOARD
    direction that should appear upward in the image; the board's +Y is up the
    printed page, so that is the natural default.
    """
    eye = np.asarray(eye_mm, dtype=np.float64).reshape(3)
    tgt = np.asarray(target_mm, dtype=np.float64).reshape(3)
    z = tgt - eye
    z /= np.linalg.norm(z)
    up = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up)                  # x right when `up` appears upward
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)      # rows are camera axes in BOARD frame
    t = -R @ eye
    return R, t


def _synth_view(
    spec: BoardSpec, R: np.ndarray, t: np.ndarray, K: np.ndarray, D: np.ndarray,
    img_wh: tuple[int, int], src_px_per_mm: float = 8.0,
) -> np.ndarray:
    """Render a photo-like grayscale view of the printed board from a known pose."""
    pps = int(round(spec.square_mm * src_px_per_mm))
    s = pps / spec.square_mm                     # actual src px per mm
    bw, bh = spec.squares_x * pps, spec.squares_y * pps
    quiet = int(round(10.0 * s))
    board_img = make_board(spec).generateImage((bw, bh), marginSize=0, borderBits=1)
    src = np.full((bh + 2 * quiet, bw + 2 * quiet), 255, np.uint8)
    src[quiet:quiet + bh, quiet:quiet + bw] = board_img

    # src pixel (u,v) -> BOARD mm (X,Y). Strip the quiet margin, convert to mm,
    # shift the origin to the top-left interior corner, and negate Y because
    # BOARD +Y runs *up* the printed page while image rows run down it.
    M = np.array(
        [[1.0 / s, 0.0, -quiet / s - spec.square_mm],
         [0.0, -1.0 / s, quiet / s + spec.square_mm],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    P = K @ np.column_stack([R[:, 0], R[:, 1], t.reshape(3)])   # board plane Z=0
    H = P @ M
    view = cv2.warpPerspective(
        src, H, img_wh, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    if np.any(np.abs(np.asarray(D).reshape(-1)) > 1e-12):
        raise ValueError("_synth_view models a pinhole camera only; pass D = zeros")
    return view


def _verify_raster(png: Path) -> list[str]:
    """Re-measure a rendered target from its own pixels.

    Everything downstream trusts that the printed square really is
    `effective_square_mm` across, so read that number back out of the raster
    with the real detector instead of trusting the drawing code.
    """
    problems: list[str] = []
    side = json.loads(png.with_suffix(".json").read_text())
    ppmm = side["dpi"] / MM_PER_INCH
    spec = BoardSpec.from_json(side["spec"])
    with Image.open(png) as im:
        arr = np.array(im.convert("L"))
    dark = arr < 128
    h = arr.shape[0]

    # The 100 mm bar shares its rows with the vertical bar, so measure the
    # first contiguous run of ink rather than the row's full extent.
    target_px = int(round(RULER_MM * ppmm))
    bar = None
    for r in range(int(h * 0.55), h):
        cols = np.flatnonzero(dark[r])
        if len(cols) == 0:
            continue
        brk = np.flatnonzero(np.diff(cols) > 1)
        end = cols[brk[0]] if len(brk) else cols[-1]
        if abs(int(end - cols[0] + 1) - target_px) <= 2:
            bar = (int(cols[0]), int(end))
            break
    if bar is None:
        problems.append(f"{png.name}: could not find the {RULER_MM} mm bar in the raster")
    else:
        inked_mm = (bar[1] - bar[0] + 1) / ppmm
        print(f"    ruler bar inked extent = {bar[1] - bar[0] + 1} px = {inked_mm:.4f} mm "
              f"(sidecar {side['ruler_mm_rendered']:.4f}, nominal {RULER_MM})")
        if abs(inked_mm - side["ruler_mm_rendered"]) > 0.01:
            problems.append(f"{png.name}: ruler bar is {inked_mm:.4f} mm, sidecar claims "
                            f"{side['ruler_mm_rendered']:.4f}")

    corners, ids, n = detect(arr, spec)
    if n != spec.n_corners:
        problems.append(f"{png.name}: detector found {n}/{spec.n_corners} corners in our own render")
        return problems
    pts = corners.reshape(-1, 2)
    idv = ids.reshape(-1)
    nx = spec.squares_x - 1
    grid = np.stack([idv % nx, idv // nx], axis=1).astype(np.float64)
    A = np.column_stack([grid, np.ones(len(grid))])
    sol, _, _, _ = np.linalg.lstsq(A, pts, rcond=None)
    resid = float(np.abs(pts - A @ sol).max())
    pitch_x_mm, pitch_y_mm = sol[0, 0] / ppmm, sol[1, 1] / ppmm
    print(f"    detector re-measures pitch = {pitch_x_mm:.5f} x {pitch_y_mm:.5f} mm/square "
          f"(sidecar {side['effective_square_mm']:.5f}); grid residual {resid * 1000 / ppmm:.1f} um")
    for axis, val in (("x", pitch_x_mm), ("y", pitch_y_mm)):
        if abs(val - side["effective_square_mm"]) > 1e-3:
            problems.append(f"{png.name}: measured {axis} pitch {val:.5f} mm != sidecar "
                            f"{side['effective_square_mm']:.5f} mm")
    if resid > 0.05:
        problems.append(f"{png.name}: corner grid residual {resid:.3f} px -- raster is not square")
    return problems


def _selftest() -> int:
    np.set_printoptions(precision=6, suppress=True)
    spec = BoardSpec()
    fails: list[str] = []

    print("=" * 78)
    print("1. TARGET RENDER")
    print("=" * 78)
    for dpi in (300, 600):
        out = config.TARGETS_DIR / f"charuco_{spec.squares_x}x{spec.squares_y}_{spec.square_mm:g}mm_{dpi}dpi.png"
        render_target(spec, out, dpi=dpi)
        side = json.loads(out.with_suffix(".json").read_text())
        ppmm = dpi / MM_PER_INCH
        exp_sq = int(round(spec.square_mm * ppmm))
        with Image.open(out) as im:
            w, h = im.size
        print(f"  {out.name}")
        print(f"    dpi={dpi}  ppmm={ppmm:.6f}  px/square={side['px_per_square']} (expected {exp_sq})")
        print(f"    effective square = {side['effective_square_mm']:.5f} mm "
              f"(nominal {spec.square_mm:.2f}, error {side['effective_square_mm'] - spec.square_mm:+.5f} mm)")
        print(f"    page = {w} x {h} px = {side['page_mm'][0]:.2f} x {side['page_mm'][1]:.2f} mm")
        print(f"    rendered ruler = {side['ruler_mm_rendered']:.4f} mm (nominal {RULER_MM})")
        if side["px_per_square"] != exp_sq:
            fails.append(f"px_per_square mismatch at {dpi} dpi")
        if abs(w / ppmm - side["page_mm"][0]) > 1e-6:
            fails.append("page width mm/px inconsistent")
        if not out.with_suffix(".pdf").exists():
            fails.append(f"no PDF emitted for {out.name}")
        fails.extend(_verify_raster(out))

    # The default 7x5 @ 30 mm board is 210 mm wide and cannot be printed at
    # 100% on Letter or A4, so also emit a variant that fits a normal printer.
    letter_spec = BoardSpec(squares_x=7, squares_y=5, square_mm=24.0, marker_mm=18.0)
    lp = render_target(
        letter_spec, config.TARGETS_DIR / "charuco_7x5_24mm_letter_300dpi.png",
        dpi=300, paper="letter",
    )
    lside = json.loads(lp.with_suffix(".json").read_text())
    print(f"  {lp.name}  (paper=letter)")
    print(f"    page = {lside['page_mm'][0]:.2f} x {lside['page_mm'][1]:.2f} mm "
          f"(letter is 215.90 x 279.40)")
    print(f"    effective square = {lside['effective_square_mm']:.5f} mm")
    if abs(lside["page_mm"][0] - 215.9) > 0.2 or abs(lside["page_mm"][1] - 279.4) > 0.2:
        fails.append("letter page is not letter-sized")
    fails.extend(_verify_raster(lp))

    print()
    print("=" * 78)
    print("2. ANALYTIC PnP ROUND TRIP (projectPoints -> solvePnP)")
    print("=" * 78)
    W, H = 3840, 2160
    K = np.array([[3000.0, 0.0, 1919.5], [0.0, 3000.0, 1079.5], [0.0, 0.0, 1.0]])
    D = np.zeros((5, 1))
    objp = board_object_points(spec)
    print(f"  board object points: {objp.shape[0]} corners, first={objp[0]}, last={objp[-1]}")
    if not np.allclose(objp[0], [0.0, 0.0, 0.0]):
        fails.append("BOARD origin is not the top-left interior corner")
    span = objp.max(axis=0) - objp.min(axis=0)
    exp_span = [(spec.squares_x - 2) * spec.square_mm, (spec.squares_y - 2) * spec.square_mm, 0.0]
    print(f"  corner grid span = {span} mm (expected {exp_span})")
    if not np.allclose(span, exp_span):
        fails.append("interior corner spacing is not square_mm")

    centre_mm = board_center_mm(spec)
    print(f"  board centre = {centre_mm} mm")
    eyes = [
        centre_mm + np.array([15.0, 15.0, 400.0]),
        centre_mm + np.array([-195.0, -5.0, 380.0]),
        centre_mm + np.array([185.0, 165.0, 300.0]),
        centre_mm + np.array([-5.0, -205.0, 520.0]),
    ]
    worst = 0.0
    for eye in eyes:
        Rt, tt = _look_at(eye, centre_mm)
        proj, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), cv2.Rodrigues(Rt)[0], tt.reshape(3, 1), K, D)
        imgp = proj.reshape(-1, 2)
        rvec, tvec = _solve_pnp(objp, imgp, K, D)
        Rr = cv2.Rodrigues(rvec)[0]
        cr = (-Rr.T @ tvec.reshape(3))
        err = float(np.linalg.norm(cr - eye))
        rms = _reproj_rms_px(objp, imgp, rvec, tvec, K, D)
        ang = float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rr @ Rt.T)[0])))
        worst = max(worst, err)
        print(f"  true centre {eye}  ->  recovered {cr}")
        print(f"    centre error = {err:.9f} mm | rotation error = {ang:.3e} deg | reproj RMS = {rms:.3e} px")
        if err > 0.01:
            fails.append(f"analytic centre error {err} mm > 0.01 mm")
    print(f"  WORST analytic centre error: {worst:.9f} mm  (gate: < 0.01 mm)")

    print()
    print("=" * 78)
    print("3. END-TO-END SYNTHETIC IMAGE (render -> warp -> detect -> pose_for_frame)")
    print("=" * 78)
    intr = Intrinsics(fx=3000.0, fy=3000.0, cx=1919.5, cy=1079.5,
                      dist=[0.0] * 5, width=W, height=H, rms_px=0.0)
    e2e_worst = 0.0
    e2e_n = 0
    for eye in eyes:
        Rt, tt = _look_at(eye, centre_mm)
        view = _synth_view(spec, Rt, tt, K, D, (W, H))
        _c, _i, n = detect(view, spec)
        pose = pose_for_frame(view, spec, intr, name="synthetic")
        if pose is None:
            fails.append(f"pose_for_frame returned None for synthetic view at {eye}")
            print(f"  true centre {eye}: detected {n} corners, POSE REJECTED")
            continue
        e2e_n += 1
        err = float(np.linalg.norm(pose.center_np - eye))
        e2e_worst = max(e2e_worst, err)
        chk = -pose.R_np.T @ np.array(pose.t)
        print(f"  true centre {eye}  ->  recovered {pose.center_np}")
        print(f"    corners={pose.n_corners}/{spec.n_corners}  centre error={err:.4f} mm  "
              f"reproj RMS={pose.reproj_rms_px:.4f} px  center==-R.T@t: {np.allclose(chk, pose.center_np)}")
        if not np.allclose(chk, pose.center_np, atol=1e-9):
            fails.append("FramePose.center != -R.T @ t")
        if err > 1.0:
            fails.append(f"end-to-end centre error {err:.3f} mm > 1.0 mm")
    print(f"  end-to-end views solved: {e2e_n}/{len(eyes)}, worst centre error {e2e_worst:.4f} mm")

    print()
    print("=" * 78)
    print("4. calibrate_camera + poses_for_dir ON SYNTHETIC VIEWS")
    print("=" * 78)
    tmp = config.OUT_DIR / "_charuco_selftest_views"
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("*.png"):
        old.unlink()
    rng = np.random.default_rng(7)
    truth: dict[str, np.ndarray] = {}
    for i in range(14):
        eye = centre_mm + np.array([
            rng.uniform(-190, 190), rng.uniform(-190, 190), rng.uniform(280, 560)
        ])
        Rt, tt = _look_at(eye, centre_mm + np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), 0.0]))
        view = _synth_view(spec, Rt, tt, K, D, (W, H))
        _c, _i, n = detect(view, spec)
        if n < config.MIN_CHARUCO_CORNERS:
            continue
        name = f"view{i:02d}"
        cv2.imwrite(str(tmp / f"{name}.png"), view)
        truth[name] = eye
    print(f"  wrote {len(truth)} synthetic views to {tmp}")
    cal = calibrate_camera(sorted(tmp.glob("*.png")), spec)
    print(f"  calibrated: fx={cal.fx:.3f} (true 3000)  fy={cal.fy:.3f} (true 3000)")
    print(f"              cx={cal.cx:.3f} (true 1919.5)  cy={cal.cy:.3f} (true 1079.5)")
    print(f"              dist={np.array(cal.dist)}")
    print(f"              rms_px={cal.rms_px:.5f}  ({cal.width}x{cal.height})")
    for label, got, want in (("fx", cal.fx, 3000.0), ("fy", cal.fy, 3000.0),
                             ("cx", cal.cx, 1919.5), ("cy", cal.cy, 1079.5)):
        if abs(got - want) / abs(want) > 0.01:
            fails.append(f"calibrated {label}={got} differs from truth {want} by >1%")

    poses = poses_for_dir(tmp, spec, intr)
    errs = [float(np.linalg.norm(poses[k].center_np - truth[k])) for k in poses]
    print(f"  poses_for_dir: {len(poses)}/{len(truth)} frames solved")
    if errs:
        print(f"  centre error vs truth (using the TRUE K): "
              f"mean {np.mean(errs):.4f} mm, max {np.max(errs):.4f} mm")
        print(f"  reproj RMS: mean {np.mean([poses[k].reproj_rms_px for k in poses]):.4f} px")
    if len(poses) < len(truth):
        fails.append(f"poses_for_dir solved only {len(poses)}/{len(truth)}")

    print()
    print("=" * 78)
    print("5. REJECTION BEHAVIOUR")
    print("=" * 78)
    blank = np.full((H, W), 255, np.uint8)
    c, i, n = detect(blank, spec)
    print(f"  blank image: detect -> n={n}, shapes {c.shape}/{i.shape}; "
          f"pose_for_frame -> {pose_for_frame(blank, spec, intr)}")
    if n != 0 or pose_for_frame(blank, spec, intr) is not None:
        fails.append("blank image was not rejected")
    noise = rng.integers(0, 255, (720, 1280), dtype=np.uint8)
    print(f"  random noise: pose_for_frame -> {pose_for_frame(noise, spec, intr)}")
    if pose_for_frame(noise, spec, intr) is not None:
        fails.append("noise image was not rejected")
    try:
        calibrate_camera(sorted(tmp.glob("*.png"))[:2], spec)
        fails.append("calibrate_camera accepted 2 views")
    except RuntimeError as e:
        print(f"  calibrate_camera(2 views) correctly raised: {str(e)[:90]}...")
    try:
        render_target(BoardSpec(square_mm=30.0), config.OUT_DIR / "_toobig.png", dpi=300, paper="letter")
        fails.append("render_target accepted an oversized board on letter")
    except ValueError as e:
        print(f"  render_target(paper='letter', 210mm board) correctly raised: {str(e)[:90]}...")

    print()
    print("=" * 78)
    if fails:
        print(f"SELF-TEST FAILED ({len(fails)} problems)")
        for f in fails:
            print("  - " + f)
        return 1
    print("SELF-TEST PASSED")
    print(f"  worst analytic centre error   : {worst:.9f} mm")
    print(f"  worst end-to-end centre error : {e2e_worst:.4f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
