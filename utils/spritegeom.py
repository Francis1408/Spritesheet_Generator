"""
spritegeom.py - single source of truth for sprite sheet geometry.

Every later stage (dataset build, conditioning maps, inference, post-processing,
metrics) MUST import its geometry from here. If two scripts disagree about the
quadrant map or the upscale factor, your positional captions become wrong and no
amount of ControlNet tuning will fix it.

Run order:
    1. step0a_verify_quadrants.py  -> confirm / fix QUADRANT_CELL below
    2. step0b_measure.py           -> fill in WINDOW_W, WINDOW_H, UPSCALE below
    3. step1_vae_roundtrip.py      -> gate: is the VAE floor acceptable?
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from config import QUAD_SIZE, SHEET_SIZE

# --------------------------------------------------------------------------
# Background definition. Import these in color_profile.py / sprite_crops.py
# so the crop, the color profile and the training data all agree.
# --------------------------------------------------------------------------
BG_THRESHOLD = 245
ALPHA_THRESHOLD = 10

# Background used when flattening RGBA -> RGB for training.
# Avoid magenta: it is far out of SD's distribution and bleeds into character
# edges as color fringing, which then corrupts palette snapping.
FLATTEN_BG = (128, 128, 128)

# --------------------------------------------------------------------------
# Sheet layout
# --------------------------------------------------------------------------
NATIVE_CELL = 64             # native sprite cell size in the source assets

# Upscale factor used in your CURRENT sheets (nearest-neighbour 64 -> 256).
# Only used to convert measurements taken on existing sheets back to native px.
LEGACY_UPSCALE = QUAD_SIZE // NATIVE_CELL  # 4

# --------------------------------------------------------------------------
# QUADRANT MAP  --  VERIFY THIS WITH step0a_verify_quadrants.py
#
# Values are (col, row) in the 2x2 grid: (0,0)=top-left, (1,0)=top-right,
# (0,1)=bottom-left, (1,1)=bottom-right.
#
# The default below follows Figure 1 of the TCC and the positional captions
# ("top-left front view, top-right right-facing view, bottom-left back view,
#  bottom-right left-facing view").
#
# NOTE: your sprite_crops.py QUADRANTS dict uses a DIFFERENT convention
# (front=TL, back=TR, left=BL, right=BR). Exactly one of the two matches the
# files on disk. Resolve it before training.
# --------------------------------------------------------------------------
QUADRANT_CELL = {
    "front": (0, 0),
    "back": (1, 0),
    "left":  (0, 1),
    "right":  (1, 1),
}

POSITIONS = ("front", "right", "back", "left")


def quadrant_box(position: str, quad_size: int = QUAD_SIZE):
    """PIL crop box (left, upper, right, lower) for a position."""
    col, row = QUADRANT_CELL[position]
    return (col * quad_size, row * quad_size,
            (col + 1) * quad_size, (row + 1) * quad_size)


# --------------------------------------------------------------------------
# LOCKED GEOMETRY -- fill in from step0b_measure.py output, then never change
# --------------------------------------------------------------------------
WINDOW_W: int | None = 51  # native px, e.g. 42
WINDOW_H: int | None = 51  # native px, e.g. 52
UPSCALE: int | None = 5    # integer factor, e.g. 6


def _require_locked():
    if WINDOW_W is None or WINDOW_H is None or UPSCALE is None:
        raise RuntimeError(
            "Geometry not locked. Run step0b_measure.py and set WINDOW_W, "
            "WINDOW_H and UPSCALE in spritegeom.py before using this function."
        )


def content_size():
    """Size in quadrant px of the upscaled window."""
    _require_locked()
    return WINDOW_W * UPSCALE, WINDOW_H * UPSCALE


def content_offset():
    """
    Offset of the upscaled window inside its 256x256 quadrant.

    Centred, so the inverse transform is a fixed crop. Must be exact: any
    ambiguity here makes the 256 -> native downsample unrecoverable.
    """
    cw, ch = content_size()
    if cw > QUAD_SIZE or ch > QUAD_SIZE:
        raise ValueError(
            f"window {WINDOW_W}x{WINDOW_H} at {UPSCALE}x = {cw}x{ch} px, "
            f"which does not fit in a {QUAD_SIZE}px quadrant"
        )
    return (QUAD_SIZE - cw) // 2, (QUAD_SIZE - ch) // 2


# --------------------------------------------------------------------------
# Masking / bbox
# --------------------------------------------------------------------------
def character_mask(arr: np.ndarray,
                   bg_threshold: int = BG_THRESHOLD,
                   alpha_threshold: int = ALPHA_THRESHOLD) -> np.ndarray:
    """Boolean mask of visible character pixels in an RGBA array."""
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    is_background = (r > bg_threshold) & (g > bg_threshold) & (b > bg_threshold)
    is_transparent = a < alpha_threshold
    mask = ~(is_background | is_transparent)
    if not mask.any():
        # Fallback: a fully opaque pale sprite on a white background is
        # indistinguishable from background by RGB. Trust alpha alone.
        mask = a >= alpha_threshold
    return mask


def tight_bbox(img: Image.Image):
    """
    Tight bounding box of the character, NO padding.

    Padding belongs to the fixed window, not to the measurement. Returns
    (left, top, right, bottom) with right/bottom exclusive, or None if empty.
    """
    arr = np.array(img.convert("RGBA"))
    mask = character_mask(arr)
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return int(left), int(top), int(right) + 1, int(bottom) + 1


# --------------------------------------------------------------------------
# Forward transform: native cell -> quadrant
# --------------------------------------------------------------------------
def crop_window(native_img: Image.Image, bbox=None):
    """
    Crop a fixed WINDOW_W x WINDOW_H window from a native cell.

    Character is bottom-aligned (consistent ground line across the dataset,
    which matters because the mask boundary and conditioning maps live in the
    same coordinate frame) and horizontally centred.

    Returns (cropped RGBA image, clipped: bool). `clipped` is True when the
    character did not fit -- track these, do not let them silently through.
    """
    _require_locked()
    if bbox is None:
        bbox = tight_bbox(native_img)
    if bbox is None:
        return None, False

    left, top, right, bottom = bbox
    cw, chh = right - left, bottom - top
    clipped = cw > WINDOW_W or chh > WINDOW_H

    # horizontally centre on the character's centre of mass
    cx = (left + right) / 2.0
    x0 = int(round(cx - WINDOW_W / 2.0))
    # bottom-align: window bottom sits on the character's bottom
    y0 = bottom - WINDOW_H

    # keep the window inside the native cell
    w, h = native_img.size
    x0 = max(0, min(x0, w - WINDOW_W))
    y0 = max(0, min(y0, h - WINDOW_H))

    return native_img.convert("RGBA").crop(
        (x0, y0, x0 + WINDOW_W, y0 + WINDOW_H)), clipped


def upscale_nearest(img: Image.Image, factor: int | None = None):
    """Integer nearest-neighbour upscale. Never use any other filter here."""
    if factor is None:
        _require_locked()
        factor = UPSCALE
    return img.resize((img.width * factor, img.height * factor),
                      Image.Resampling.NEAREST)


def flatten(img: Image.Image, bg=FLATTEN_BG):
    """
    Composite RGBA onto an opaque background.

    A plain .convert("RGB") turns every transparent pixel BLACK, which
    produces phantom dark outlines and cloaks in both captions and training.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    canvas = Image.new("RGB", img.size, bg)
    canvas.paste(img, (0, 0), img)
    return canvas


def place_in_quadrant(content: Image.Image, bg=FLATTEN_BG):
    """Paste the upscaled window into a full 256x256 quadrant at the fixed offset."""
    ox, oy = content_offset()
    quad = Image.new("RGB", (QUAD_SIZE, QUAD_SIZE), bg)
    if content.mode == "RGBA":
        quad.paste(content, (ox, oy), content)
    else:
        quad.paste(content, (ox, oy))
    return quad


# --------------------------------------------------------------------------
# Inverse transform: quadrant -> native cell  (Phase III post-processing)
# --------------------------------------------------------------------------
def quadrant_to_native(quad_img: Image.Image, reduce: str = "median"):
    """
    Invert the forward transform: 256x256 quadrant -> WINDOW_W x WINDOW_H native.

    Per-block median (not nearest) because a single sampled pixel may sit on a
    VAE artifact, while the median over UPSCALE^2 pixels is robust to a few
    corrupted ones.
    """
    _require_locked()
    ox, oy = content_offset()
    cw, ch = content_size()
    arr = np.array(quad_img.convert("RGB").crop((ox, oy, ox + cw, oy + ch)))

    blocks = arr.reshape(WINDOW_H, UPSCALE, WINDOW_W, UPSCALE, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(
        WINDOW_H, WINDOW_W, UPSCALE * UPSCALE, 3)

    if reduce == "median":
        out = np.median(blocks, axis=2)
    elif reduce == "mean":
        out = blocks.mean(axis=2)
    elif reduce == "mode":
        # exact-tuple mode: slower, but returns a color that really occurred
        flat = (blocks[..., 0].astype(np.int64) << 16 |
                blocks[..., 1].astype(np.int64) << 8 |
                blocks[..., 2].astype(np.int64))
        out = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
        for y in range(WINDOW_H):
            for x in range(WINDOW_W):
                vals, counts = np.unique(flat[y, x], return_counts=True)
                v = int(vals[counts.argmax()])
                out[y, x] = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
        return out
    else:
        raise ValueError(f"unknown reduce: {reduce}")

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
def extract_palette(arr: np.ndarray, max_colors: int = 512) -> np.ndarray:
    """Unique RGB colors present in a native-resolution array, (N, 3) uint8."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    flat = arr.reshape(-1, 3)
    pal = np.unique(flat, axis=0)
    if len(pal) > max_colors:
        raise ValueError(
            f"{len(pal)} unique colors -- this does not look like clean "
            f"pixel art. Check the upscale was nearest-neighbour."
        )
    return pal.astype(np.uint8)


def snap_palette(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Snap every pixel to the nearest palette entry (squared L2 in RGB).

    RGB distance is adequate here because pixel-art palettes have well
    separated entries. Switch to CIELAB only if you see wrong-hue snaps.
    """
    h, w = arr.shape[:2]
    px = arr.reshape(-1, 1, 3).astype(np.int32)
    pal = palette.reshape(1, -1, 3).astype(np.int32)
    d = ((px - pal) ** 2).sum(axis=2)
    return palette[d.argmin(axis=1)].reshape(h, w, 3).astype(np.uint8)
