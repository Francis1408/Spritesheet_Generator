"""
sprite_crops.py

Extracts per-position character crops from a 2x2 sprite sheet and builds a
compact full-sheet image for the full-view VLM call.

Design notes:
- Crops are returned RAW (tight bbox, small pad, RGBA, NOT upscaled). This is
  what the color profile should consume: percentages are scale-invariant, so
  working on raw pixels is faster and avoids any resampling ambiguity.
- Upscaling and alpha flattening happen at the point of use (VLM encoding),
  via scale_to_target() and flatten(). Keeping them separate means the color
  profile never sees a flattened background color it would then measure.
- BG_THRESHOLD / ALPHA_THRESHOLD live here and should be imported by
  color_profile.py so the crop and the profile always agree on what counts
  as background.
"""

from PIL import Image
import numpy as np
from pathlib import Path

QUADRANTS = {
    "front": (0, 0, 256, 256),
    "back":  (256, 0, 512, 256),
    "left":  (0, 256, 256, 512),
    "right": (256, 256, 512, 512),
}

# Shared background definition — import these in color_profile.py
BG_THRESHOLD = 245
ALPHA_THRESHOLD = 10

# Background color used when flattening alpha. Mid-gray collides with
# gray-skinned characters; magenta never collides with anything but looks odd.
# Whichever you pick, name it in the VLM prompts as background.
FLATTEN_BG = (128, 128, 128)

CROP_PAD = 2          # raw pixels of margin kept around the character
VLM_CROP_TARGET = 512  # target longest edge for a single-position crop
VLM_SHEET_TARGET = 1024  # target longest edge for the compact full sheet


def _character_mask(arr, bg_threshold, alpha_threshold):
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    is_background = (r > bg_threshold) & (g > bg_threshold) & (b > bg_threshold)
    is_transparent = a < alpha_threshold
    return ~(is_background | is_transparent)


def get_character_bbox(quad_img, bg_threshold=BG_THRESHOLD,
                       alpha_threshold=ALPHA_THRESHOLD, pad=CROP_PAD):
    """
    Bounding box of the visible character in a quadrant, with a small pad.

    Falls back to alpha-only detection when the RGB threshold finds nothing —
    that case is a white or very pale character being erased by the near-white
    background test, which is exactly the sprite you least want to drop.

    Returns (left, top, right, bottom) or None.
    """
    arr = np.array(quad_img.convert("RGBA"))
    h, w = arr.shape[:2]

    mask = _character_mask(arr, bg_threshold, alpha_threshold)

    if not mask.any():
        # Fallback: trust alpha alone. A fully opaque pale sprite on a white
        # background is indistinguishable from background by RGB, so if this
        # also fails the quadrant is genuinely empty.
        mask = arr[:, :, 3] >= alpha_threshold

    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    return (
        int(max(0, left - pad)),
        int(max(0, top - pad)),
        int(min(w, right + 1 + pad)),
        int(min(h, bottom + 1 + pad)),
    )


def scale_to_target(img, target=VLM_CROP_TARGET, max_factor=16):
    """
    Integer nearest-neighbour upscale so the longest edge approaches `target`.

    Integer factors only: any non-integer scale, or any filter other than
    NEAREST, softens the hard pixel edges the prompts ask the VLM to read
    (thin trim lines, plate seams, small buckles).
    """
    longest = max(img.size)
    if longest == 0:
        return img
    factor = min(max_factor, max(1, target // longest))
    if factor == 1:
        return img
    return img.resize((img.width * factor, img.height * factor),
                      Image.Resampling.NEAREST)


def flatten(img, bg=FLATTEN_BG):
    """
    Composite RGBA onto an opaque background.

    Without this, a plain .convert("RGB") turns every transparent pixel BLACK
    and the VLM reports black boots, dark cloaks, and outlines that do not
    exist in the sprite.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    canvas = Image.new("RGB", img.size, bg)
    canvas.paste(img, (0, 0), img)
    return canvas


def extract_position_crops(image_path, available_pos):
    """
    Extract the four character positions as raw, tightly-cropped RGBA images.

    Returns (crops, missing):
        crops   -> {"front": PIL.Image, ...} raw, unscaled, RGBA
        missing -> ["back", ...] positions with no detectable character

    `missing` is returned rather than silently skipped: synthesis needs to know
    a view was absent instead of quietly reasoning from three views.
    """
    image = Image.open(image_path).convert("RGBA")

    crops = {}
    missing = []

    for position, box in QUADRANTS.items():

        if available_pos[position]:
            quad_img = image.crop(box)
            bbox = get_character_bbox(quad_img)

            if bbox is None:
                missing.append(position)
                continue

        else:
            missing.append(position)
            continue

        character_img = quad_img.crop(bbox)
        if character_img.width == 0 or character_img.height == 0:
            missing.append(position)
            continue

        crops[position] = character_img


    return crops, missing


def build_compact_sheet(crops, pad=4, target=VLM_SHEET_TARGET):
    """
    Reassemble the raw crops into a dense 2x2 sheet, then upscale once.

    The original 512x512 sheet is mostly empty: a 40x60 sprite inside a
    256x256 quadrant is ~97% background, so the vision encoder spends its
    patch budget on nothing. This is the likely cause of full-sheet quality
    dropping on small characters.

    Cells are uniform (max crop width/height) so relative character sizes are
    preserved across views. Characters are bottom-aligned and horizontally
    centred, which keeps the sheet readable as a sprite sheet.

    Returns a flattened RGB image ready for the VLM, or None if no crops.
    """
    if not crops:
        return None

    order = [p for p in ("front", "back", "left", "right") if p in crops]
    cw = max(c.width for c in crops.values())
    ch = max(c.height for c in crops.values())

    sheet = Image.new("RGBA", (cw * 2 + pad * 3, ch * 2 + pad * 3), (0, 0, 0, 0))

    for i, position in enumerate(order):
        c = crops[position]
        col, row = i % 2, i // 2
        x = pad + col * (cw + pad) + (cw - c.width) // 2
        y = pad + row * (ch + pad) + (ch - c.height)  # bottom-aligned
        sheet.paste(c, (x, y), c)

    return flatten(scale_to_target(sheet, target=target))