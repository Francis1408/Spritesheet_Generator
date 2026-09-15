# color_profile.py

from PIL import Image
import numpy as np
import json

REGIONS = {
    "upper":  (0.00, 0.45),
    "middle": (0.30, 0.75),
    "lower":  (0.65, 1.00),
}

# Load CSS color pallete
with open("schemas/color_palletes.json", "r") as file:
   _CSS3_NAMES_TO_HEX = json.load(file) 

_CSS3_NAMES = list(_CSS3_NAMES_TO_HEX.keys())
_CSS3_RGB = np.array([
    [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
    for h in _CSS3_NAMES_TO_HEX.values()
])


def closest_css3_name(hex_color):
    """Nearest named color by simple Euclidean RGB distance. Deterministic,
    no model call — just a lookup table."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    dists = ((_CSS3_RGB - np.array([r, g, b])) ** 2).sum(axis=1)
    return _CSS3_NAMES[int(dists.argmin())]


def get_character_mask(crop, bg_threshold=240, alpha_threshold=10):
    """Boolean mask of non-background pixels, plus the raw RGBA array."""
    arr = np.array(crop.convert("RGBA"))
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    is_bg = (r > bg_threshold) & (g > bg_threshold) & (b > bg_threshold)
    is_transparent = a < alpha_threshold
    mask = ~(is_bg | is_transparent)
    return mask, arr


def build_global_palette(position_crops, n_colors=16):
    """
    Pools non-background pixels from ALL views into one set and quantizes
    them together, so the same visual color gets the same palette bucket
    regardless of which view it appears in. This is what makes cross-view
    percentage comparison ("42% in back-middle, 4% in front-upper") valid.
    """
    all_pixels = []
    for crop in position_crops.values():
        mask, arr = get_character_mask(crop)
        pixels = arr[mask][:, :3]
        if len(pixels) > 0:
            all_pixels.append(pixels)

    if not all_pixels:
        return []

    all_pixels = np.concatenate(all_pixels, axis=0)

    # Cap sample size for speed on large sprite sheets
    if len(all_pixels) > 20000:
        idx = np.random.choice(len(all_pixels), 20000, replace=False)
        all_pixels = all_pixels[idx]

    # Build a synthetic image so we can reuse PIL's quantizer
    n = len(all_pixels)
    side = int(np.floor(np.sqrt(n)))
    trimmed = all_pixels[: side * side]
    synthetic_img = Image.fromarray(trimmed.reshape(side, side, 3), "RGB")

    quantized = synthetic_img.quantize(colors=n_colors, method=Image.MEDIANCUT)
    raw_palette = quantized.getpalette()[: n_colors * 3]
    palette = [tuple(raw_palette[i:i + 3]) for i in range(0, len(raw_palette), 3)]
    return palette


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def compute_color_profile(position_crops, palette, regions=REGIONS, min_significant_pct=0.05):
    """
    For each shared palette color, computes what percentage of each
    view/region it occupies. Returns:
    { hex_color: { "views": { position: { region: pct, ... }, ... } } }
    """
    if not palette:
        return {}

    palette_arr = np.array(palette, dtype=int)
    profile = {}

    for position, crop in position_crops.items():
        mask, arr = get_character_mask(crop)
        h, w = mask.shape

        for region_name, (start_frac, end_frac) in regions.items():
            y0, y1 = int(h * start_frac), int(h * end_frac)

            region_mask = np.zeros_like(mask)
            region_mask[y0:y1, :] = mask[y0:y1, :]

            region_pixels = arr[region_mask][:, :3].astype(int)
            total = len(region_pixels)
            if total == 0:
                continue

            # Vectorized nearest-palette-color assignment
            diffs = region_pixels[:, None, :] - palette_arr[None, :, :]
            dists = (diffs ** 2).sum(axis=2)
            nearest_idx = dists.argmin(axis=1)

            counts = np.bincount(nearest_idx, minlength=len(palette))
            pcts = counts / total

            for idx, pct in enumerate(pcts):
                if pct < min_significant_pct:
                    continue
                hex_color = rgb_to_hex(palette[idx])
                profile.setdefault(hex_color, {"views": {}})
                profile[hex_color]["views"].setdefault(position, {})
                profile[hex_color]["views"][position][region_name] = round(float(pct), 3)

    return profile


def format_color_profile_for_prompt(profile, top_n=8):
    """
    Converts the raw profile dict into the sorted, size-limited structure
    ready to inject into an LLM prompt, plus a plain-language interpretation
    guide explaining how to read the numbers.
    """
    # Rank colors by their single highest occurrence anywhere, so the most
    # visually significant colors (even if localized) surface first.
    def max_pct(entry):
        return max(
            pct for views in entry["views"].values() for pct in views.values()
        )

    ranked = sorted(profile.items(), key=lambda kv: max_pct(kv[1]), reverse=True)
    top_colors = ranked[:top_n]

    color_list = [
        {"color": hex_color, "name": closest_css3_name(hex_color), "views": data["views"]}
        for hex_color, data in top_colors
    ]

    if not color_list:
        return ""

    guide = """COLOR PROFILE (pixel-measured, not a model interpretation)

Percentage of each view's region occupied by each dominant color.
Regions overlap vertically (upper 0-45%, middle 30-75%, lower 65-100% of
height), so one color can appear in two regions legitimately.
Percentages are per view+region, so they do not sum across views.
"name" is a nearest-CSS-name lookup, a convenience label only — if it looks
implausible, trust the hex and the verbal observations over the name.
Colors under 5% of a region and beyond the top 8 are omitted, so absence
from this list is not evidence a color is absent from the sprite.

DATA:
"""
    return guide + json.dumps(color_list, indent=2)