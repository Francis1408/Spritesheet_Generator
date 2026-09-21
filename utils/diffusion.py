"""
step4_generate.py

Standalone inference: given three known poses of a character, generate the
missing fourth and return it as a native-resolution pixel-art sprite.


Post-processing is the part validate() never did: the generated quadrant is
reduced back to native resolution by per-block median, then every pixel is
snapped to the palette of the THREE KNOWN poses. Not to the ground truth
palette -- at inference that does not exist, and using it would inflate results.

Usage:
    python step4_generate.py \
        --lora ./lora-4x-64x64/checkpoint-1500 \
        --sheet ./dataset/dev/42.png --missing back \
        --caption "pxchar, pixel art sprite sheet, female elf, pink hair, ..." \
        --out ./generated
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)

import utils.spritegeom as G
from config import SD15, DEFAULT_OUTPUT, CONTROLNET_INPAINT, SHEET_SIZE, QUAD_SIZE, FLATTEN_BG, LORA_PATH, FP32, DEVICE, STEP, CN_SCALE, GUIDANCE, NUM_SAMPLES, SEED, REDUCE


NEGATIVE = ("blurry, smooth, antialiased, gradient, photorealistic, 3d render, "
            "text, watermark, jpeg artifacts")


# ==========================================================================
# Pipeline
# ==========================================================================
def load_pipeline(lora_path, device="cuda", dtype=torch.float16, controlnet_model=CONTROLNET_INPAINT):
    """
    Build the inference pipeline and attach the trained LoRA.
    """
    controlnet = ControlNetModel.from_pretrained(controlnet_model, dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD15, controlnet=controlnet, dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)

    # UniPC converges in far fewer steps than the default PNDM
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    lora = Path(lora_path)
    weights = lora if lora.is_file() else lora / "pytorch_lora_weights.safetensors"
    if not weights.exists():
        raise SystemExit(f"LoRA weights not found: {weights}")
    pipe.load_lora_weights(str(weights.parent), weight_name=weights.name)

    pipe.set_progress_bar_config(disable=True)
    return pipe


# ==========================================================================
# Input assembly
# ==========================================================================
def quad_box(pos):
    col, row = G.QUADRANT_CELL[pos]
    return (col * G.QUAD_SIZE, row * G.QUAD_SIZE,
            (col + 1) * G.QUAD_SIZE, (row + 1) * G.QUAD_SIZE)


def sheet_from_sprites(sprites: dict) -> Image.Image:
    """
    Assemble a 512x512 sheet from native sprites, applying locked geometry.

    `sprites` maps position -> PIL image, either native (WINDOW_W x WINDOW_H)
    or an exact integer upscale of it. Missing positions are left as background.
    """
    sheet = Image.new("RGB", (G.SHEET_SIZE, G.SHEET_SIZE), G.FLATTEN_BG)
    for pos, img in sprites.items():
        img = img.convert("RGBA")
        if img.size == (G.WINDOW_W, G.WINDOW_H):
            content = G.upscale_nearest(img)
        elif img.size == (G.WINDOW_W * G.UPSCALE, G.WINDOW_H * G.UPSCALE):
            content = img
        else:
            raise ValueError(
                f"{pos}: got {img.size}, expected "
                f"{(G.WINDOW_W, G.WINDOW_H)} or "
                f"{(G.WINDOW_W*G.UPSCALE, G.WINDOW_H*G.UPSCALE)}")
        x0, y0, _, _ = quad_box(pos)
        ox, oy = G.content_offset()
        sheet.paste(content, (x0 + ox, y0 + oy), content)
    return sheet


def make_hint(sheet: Image.Image, missing: str) -> torch.Tensor:
    """
    ControlNet inpaint conditioning: [0, 1] with -1.0 marking missing pixels.
    """
    arr = np.array(sheet.convert("RGB")).astype(np.float32) / 255.0
    x0, y0, x1, y1 = quad_box(missing)
    arr[y0:y1, x0:x1, :] = -1.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


# ==========================================================================
# Post-processing
# ==========================================================================
def known_palette(sheet: Image.Image, known_positions) -> np.ndarray:
    """
    Palette pooled from the known quadrants only.

    This is the honest inference-time palette. A colour that appears solely in
    the missing pose is unavailable, and that limitation is real -- do not
    substitute the ground-truth palette when evaluating.
    """
    cols = []
    for pos in known_positions:
        x0, y0, x1, y1 = quad_box(pos)
        native = G.quadrant_to_native(sheet.crop((x0, y0, x1, y1)))
        cols.append(native.reshape(-1, 3))
    pal = np.unique(np.concatenate(cols, axis=0), axis=0)
    return pal.astype(np.uint8)


def extract_native(sheet: Image.Image, pos: str, palette=None, reduce="median"):
    """Generated quadrant -> native sprite, optionally palette-snapped."""
    x0, y0, x1, y1 = quad_box(pos)
    native = G.quadrant_to_native(sheet.crop((x0, y0, x1, y1)), reduce=reduce)
    if palette is not None and len(palette):
        native = G.snap_palette(native, palette)
    return native


# ==========================================================================
def generate(pipe, sheet, missing, caption, *, steps=30, guidance=7.5,
             cn_scale=1.0, seed=42, device="cuda", reduce="median"):
    """
    Returns dict with:
        raw_sheet   PIL, the model's full 512x512 output
        composited  PIL, known quadrants preserved + generated one inserted
        native      np.uint8 (WINDOW_H, WINDOW_W, 3), palette-snapped sprite
    """
    hint = make_hint(sheet, missing)
    gen = torch.Generator(device=device).manual_seed(seed)

    raw = pipe(
        caption,
        negative_prompt=NEGATIVE,
        image=hint,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=cn_scale,
        generator=gen,
    ).images[0]

    x0, y0, x1, y1 = quad_box(missing)
    comp_raw = sheet.copy()
    comp_raw.paste(raw.crop((x0, y0, x1, y1)), (x0, y0))

    known = [p for p in G.POSITIONS if p != missing]
    native = extract_native(raw, missing, known_palette(sheet, known), reduce)

    # rebuild the composited sheet from the cleaned sprite so what you see
    # matches what the metrics score
    clean = Image.fromarray(native).convert("RGBA")
    ox, oy = G.content_offset()
    comp_clean = sheet.copy()
    comp_clean.paste(G.upscale_nearest(clean), (x0 + ox, y0 + oy))

    return {
        "raw_sheet": raw, 
        "comp_raw": comp_raw, 
        "comp_clean": comp_clean,
        "native": native
    }


# ==========================================================================
def run_diffusion(image_crops, caption, missing_pos, parameters):
  
    """
    Generate one missing pose. Returns paths + metrics.
    Geometry comes from the checkpoint, so 4x and 5x LoRAs both just work.
    """

    G.apply_geometry(64, 64, parameters.get("upscale"))
    sheet_in, clipped = G.build_sprite_sheet_by_upscale(image_crops)
    if sheet_in is None or clipped:
        raise ValueError(f"Upscale {parameters.get('upscale')} clipped the crops. Please try a lower upscale")

    if not parameters.get("output"):
        out_dir = Path(DEFAULT_OUTPUT)
    else:
        out_dir = Path(parameters.get("output"))
        
    out_dir.mkdir(parents=True, exist_ok=True)

    # blank the target quadrant so no source pixel leaks into hint or palette
    sheet = sheet_in.copy()
    x0, y0, x1, y1 = quad_box(missing_pos)
    sheet.paste(Image.new("RGB", (G.QUAD_SIZE, G.QUAD_SIZE), G.FLATTEN_BG), (x0, y0))
        
    # Get lora_path
    lora_path = Path(LORA_PATH) / f"x{parameters.get('upscale')}_lora_weights.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f" LoRA with path {lora_path} does not exists")

    # ---- run ----
    print(f"loading {lora_path}")
    pipe = load_pipeline(lora_path, DEVICE, torch.float32 if FP32 else torch.float16)
    print(f"generating '{missing_pos}' | steps={STEP} "
          f"cfg={GUIDANCE} cn={CN_SCALE}")


    try:
        r = generate(
                pipe, 
                sheet=sheet, 
                missing=missing_pos, 
                caption=caption,
                steps=STEP,
                guidance=GUIDANCE,
                cn_scale=CN_SCALE,
                seed=SEED,
                device=DEVICE,
                reduce=REDUCE
            )
    finally:
        del pipe
        torch.cuda.empty_cache()

    snapped = out_dir / f"{missing_pos}_crop.png"
    raw     = out_dir / f"{missing_pos}_crop_raw.png"
    
    Image.fromarray(r["native"]).save(snapped)
    r["comp_raw"].save(raw)
    r["raw_sheet"].save(out_dir / f"{missing_pos}_sheet_raw.png")
    r["comp_clean"].save(out_dir / f"{missing_pos}_sheet_comp_clean.png")

    return {
        "crop_snapped": str(snapped),
        "crop_raw": str(raw),
        "sheet": str(out_dir / f"{missing_pos}_sheet.png"),
        "palette_size": int(len(r["palette"])),
    }

