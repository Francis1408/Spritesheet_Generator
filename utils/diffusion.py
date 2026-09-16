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

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)

import spritegeom as G
from config import SD15, CONTROLNET_INPAINT, SHEET_SIZE, QUAD_SIZE, FLATTEN_BG, LORA_PATH, FP32, DEVICE, STEP, CN_SCALE, GUIDANCE, NUM_SAMPLES, SEED, REDUCE



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
    comp = sheet.copy()
    comp.paste(raw.crop((x0, y0, x1, y1)), (x0, y0))

    known = [p for p in G.POSITIONS if p != missing]
    native = extract_native(raw, missing, known_palette(sheet, known), reduce)

    # rebuild the composited sheet from the cleaned sprite so what you see
    # matches what the metrics score
    clean = Image.fromarray(native).convert("RGBA")
    ox, oy = G.content_offset()
    comp.paste(G.upscale_nearest(clean), (x0 + ox, y0 + oy))

    return {"raw_sheet": raw, "composited": comp, "native": native}


# ==========================================================================
def run_diffusion(image_path, caption, missing_pos):
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--lora", required=True, help="checkpoint dir or .safetensors")
    # ap.add_argument("--missing", required=True, choices=list(G.POSITIONS))
    # ap.add_argument("--caption", default=None)
    # ap.add_argument("--out", default="./generated")

    # ap.add_argument("--sheet", default=None, help="eval mode: complete 512x512 sheet")
    # for pos in G.POSITIONS:
    #     ap.add_argument(f"--{pos}", default=None, help=f"real mode: {pos} sprite")

    # ap.add_argument("--steps", type=int, default=30)
    # ap.add_argument("--guidance", type=float, default=7.5)
    # ap.add_argument("--cn_scale", type=float, default=1.0)
    # ap.add_argument("--seed", type=int, default=42)
    # ap.add_argument("--num_samples", type=int, default=1,
    #                 help="generate N variants with consecutive seeds")
    # ap.add_argument("--reduce", default="median", choices=["median", "mean", "mode"])
    # ap.add_argument("--device", default="cuda")
    # ap.add_argument("--fp32", action="store_true")
    # args = ap.parse_args()

    if G.WINDOW_W is None:
        raise SystemExit("Geometry not locked. Set spritegeom.py first.")


    # ---- assemble input ----
    truth = None
    if image_path:
        truth = Image.open(image_path).convert("RGB")
        if truth.size != (SHEET_SIZE, SHEET_SIZE):
            raise SystemExit(f"sheet is {truth.size}, expected " f"{SHEET_SIZE}x{SHEET_SIZE}")
        sheet = truth.copy()
        # blank the missing quadrant so no ground-truth pixel can leak into
        # the palette or the composite
        x0, y0, x1, y1 = quad_box(missing_pos)
        sheet.paste(Image.new("RGB", (QUAD_SIZE, QUAD_SIZE), FLATTEN_BG), (x0, y0))
        


    # ---- run ----
    print(f"loading {LORA_PATH}")
    pipe = load_pipeline(LORA_PATH, DEVICE, torch.float32 if FP32 else torch.float16)
    print(f"generating '{missing_pos}' | steps={STEP} "
          f"cfg={GUIDANCE} cn={CN_SCALE}")

    for i in range(NUM_SAMPLES):
        seed = SEED + i
        r = generate(pipe, sheet, missing_pos, caption, steps=STEP,
                     guidance=GUIDANCE, cn_scale=CN_SCALE, seed=seed,
                     device=DEVICE, reduce=REDUCE)
        tag = f"{missing_pos}_seed{seed}"

        # r["composited"].save(out / f"{tag}_sheet.png") 

        # Image.fromarray(r["native"]).save(out / f"{tag}_native.png")
        # G.upscale_nearest(Image.fromarray(r["native"]).convert("RGBA"), 8).save(
        #     out / f"{tag}_native_x8.png")

        # Apply Post-processing
        if truth is not None:
            gt = G.quadrant_to_native(truth.crop(quad_box(missing_pos)))
            exact = float((r["native"] == gt).all(axis=2).mean())
            bg = np.array(G.FLATTEN_BG, dtype=np.uint8)
            fg = ~(gt == bg).all(axis=2)
            exact_fg = float((r["native"] == gt).all(axis=2)[fg].mean()) if fg.any() else 1.0
            print(f"  seed {seed}: exact {100*exact:.1f}% | " f"exact on character pixels {100*exact_fg:.1f}%")

            pad = np.pad(gt, ((0, 0), (0, 2), (0, 0)))
            strip = np.concatenate([pad, r["native"]], axis=1)
            G.upscale_nearest(Image.fromarray(strip).convert("RGBA"), 8).save(
                out / f"{tag}_truth-vs-generated.png")
        else:
            print(f"seed {seed}: written")

    print(f"\n-> {out}")
