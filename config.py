SPRITE_SIZE = (512, 512)
NEW_DIMENSIONS = (256, 256)
SPRITE_POS = {
    "front":  (0, 0),
    "back" :  (256, 0),
    "left" :  (0, 256),
    "right":  (256, 256)
}
PIPELINE = [
    {"name": "spritesheet", "label": "Build sheet",      "needsFiles": True},
    {"name": "captioner",   "label": "Generate caption", "needsFiles": False},
    {"name": "diffusion",   "label": "Generate image",   "needsFiles": False},
]
MIN_IMAGES = 3