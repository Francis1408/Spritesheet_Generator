SPRITE_SIZE = (512, 512)
NEW_DIMENSIONS = (256, 256)
SHEET_SIZE = 512
QUAD_SIZE = SHEET_SIZE // 2  # 256
FLATTEN_BG = (128, 128, 128)


SPRITE_POS = {
    "front":  (0, 0),
    "back" :  (256, 0),
    "left" :  (0, 256),
    "right":  (256, 256)
}
PIPELINE = [
    {"name": "spritesheet", "label": "Build sheet",      "needsFiles": True},
    {"name": "captioner_vlm",   "label": "Generate VLM evidences", "needsFiles": False},
    {"name": "captioner_llm",   "label": "Generate caption", "needsFiles": False},
    {"name": "diffusion",   "label": "Generate image",   "needsFiles": False},
]
MIN_IMAGES = 3

# ===== CAPTIONER VARIABLES  =======
VLM_MODEL = "qwen3-vl:8b-instruct"
LLM_MODEL = "qwen3:14b"
OLLAMA_CHAT = "http://localhost:11434/api/chat"
PROMPT_PATH = r"prompts/"

# ===== DIFFUSION VARIABLES  =======
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_INPAINT = "lllyasviel/control_v11p_sd15_inpaint"
LORA_PATH = "loras"


# ARGS
FP32 = False
DEVICE = "cuda"
STEP = 30
GUIDANCE = 7.5
CN_SCALE = 1.0
SEED = 42
NUM_SAMPLES = 1
REDUCE = "median"