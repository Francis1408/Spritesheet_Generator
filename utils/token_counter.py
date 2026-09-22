"""
token_captioner.py

Calculates the amount of tokens for a string passed as a parameter

Design notes:
- The count_captioner_tokens was designed to check if the generated caption exceeds the amount of tokens  
supported by the CLIP encoder inside the SD1.5
"""

import json
from pathlib import Path
from transformers import CLIPTokenizer
from config import LIMIT


def count_caption_tokens(caption):
    tok = CLIPTokenizer.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="tokenizer"
    )

    tokens = tok(caption, truncation=False, add_special_tokens=True)["input_ids"]

    length = len(tokens)
    exceed = length > LIMIT

    return {
        "length": length,
        "exceed": exceed
    }
