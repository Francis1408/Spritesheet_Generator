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

    tok = CLIPTokenizer.from_pretrained( "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    length = len(tok(caption))

    exceed = True if length > LIMIT else False
    return {
        "length" : length,
        "exceed" : exceed
    }
