# MODEL THAT CALLS THE MODEL VIA OLLAMA API
# LOADS THE PROMPT AS TXT AND PASS IT AS ARGUMENT TO THE CLIENT
# ENCODES THE IMAGE AND PASS IS A B64 FORMAT FOR THE MODEL
# RETURNS THE CAPTION GENERATED FROM THE MODEL


from config import PROMPT_PATH, VLM_MODEL, LLM_MODEL, OLLAMA_CHAT
from utils.crop import extract_position_crops, build_compact_sheet, scale_to_target, flatten
from utils.color_hinter import build_global_palette, compute_color_profile, format_color_profile_for_prompt
import base64
import requests
import json
from io import BytesIO


# Load Schemas
with open("schemas/pxchar_schema.json", "r", encoding="utf-8") as file:
    _PXCHAR_SCHEMA = json.load(file)
 
with open("schemas/synth_schema.json", "r", encoding="utf-8") as file:
    _SYNTH_SCHEMA = json.load(file)

with open("schemas/view_chema.json", "r", encoding="utf-8") as file:
   _VIEW_SCHEMA = json.load(file)


# Must match SYNTH_SCHEMA's required keys and the compression field order
CATEGORIES = [
    "character_type", "skin_tone", "body_features", "body",
    "extra_armor", "head", "eyes", "lower", "equipment",
]
 



class ModelPrompter():

    def __init__(self, vlm_model, llm_model, prompt_path):

        self.session = requests.Session()
        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.prompt_path = prompt_path


    def _chat(self, model, prompt, schema, *, think, num_ctx, num_predict, images=None, label=""):
    
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
 
        r = self.session.post(OLLAMA_CHAT, json={
            "model": model,
            "messages": [message],
            "think": think,
            "format": schema,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "repeat_penalty": 1.0,
                "seed": 42,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }, timeout=600)
        r.raise_for_status()
        d = r.json()
 
        if d.get("done_reason") != "stop":
            raise RuntimeError(
                f"{label or model}: done_reason={d.get('done_reason')} "
                f"eval_count={d.get('eval_count')}"
            )
        if d.get("prompt_eval_count", 0) > num_ctx * 0.9:
            print(f"  WARN {label}: prompt {d['prompt_eval_count']} "
                  f"near num_ctx {num_ctx}")
 
        return json.loads(d["message"]["content"])
    

    def encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")



    def build_final_caption(self, normalized_json):
 
        field_order = [
            "character_type",
            "skin_tone",
            "body_features",
            "body",
            "extra_armor",
            "head",
            "eyes",
            "lower",
            "equipment",
        ]

        parts = ["pxchar", "pixel art sprite sheet"]

        for field in field_order:
            value = normalized_json.get(field)

            if value is None:
                continue

            if isinstance(value, list):
                # Skip empty lists entirely
                cleaned_items = [str(item).strip() for item in value if item and str(item).strip()]
                if cleaned_items:
                    parts.extend(cleaned_items)

            elif isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    parts.append(cleaned)

        parts.append("rpg style")

        # Dedupt redundant descriptions
        seen = set()
        deduped = []
        for p in parts:
            key = p.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        return ", ".join(deduped)
        


    def synthetize_info(self, position_observations, full_sheet_observation, color_profile, missing_views=None):


        prompt = self.extract_file_text(self.prompt_path + "synthesis.txt")
 
        evidence = "POSITION OBSERVATIONS:\n" + json.dumps(position_observations, ensure_ascii=False)
        evidence += "\n\nFULL SHEET OBSERVATION:\n" + json.dumps(full_sheet_observation, ensure_ascii=False)
        evidence += "\n\n" + color_profile

        # Tell synthesis a view was absent so it does not read the gap as
        # evidence of absence.
        if missing_views:
            evidence += (
                "\n\nVIEWS NOT PRESENT IN THIS SHEET: "
                + ", ".join(missing_views)
                + "\nTreat these as unavailable, not as evidence a feature is absent."
            )

        prompt = prompt.replace("{EVIDENCE}", evidence)
    
        return self._chat(
            self.llm_model, prompt, _SYNTH_SCHEMA,
            think=False, num_ctx=16384, num_predict=2000,
            label="llm/synthesis",
        )


    def compress_info(self, synthesis):

        if isinstance(synthesis, str):
            synthesis = json.loads(synthesis)
 
        conclusions = {}
        for cat in CATEGORIES:
            entry = synthesis.get(cat)
            if isinstance(entry, dict):
                conclusions[cat] = entry.get("conclusion", "")
            elif isinstance(entry, str):
                conclusions[cat] = entry
 
        missing = [c for c in CATEGORIES if not conclusions.get(c)]
        if missing:
            print(f"  WARN synthesis missing conclusions: {', '.join(missing)}")
        if len(conclusions) < 5:
            raise ValueError("synthesis produced too few conclusions to compress")
 
        prompt = self.extract_file_text(self.prompt_path + "compression.txt")
        prompt = prompt.replace(
            "{SYNTHESIS_TEXT}",
            json.dumps(conclusions, indent=2, ensure_ascii=False),
        )
 
        return self._chat(
            self.llm_model, prompt, _PXCHAR_SCHEMA,
            think=False, num_ctx=2048, num_predict=400,
            label="llm/compression",
        )

     
    def analyze_full_sheet(self, sheet_img):

        prompt = self.extract_file_text(self.prompt_path + "full.txt")
        return self._chat(
            self.vlm_model, prompt, _VIEW_SCHEMA,
            think=False, num_ctx=16384, num_predict=1500,
            images=[self.encode_pil_image(sheet_img)],
            label="vlm/full_sheet",
        )
     

    def analyze_positional_image(self, position, crop=None):
    
            prompt = self.extract_file_text(self.prompt_path + "general.txt")
            prompt = prompt.replace(
                "{POSITION DESC}",
                self.extract_file_text(self.prompt_path + f"{position}.txt"),
            )
            return self._chat(
                self.vlm_model, prompt, _VIEW_SCHEMA,
                think=False, num_ctx=8192, num_predict=1200,
                images=[self.encode_pil_image(crop)],
                label=f"vlm/{position}",
            )


    def generate_vlm_evidence(self, image_path, available_pos, id):

        crops, missing = extract_position_crops(image_path, available_pos)
 
        if not crops:
            raise ValueError(f"{image_path}: no character detected")
        if missing:
            print(f"  NOTE {id}: no character in {', '.join(missing)}")
 
        position_observations = {}
        for position, raw_crop in crops.items():
            vlm_img = flatten(scale_to_target(raw_crop))
            try:
                position_observations[position] = \
                    self.analyze_positional_image(position, vlm_img)
            except Exception as e:
                print(f"WARN {id}/{position} failed: {e}")
                raise e
 
        
        sheet_img = build_compact_sheet(crops)
 
        try:
            full_observation = self.analyze_full_sheet(sheet_img)
        except Exception as e:
            print(f"WARN {id}/full_sheet failed: {e}")
            
 
        palette = build_global_palette(crops, n_colors=24)
        profile = compute_color_profile(crops, palette, min_significant_pct=0.01)

        return {
            "positions" : position_observations,
            "fullsheet" : full_observation,
            "color"     : profile 
        }


    def generate_caption_from_evidence(self, id, evidences, missing_views):
   
        position_observations = evidences['positions']
        full_observation = evidences['fullsheet']
        profile = evidences['color']
        
        color_hint = format_color_profile_for_prompt(profile, top_n=12)

        # Synthesis LLM call
        synthesis = self.synthetize_info(
            position_observations, full_observation, color_hint,
            missing_views,
        )
        
        # Compressed LLM call
        compressed = self.compress_info(synthesis)
        
        return self.build_final_caption(compressed)


    # ================================
    # MÉTODOS AUXILIARES
    # ================================

    def extract_file_text(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            raise FileNotFoundError(f"Arquivo nao encontrado em: {path}")


    def pil_image_content(self, image):
        b64 = self.encode_pil_image(image)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        }

    def image_content(self, image_path):
        b64 = self.encode_image(image_path)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        }

     # Save crop images in the buffer
    def encode_pil_image(self, image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def save_json(self, path, content):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(content, file, indent=4, ensure_ascii=False)


modelPrompterSingleton = ModelPrompter(VLM_MODEL, LLM_MODEL, PROMPT_PATH)

