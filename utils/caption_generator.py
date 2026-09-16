import json
from utils.model_prompter import modelPrompterSingleton
from config import VLM_MODEL, LLM_MODEL
from utils.unload_model import unload_model


class CaptionGenerator(): 

    def __init__(self):
        pass

    # def load_captions(self):
    #     if CAPTIONS_FILE.exists():
    #         with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    #             return json.load(f)
    #     return {}

    # def save_captions(self, captions):
    #     with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
    #         json.dump(captions, f, indent=2, ensure_ascii=False)

    # def get_pending_images(self, captions, step="vlm"):
    #     """Return image paths that don't have a caption yet, sorted numerically."""

    #     image_files = sorted(DATASET_DIR.glob("*.png"), key=lambda p: int(p.stem))
    #     pending = []
    #     for p in image_files:
    #         if p.stem in captions:
    #             continue  

    #         position_evidence_path = GENERATOR_OUTPUT_FOLDER / p.stem / "positions.json"
    #         fullview_evidence_path = GENERATOR_OUTPUT_FOLDER / p.stem / "fullview.json"
    #         # Elegible for VLM pipeline
    #         if step == "vlm":
    #             if not position_evidence_path.exists() or not fullview_evidence_path.exists():
    #                 pending.append(p)

    #         # Elegible for LLM pipeline
    #         else:
    #             if position_evidence_path.exists() and fullview_evidence_path.exists():
    #                 pending.append(p)

    #     return pending

    def run_vlm_phase(self, img_path, available_pos, unload_after=True):
        """Phase 1: VLM evidence extraction only. Run this first, in bulk."""

        model_prompter = modelPrompterSingleton

        try:
            
            key = img_path.stem
            print(f"Extracting evidence: {img_path.name}...")
            try:
                return model_prompter.generate_vlm_evidence(str(img_path), available_pos, key)
            except Exception as e:
                raise e
                    
        finally:
            if unload_after:
                unload_model(VLM_MODEL)

    def run_llm_phase(self, image_path, evidences, missing_views, unload_after=True):
        """Phase 2: LLM synthesis/compression only. Run this after phase 1."""

        model_prompter = modelPrompterSingleton
        
        try:
            
            key = image_path.stem
            print(f"Captioning: {image_path.name}...")
            try:
                return model_prompter.generate_caption_from_evidence(key, evidences, missing_views)
            except Exception as e:
                raise e
        
        finally:
            if unload_after:
                unload_model(LLM_MODEL)



