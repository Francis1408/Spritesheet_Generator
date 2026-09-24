from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort, send_from_directory
from utils.spritesheet_builder import build_sprite_sheet
from utils.token_counter import count_caption_tokens
from utils.diffusion import run_diffusion
from utils.metrics import run_metrics
import config
from utils.caption_generator import CaptionGenerator
import uuid, json
import cv2 as cv
import os
import re
import shutil


app = Flask(__name__, static_folder="static", static_url_path="")
JOBS = Path("/tmp/spritejobs") # Image caching path
JOBS.mkdir(parents=True, exist_ok=True)


# ======= JOB HELPERS ===========

def job_dir(job_id):
    d = JOBS / job_id
    if not d.is_dir():
        abort(404, "unknown job")
    return d


def read_state(d):
    return json.loads((d / "state.json").read_text())


def write_state(d, state):
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(d / "state.json")


def advance(d, name, **extra):
    """Mark a step finished. Extra kwargs are merged into state.json."""
    state = read_state(d)
    if name not in state["done"]:
        state["done"].append(name)
    state.update(extra)
    write_state(d, state)


def invalidate_from(d, name):
    """Drop this step and everything after it, so a re-run can't leave stale output."""
    order = [s["name"] for s in config.PIPELINE]
    stale = order[order.index(name):]

    for step in stale:
        for p in (d / "steps").glob(f"{step}.*"):
            p.unlink()

    state = read_state(d)
    state["done"] = [s for s in state["done"] if s not in stale]
    write_state(d, state)


def save_image(d, name, img):
    ok, buf = cv.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"could not encode {name} as png")

    out = d / "steps" / f"{name}.png"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(out)


def save_data(d, name, data):
    out = d / "steps" / f"{name}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(out)


def require_artifact(d, name, ext):
    p = d / "steps" / f"{name}.{ext}"
    if not p.exists():
        abort(409, f"run the {name} step first")
    return p

# ========== UTILS =============

def parse_diffusion_variables(request_form):

    f = request_form.form

    return {
        "steps"       : int(f.get("step", 30)),
        "guidance"    : float(f.get("guidance", 7.5)),
        "cn_scale"    : float(f.get("cn_scale", 1.0)),
        "seed"        : int(f.get("seed", 0)),
        "num_samples" : int(f.get("num_samples", 1)),
        "precision"   : f.get("precision", "fp16"),
        "reduce"      : f.get("reduce", "median"),
        "upscale"     : f.get("upscale", "4"),
        "caption"     : f.get("caption", "")
    }
    
# ======= JOB ROUTES ===========
@app.get("/api/pipeline")
def get_pipeline():
    return jsonify(config.PIPELINE)

@app.get("/api/minimages")
def get_minimages():
    return jsonify(value=config.MIN_IMAGES)


@app.route("/api/upscale-options")
def upscale_options():

    SAFETENSORS_RE = re.compile(r"^x(?P<upscale>\d+(?:\.\d+)?)_lora_weights\.safetensors$", re.I,)

    options = []
    with os.scandir(config.LORA_PATH) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            m = SAFETENSORS_RE.match(entry.name)
            if m:
                options.append({
                    "upscale": float(m.group("upscale")),
                    "filename": entry.name,
                })
    options.sort(key=lambda o: o["upscale"])
    return jsonify(options)


@app.post("/api/jobs")
def create_job():
    job_id = uuid.uuid4().hex
    d = JOBS / job_id
    (d / "steps").mkdir(parents=True)
    (d / "sources").mkdir(parents=True)
    write_state(d, {"done": [], "sources": {}})
    return {"job_id": job_id}

@app.get("/api/jobs/<job_id>/state")
def get_state(job_id):
    d = job_dir(job_id)
    state = read_state(d)

    artifacts = {}
    for p in sorted((d / "steps").iterdir()):
        if p.suffix == ".tmp":
            continue
        if p.suffix == ".json":
            artifacts[p.stem] ={
                "kind": "data",
                "file": p.name,
                "data": json.loads(p.read_text()),
            }
        else:
            artifacts[p.stem] = {
                "kind": "image",
                "file": p.name,
                "url": f"/api/jobs/{job_id}/steps/{p.name}",
            }

    state["artifacts"] = artifacts
    return state

@app.get("/api/jobs/<job_id>/steps/<name>")
def get_step_file(job_id, name):
    return send_from_directory(job_dir(job_id) / "steps", name, max_age=0, conditional=True)

@app.get("/api/jobs/<job_id>/sources/<name>")
def get_source_file(job_id, name):
    return send_from_directory(job_dir(job_id) / "sources", name)

@app.get("/api/jobs/<job_id>/output/<name>")
def get_output_file(job_id, name):
    return send_from_directory(job_dir(job_id) / "output", name, max_age=0)

@app.post("/api/jobs/<job_id>/reset")
def reset_job(job_id):
    d = job_dir(job_id)
    invalidate_from(d, config.PIPELINE[0]['name'])
    return {"status": "success"}

@app.post("/api/jobs/<job_id>/export")
def export_job(job_id):

    f = request.form
    path = f.get("output_path")
    if not path:
        path = config.DEFAULT_OUTPUT
    
    d = job_dir(job_id)
    dst = Path(path) / job_id
    dst.mkdir(parents=True, exist_ok=True)
    for p in (d / "output").iterdir():
        shutil.copy(p, dst / p.name)
    return {"status": "success", "path": str(dst)}

# ======= MAIN ROUTES =============

# Main page
@app.route("/")
def index():
    return  app.send_static_file("index.html")


@app.post("/api/jobs/<job_id>/spritesheet")
def build_sprite_sheet_call(job_id):
    d = job_dir(job_id) 
    invalidate_from(d, "spritesheet")

    images = {}
    available_images = {}
    saved = {}
  
    for pos in ("front", "back", "left", "right"):
        f = request.files.get(pos)
        available_images[pos] = True
        if not f:
            images[pos] = False
            available_images[pos] = False
            continue

        data = f.read()
        ext = Path(f.filename or "").suffix.lower() or ".png"
        src = d / "sources" / f"{pos}{ext}"
        src.write_bytes(data)
        saved[pos] = src.relative_to(d).as_posix() 
        images[pos] = data

    # Checks if there is three images 
    if sum(1 for v in images.values() if v) < config.MIN_IMAGES:
        return jsonify({
            "status": "error",
            "message": f"need at least {config.MIN_IMAGES} images",
        }), 400

    try:
        sprite_sheet = build_sprite_sheet(images)

    except Exception as e:
        app.logger.exception("build_sprite_sheet failed")
        return jsonify({"status": "error", "message": str(e)}), 500

    save_image(d, "spritesheet", sprite_sheet)
    save_data(d, "spritesheet", available_images)

    # Save source files
    advance(d, "spritesheet", sources=saved)

    return {
        "status": "success",
        "step": "spritesheet",
        "preview": f"/api/jobs/{job_id}/steps/spritesheet.png",
    }

@app.post("/api/jobs/<job_id>/captioner_vlm")
def captioner_vlm_call(job_id):
    
    d = job_dir(job_id)
    sheet_path = require_artifact(d, 'spritesheet', 'png')
    available_pos_path = require_artifact(d, 'spritesheet', 'json')
    with open(available_pos_path) as f:
        available_pos = json.load(f)

    invalidate_from(d, "captioner_vlm")

    # Declare and run funtion passing the image path as paratemeter
    generator = CaptionGenerator()

    try:
        vlm_evidence = generator.run_vlm_phase(sheet_path, available_pos)

    except Exception as e:
        app.logger.exception("VLM run failed")
        return jsonify({"status": "error", "message": str(e)}), 500

    # Save evidence
    save_data(d, "captioner_vlm", vlm_evidence)
    advance(d, "captioner_vlm")

    return {
        "status": "success",
        "step": "captioner_vlm",
        "data": vlm_evidence,
    }

@app.post("/api/jobs/<job_id>/captioner_llm")
def captioner_llm_call(job_id):

    d = job_dir(job_id)
    sheet_path = require_artifact(d, 'spritesheet', 'png')

    available_pos_path = require_artifact(d, 'spritesheet', 'json')
    with open(available_pos_path) as f:
        available_pos = json.load(f)

    # Get views missing
    missing_views = [view for view, available in available_pos.items() if not available]

    vlm_evidence_path = require_artifact(d, 'captioner_vlm', 'json')
    with open(vlm_evidence_path) as f:
        vlm_evidence = json.load(f)

    invalidate_from(d, "captioner_llm")

    # Check evidenves integrity
    keys_to_check = ['positions', 'fullsheet', 'color']
    for key in keys_to_check:
        data = vlm_evidence.get(key)
        if not data:
            return jsonify({
                    "status": "error",
                    "message": f"Evidence {key} is missing",
                }), 400

    generator = CaptionGenerator()

    try:
        caption = generator.run_llm_phase(sheet_path, vlm_evidence, missing_views)


    except Exception as e:
        app.logger.exception("LLM run failed")
        return jsonify({"status": "error", "message": str(e)}), 500


    # Check if the caption generated exceeds the number of tokens
    results = count_caption_tokens(caption)

    # Save evidence
    save_data(d, "captioner_llm", {"data": caption, **results})
    advance(d, "captioner_llm")

    return {
        "status": "success",
        "step": "captioner_llm",
        "data": {"caption": caption, **results}
    }

@app.post("/api/jobs/<job_id>/diffusion")
def diffusion_call(job_id):

    d = job_dir(job_id)  
    saved_crops = {pos: str(d / rel) for pos, rel in read_state(d).get("sources", {}).items()}

    if not saved_crops:
        return jsonify({
            "status": "error",
            "message": f"No source images found. Restart the pipeline",
        }), 400
    
    available_pos_path = require_artifact(d, 'spritesheet', 'json')
    with open(available_pos_path) as f:
        available_pos = json.load(f)

    # Request variables
    request_data = parse_diffusion_variables(request)

    # Get views missing
    missing_views = [view for view, available in available_pos.items() if not available]
    if len(missing_views) != 1:
        return jsonify({"status": "error",
                        "message": f"expected exactly 1 missing view, got {missing_views}"}), 400
    missing_view = missing_views[0]

    # Get caption
    with open(require_artifact(d, 'captioner_llm', 'json')) as f:
        caption = json.load(f)["data"]

     # Validates if caption was edited
    if request_data['caption'] != caption:
        caption = request_data['caption'] 
        save_data(d, "captioner_llm", {"data": caption, **count_caption_tokens(caption)}) # Saves new caption

    invalidate_from(d, "diffusion")

    # Output dir
    out_dir = d / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        final_data = run_diffusion(image_crops=saved_crops, caption=caption, missing_pos=missing_view, parameters=request_data, out_dir=out_dir)
    
    except Exception as e:
        app.logger.exception("Diffusion failed")
        return jsonify({"status": "error", "message": str(e)}), 500

    artifact = {
        key: f"output/{Path(p).name}"
        for key, p in final_data.items()
    }

    save_data(d, "diffusion", artifact)
    advance(d, "diffusion")

    return {

        "status": "success",
        "step": "diffusion",
        "data": artifact,
    }

@app.post("/api/jobs/<job_id>/metrics")
def metrics_call(job_id):

    d = job_dir(job_id) 

    f = request.files.get("truth")
    if f is None:
        return jsonify({"status": "error", "message": "truth image required"}), 400

    data = f.read()
    ext = {Path(f.filename or '').suffix.lower() or '.png'}
    truth_path = d / "truth" / f"truth{ext}"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_bytes(data)


    with open(require_artifact(d, 'diffusion', 'json')) as fh:
        generated = {k: str(d / rel) for k, rel in json.load(fh).items()}

    # Filter the iamges that will be assessed
    filtered_generated = {k: rel for k, rel in generated if k in ['crop_snapped', 'crop_raw']}

    try:
        metrics = run_metrics(truth_path, filtered_generated)

    except Exception as e:
        app.logger.exception("generate metrics failed")
        return jsonify({"status": "error", "message": str(e)}), 500

    save_data(d, "metrics", metrics)
