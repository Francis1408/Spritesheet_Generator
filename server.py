from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort, send_from_directory
from utils.spritesheet_builder import build_sprite_sheet
from config import SPRITE_POS, MIN_IMAGES, PIPELINE
from utils.caption_generator import CaptionGenerator
import uuid, json
import cv2 as cv


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
    order = [s["name"] for s in PIPELINE]
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
    


# ======= JOB ROUTES ===========
@app.get("/api/pipeline")
def get_pipeline():
    return jsonify(PIPELINE)

@app.get("/api/minimages")
def get_minimages():
    return jsonify(value=MIN_IMAGES)

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

@app.post("/api/jobs/<job_id>/reset")
def reset_job(job_id):
    d = job_dir(job_id)
    invalidate_from(d, PIPELINE[0]['name'])
    return {"status": "success"}

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
    saved = read_state(d).get("sources", {})

    for pos in ("front", "back", "left", "right"):
        f = request.files.get(pos)
        available_images[pos] = True
        if f is None:
            images[pos] = False
            available_images[pos] = False
            continue

        data = f.read()
        ext = Path(f.filename or "").suffix.lower() or ".png"
        src = d / "sources" / f"{pos}{ext}"
        src.write_bytes(data)
        saved[pos] = src.name
        images[pos] = data


    # Checks if there is three images 
    if sum(1 for v in images.values() if v) < MIN_IMAGES:
        return jsonify({
            "status": "error",
            "message": f"need at least {MIN_IMAGES} images",
        }), 400

    try:
        sprite_sheet = build_sprite_sheet(images)

    except Exception as e:
        app.logger.exception("build_sprite_sheet failed")
        return jsonify({"status": "error", "message": str(e)}), 500

    save_image(d, "spritesheet", sprite_sheet)
    save_data(d, "spritesheet", available_images)
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

    # Save evidence
    save_data(d, "captioner_llm", {"data": caption})
    advance(d, "captioner_llm")

    return {
        "status": "success",
        "step": "captioner_llm",
        "data": caption,
    }


