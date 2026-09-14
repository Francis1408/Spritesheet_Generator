from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort, send_from_directory
from utils.spritesheet_builder import build_sprite_sheet
import uuid, json
import cv2 as cv

app = Flask(__name__, static_folder="static", static_url_path="")
JOBS = Path("/tmp/spritejobs") # Image caching path


# ======= CACHING ROUTES ===========
def job_dir(job_id):
    d = JOBS / job_id
    if not d.is_dir():
        abort(404, "unknown job")
    return d

@app.post("/api/jobs")
def create_job():
    job_id = uuid.uuid4.hex()
    d = JOBS / job_id
    (d / "steps").mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"step": 0, "done": []}))
    return {"job_id": job_id}


# ======= UTILS ROUTES ===========
def advance(d, step, name, artifact=None):
    state = json.loads((d / "state.json").read_text())
    state["step"] = step
    if name not in state["done"]:
        state["done"].append(name)
    if artifact:
        state.setdefault("artifacts", {})[name] = artifact
    (d / "state.json").write_text(json.dumps(state))


@app.get("/api/jobs/<job_id>/state")
def get_state(job_id):
    d = job_dir(job_id)
    state = json.loads((d / "state.json").read_text())

    out = {}

    for p in sorted((d / "steps").iterdir()):
        entry = {"file": p.name}
        if p.suffix == ".json":
            entry["kind"] = "data"
            entry["data"] = json.loads(p.read_text()) # parsed text
        else:
            entry["kind"] = "image"
            entry["url"] = f"/api/jobs/{job_id}/steps/{p.name}" # returns as a img URL
        out[p.stem] = entry

    state["artifacts"] = out
    return state


@app.get("/api/jobs/<job_id>/steps/<name>")
def get_step_file(job_id, name):
    return send_from_directory(job_dir(job_id) / "steps", name, max_age=0, conditional=True)

# ======= MAIN ROUTES =============

# Main page
@app.route("/")
def index():
    return  app.send_static_file("index.html")


@app.post("/api/jobs/<job_id>/spritesheet")
def build_sprite_sheet_call(job_id):
    d = job_dir(job_id) 

    images = {}
    for pos in ("front", "back", "left", "right"):
        f = request.files.get(pos)
        if f is None:
            images[pos] = False
            continue
        images[pos] = f.read()

    try:
        sprite_sheet = build_sprite_sheet(images)
        out = d / "steps" / "01_sheet.png" # Save spritesheet
        cv.imwrite(str(out), sprite_sheet)

        # Mark the step 1 as finished
        state = json.loads((d / "state.json").read_text())
        state["step"] = 1
        state["done"].append("spritesheet")
        (d / "state.json").write_text(json.dumps(state))

        return {"status": "success", "step": 1, "preview": f"/api/jobs/{job_id}/steps/01_sheet.png"}

    except Exception as e:
        return jsonify({"status": "error", "message": f"{e}"})
            