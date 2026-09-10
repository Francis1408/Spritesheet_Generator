import json 
from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort

app = Flask(__name__, static_folder="static", static_url_path="")

# ======= ROUTES =============

# Main page
@app.route("/")
def index():
    return  app.send_static_file("index.html")



