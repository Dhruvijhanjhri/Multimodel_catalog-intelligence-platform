from flask import Flask, render_template, request, jsonify
import pandas as pd
from pathlib import Path

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

manifest = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "image_manifest.parquet"
)


@app.route("/")
def home():
    return render_template("index.html")

@app.post("/get-title")
def get_title():

    image = request.files["image"]

    filename = image.filename

    row = manifest[
        manifest["image_path"].str.endswith(filename)
    ]

    if len(row) == 0:
        return jsonify({
            "title": ""
        })

    return jsonify({
        "title": row.iloc[0]["title"]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)