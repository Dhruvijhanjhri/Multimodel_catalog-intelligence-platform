from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from pathlib import Path
import os

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

manifest = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "image_manifest.parquet"
)
print(manifest.columns.tolist())
print(manifest.head(3))

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/get-title")
def get_title():

    image = request.files["image"]

    filename = image.filename
    print("Selected filename:", filename)

    try:
        row = manifest[
            manifest["image_path"].apply(
                lambda x: Path(str(x)).name == filename
            )
        ]

        print("Matches found:", len(row))

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"title": "Error"})
    print("Matches found:", len(row))

    if len(row) == 0:
        return jsonify({
            "title": ""
        })

    return jsonify({
        "title": row.iloc[0]["title"]
    })

@app.get("/catalog")
def catalog():

    products = []

    for _, row in manifest.head(100).iterrows():

        filename = os.path.basename(row["image_path"])

        products.append({

            "title": row["title"],

            "image": filename,

            "category": row["category"]

        })

    return jsonify(products)

@app.get("/categories")
def categories():

    categories = (
        manifest["category"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    return jsonify(categories)

@app.get("/products/<category>")
def products(category):

    df = (
        manifest[
            manifest["category"] == category
        ]
        .sort_values("title")
        .head(100)
    )

    products = []

    for _, row in df.iterrows():

        products.append({

            "title": row["title"],

            "image": os.path.basename(row["image_path"]),

            "category": row["category"]

        })

    return jsonify(products)

@app.get("/image/<filename>")
def get_image(filename):

    image_path = (
        Path(PROJECT_ROOT)
        / "data"
        / "raw"
        / "abo-images-small"
        / "images"
        / "small"
        / filename[:2]
        / filename
    )
    print("Looking for:", image_path)
    print("Exists:", image_path.exists())

    if image_path.exists():
        return send_file(image_path)

    return "", 404

@app.route("/admin")
def admin_dashboard():
    return render_template("admin_dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)