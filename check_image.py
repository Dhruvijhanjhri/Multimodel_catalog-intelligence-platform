from pathlib import Path
import pandas as pd

manifest = pd.read_parquet(
    Path("data/processed/image_manifest.parquet")
)

filename = "8b0a71c3.jpg"

row = manifest[
    manifest["image_path"].str.contains(filename, na=False)
]

print("Matches:", len(row))

if len(row):
    print(row[["item_id", "title", "image_path"]])