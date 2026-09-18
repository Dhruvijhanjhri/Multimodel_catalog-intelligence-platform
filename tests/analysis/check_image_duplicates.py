import pandas as pd
df = pd.read_parquet("data/processed/image_manifest.parquet")
dup = df[df.duplicated(["item_id", "image_id"], keep=False)]
print("Duplicate product-image rows:", len(dup))
print("Unique duplicate product-image pairs:", dup[["item_id", "image_id"]].drop_duplicates().shape[0])