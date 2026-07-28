from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

# Load training split
train_df = pd.read_parquet(
    SPLIT_DIR / "train.parquet"
)

# Load image manifest
manifest_df = pd.read_parquet(
    PROCESSED_DIR / "image_manifest.parquet"
)

# Merge image path
train_df = train_df.merge(
    manifest_df[["item_id", "image_path"]],
    on="item_id",
    how="inner"
)

metadata = train_df[
    [
        "item_id",
        "title",
        "category",
        "image_path"
    ]
].copy()

metadata.rename(
    columns={
        "category": "target_category"
    },
    inplace=True
)

metadata.to_parquet(
    EMBEDDING_DIR / "embedding_metadata.parquet",
    index=False
)

print("Metadata created successfully.")
print(metadata.head())
print(metadata.columns.tolist())