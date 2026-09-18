import numpy as np
import pandas as pd


METADATA_FILE = "embeddings/embedding_metadata.parquet"
IMAGE_EMBEDDINGS = "embeddings/train_image_embeddings.npy"
TEXT_EMBEDDINGS = "embeddings/train_text_embeddings.npy"


metadata = pd.read_parquet(METADATA_FILE)

image_embeddings = np.load(IMAGE_EMBEDDINGS)
text_embeddings = np.load(TEXT_EMBEDDINGS)

similarities = (
    image_embeddings * text_embeddings
).sum(axis=1)

df = metadata[
    ["item_id", "title", "target_category"]
].copy()

df["similarity"] = similarities

print("Total records:", len(df))

print("\nSimilarity statistics:")
print(df["similarity"].describe())

print("\nMismatch counts by similarity range:")

ranges = [
    ("< 0.00", df["similarity"] < 0.00),
    ("0.00 - 0.05", (df["similarity"] >= 0.00) & (df["similarity"] < 0.05)),
    ("0.05 - 0.10", (df["similarity"] >= 0.05) & (df["similarity"] < 0.10)),
    ("0.10 - 0.15", (df["similarity"] >= 0.10) & (df["similarity"] < 0.15)),
    ("0.15 - 0.175", (df["similarity"] >= 0.15) & (df["similarity"] < 0.175)),
    ("0.175 - 0.20", (df["similarity"] >= 0.175) & (df["similarity"] < 0.20)),
    ("0.20 - 0.25", (df["similarity"] >= 0.20) & (df["similarity"] < 0.25)),
    ("0.25 - 0.30", (df["similarity"] >= 0.25) & (df["similarity"] < 0.30)),
    ("0.30+", df["similarity"] >= 0.30),
]

for label, condition in ranges:
    print(
        f"{label}: {int(condition.sum())}"
    )

print("\nLowest-similarity examples:")

print(
    df.sort_values("similarity")
    [
        [
            "item_id",
            "title",
            "target_category",
            "similarity",
        ]
    ]
    .head(20)
    .to_string(index=False)
)