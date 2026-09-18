import pandas as pd


METADATA_FILE = "embeddings/embedding_metadata.parquet"

metadata = pd.read_parquet(METADATA_FILE)

metadata["normalized_title"] = (
    metadata["title"]
    .astype(str)
    .str.strip()
    .str.lower()
)

metadata["normalized_image"] = (
    metadata["image_path"]
    .astype(str)
    .str.strip()
    .str.lower()
)

duplicate_pairs = metadata[
    metadata.duplicated(
        subset=["normalized_title", "normalized_image"],
        keep=False
    )
].sort_values(
    ["normalized_title", "normalized_image"]
)

print("Total catalog records:", len(metadata))
print("Records in exact title + exact image groups:", len(duplicate_pairs))

print(
    "Unique exact title + exact image groups:",
    duplicate_pairs[
        ["normalized_title", "normalized_image"]
    ].drop_duplicates().shape[0]
)

print("\nSample strong duplicate groups:")

groups = duplicate_pairs.groupby(
    ["normalized_title", "normalized_image"]
)

for (title, image), group in list(groups)[:20]:
    print("\nTitle:", title)
    print("Image:", image)
    print("Item IDs:", group["item_id"].tolist())