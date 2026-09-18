import pandas as pd


METADATA_FILE = "embeddings/embedding_metadata.parquet"
TRAIN_FILE = "data/splits/train.parquet"

metadata = pd.read_parquet(METADATA_FILE)
train = pd.read_parquet(TRAIN_FILE)

columns = [
    "item_id",
    "title",
    "brand",
    "product_type",
    "color",
    "category",
]

catalog = metadata.merge(
    train[columns],
    on=["item_id", "title"],
    how="left",
)

catalog["normalized_title"] = (
    catalog["title"]
    .astype(str)
    .str.strip()
    .str.lower()
)

catalog["normalized_image"] = (
    catalog["image_path"]
    .astype(str)
    .str.strip()
    .str.lower()
)

duplicate_groups = catalog[
    catalog.duplicated(
        subset=["normalized_title", "normalized_image"],
        keep=False
    )
].sort_values(
    ["normalized_title", "normalized_image"]
)

groups = duplicate_groups.groupby(
    ["normalized_title", "normalized_image"]
)

print("Total strong duplicate groups:", groups.ngroups)

same_attributes = 0
different_attributes = 0

print("\nAttribute analysis:")

for (title, image), group in groups:
    attribute_combinations = group[
        ["brand", "product_type", "color", "category"]
    ].drop_duplicates()

    if len(attribute_combinations) == 1:
        same_attributes += 1
    else:
        different_attributes += 1

print("Groups with identical attributes:", same_attributes)
print("Groups with different attributes:", different_attributes)

print("\nSample groups with different attributes:")

count = 0

for (title, image), group in groups:
    attribute_combinations = group[
        ["brand", "product_type", "color", "category"]
    ].drop_duplicates()

    if len(attribute_combinations) > 1:
        print("\nTitle:", title)
        print("Image:", image)
        print(
            group[
                [
                    "item_id",
                    "brand",
                    "product_type",
                    "color",
                    "category",
                ]
            ].to_string(index=False)
        )

        count += 1

        if count >= 20:
            break