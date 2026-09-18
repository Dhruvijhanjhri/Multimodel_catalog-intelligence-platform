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

true_duplicate_groups = []

for (title, image), group in groups:
    attributes = group[
        ["brand", "product_type", "color", "category"]
    ].drop_duplicates()

    if len(attributes) == 1:
        true_duplicate_groups.append(group)

if true_duplicate_groups:
    true_duplicates = pd.concat(true_duplicate_groups)
else:
    true_duplicates = pd.DataFrame()

print("Total catalog records:", len(catalog))
print("Identical-attribute groups:", len(true_duplicate_groups))
print("Records in identical-attribute groups:", len(true_duplicates))

if not true_duplicates.empty:
    print(
        "Unique item IDs in these groups:",
        true_duplicates["item_id"].nunique()
    )

    print("\nSample identical-attribute groups:")

    grouped = true_duplicates.groupby(
        ["normalized_title", "normalized_image"]
    )

    for (title, image), group in list(grouped)[:20]:
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