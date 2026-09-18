import pandas as pd


METADATA_FILE = "embeddings/embedding_metadata.parquet"

metadata = pd.read_parquet(METADATA_FILE)

title_counts = (
    metadata["title"]
    .astype(str)
    .str.strip()
    .str.lower()
    .value_counts()
)

duplicate_titles = title_counts[title_counts > 1]

print("Total catalog records:", len(metadata))
print("Records with repeated titles:", duplicate_titles.sum())
print("Unique repeated titles:", len(duplicate_titles))

print("\nTop repeated titles:")

for title, count in duplicate_titles.head(20).items():
    rows = metadata[
        metadata["title"].astype(str).str.strip().str.lower() == title
    ]

    print("\nTitle:", title)
    print("Count:", count)
    print(
        rows[
            ["item_id", "title", "target_category", "image_path"]
        ].to_string(index=False)
    )