import pandas as pd


METADATA_FILE = "embeddings/embedding_metadata.parquet"
TRAIN_FILE = "data/splits/train.parquet"


metadata = pd.read_parquet(METADATA_FILE)
train = pd.read_parquet(TRAIN_FILE)

train_columns = [
    "item_id",
    "title",
    "brand",
    "product_type",
    "color",
    "category",
]

catalog = metadata.merge(
    train[train_columns],
    on=["item_id", "title"],
    how="left",
)

# -----------------------------
# Signal 1: Low confidence
# -----------------------------

LOW_CONFIDENCE_THRESHOLD = 0.70

predictions = pd.read_sql(
    """
    SELECT
        product_id,
        predicted_category,
        confidence
    FROM product_predictions
    """,
    __import__("services.database.postgres", fromlist=["get_connection"]).get_connection(),
)

products = pd.read_sql(
    """
    SELECT
        id AS product_id,
        item_id
    FROM products
    """,
    __import__("services.database.postgres", fromlist=["get_connection"]).get_connection(),
)

predictions = predictions.merge(
    products,
    on="product_id",
    how="left",
)

catalog = catalog.merge(
    predictions[
        ["item_id", "confidence"]
    ],
    on="item_id",
    how="left",
)

low_confidence = (
    catalog["confidence"] < LOW_CONFIDENCE_THRESHOLD
)


# -----------------------------
# Signal 2: Image-text mismatch
# -----------------------------

image_embeddings = __import__(
    "numpy"
).load("embeddings/train_image_embeddings.npy")

text_embeddings = __import__(
    "numpy"
).load("embeddings/train_text_embeddings.npy")

# The embedding metadata corresponds exactly
# to the saved train embeddings.
similarities = (
    (image_embeddings * text_embeddings).sum(axis=1)
)

IMAGE_TEXT_THRESHOLD = 0.175

similarity_df = pd.DataFrame(
    {
        "item_id": metadata["item_id"],
        "image_text_similarity": similarities,
    }
)

catalog = catalog.merge(
    similarity_df,
    on="item_id",
    how="left",
)

image_text_mismatch = (
    catalog["image_text_similarity"] < IMAGE_TEXT_THRESHOLD
)


# -----------------------------
# Signal 3: Strong duplicate
# -----------------------------

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

attribute_columns = [
    "brand",
    "product_type",
    "color",
    "category",
]

catalog["strong_duplicate"] = False

groups = catalog.groupby(
    ["normalized_title", "normalized_image"]
)

for _, group in groups:

    if len(group) < 2:
        continue

    attributes = group[attribute_columns].drop_duplicates()

    if len(attributes) == 1:
        catalog.loc[
            group.index,
            "strong_duplicate"
        ] = True


strong_duplicate = catalog["strong_duplicate"]


# -----------------------------
# Combined review trigger
# -----------------------------

review_trigger = (
    low_confidence
    | image_text_mismatch
    | strong_duplicate
)


print("Total catalog records:", len(catalog))

print("\nIndividual signals:")
print("Low confidence:", int(low_confidence.sum()))
print("Image-text mismatch:", int(image_text_mismatch.sum()))
print("Strong duplicate:", int(strong_duplicate.sum()))

print("\nSignal overlaps:")
print(
    "Low confidence + mismatch:",
    int((low_confidence & image_text_mismatch).sum())
)

print(
    "Low confidence + duplicate:",
    int((low_confidence & strong_duplicate).sum())
)

print(
    "Mismatch + duplicate:",
    int((image_text_mismatch & strong_duplicate).sum())
)

print(
    "All three signals:",
    int(
        (
            low_confidence
            & image_text_mismatch
            & strong_duplicate
        ).sum()
    )
)

print(
    "\nUnique catalog records requiring review:",
    int(review_trigger.sum())
)

print(
    "Percentage requiring review:",
    round(
        review_trigger.mean() * 100,
        2
    ),
    "%"
)