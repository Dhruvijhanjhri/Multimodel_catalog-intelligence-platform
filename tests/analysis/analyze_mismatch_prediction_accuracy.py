import pandas as pd


METADATA_FILE = "embeddings/embedding_metadata.parquet"


metadata = pd.read_parquet(METADATA_FILE)

predictions = pd.read_sql(
    """
    SELECT
        p.item_id,
        pp.predicted_category,
        pp.confidence
    FROM product_predictions pp
    JOIN products p
        ON pp.product_id = p.id
    """,
    __import__(
        "services.database.postgres",
        fromlist=["get_connection"]
    ).get_connection(),
)

df = metadata.merge(
    predictions,
    on="item_id",
    how="inner",
)

image_embeddings = __import__("numpy").load(
    "embeddings/train_image_embeddings.npy"
)

text_embeddings = __import__("numpy").load(
    "embeddings/train_text_embeddings.npy"
)

df["similarity"] = (
    image_embeddings * text_embeddings
).sum(axis=1)


df["prediction_correct"] = (
    df["predicted_category"]
    == df["target_category"]
)


thresholds = [
    ("< 0.10", df["similarity"] < 0.10),
    ("< 0.15", df["similarity"] < 0.15),
    ("< 0.175", df["similarity"] < 0.175),
    ("< 0.20", df["similarity"] < 0.20),
    ("< 0.25", df["similarity"] < 0.25),
    ("All records", df["similarity"] >= -999),
]


print("Total matched records:", len(df))

print("\nPrediction accuracy by mismatch threshold:")

for label, condition in thresholds:

    subset = df[condition]

    if len(subset) == 0:
        continue

    accuracy = subset["prediction_correct"].mean() * 100

    print(
        f"{label}: "
        f"{len(subset)} records | "
        f"accuracy = {accuracy:.2f}%"
    )

print("\nOverall prediction accuracy:")

print(
    f"{df['prediction_correct'].mean() * 100:.2f}%"
)

print("\nPrediction accuracy for non-mismatch records:")

non_mismatch = df["similarity"] >= 0.175

print(
    f"{non_mismatch.sum()} records | "
    f"{df.loc[non_mismatch, 'prediction_correct'].mean() * 100:.2f}%"
)