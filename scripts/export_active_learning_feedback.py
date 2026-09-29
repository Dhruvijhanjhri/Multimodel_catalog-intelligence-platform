"""
Export human-reviewed products for Active Learning.

Eligible feedback:
1. Approved -> use the model's predicted category.
2. Rejected + corrected_category -> use the human-corrected category.
3. Rejected without corrected_category -> exclude from training export.

Only the latest review for each product is considered.
"""

from pathlib import Path
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from services.database.postgres import get_connection


# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "active_learning"
OUTPUT_FILE = OUTPUT_DIR / "reviewed_training_data.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# Load Reviewed Feedback
# ---------------------------------------------------

query = """
SELECT DISTINCT ON (pr.product_id)
    pr.id AS review_id,
    pr.product_id,
    pr.decision,
    pr.corrected_category,
    pr.feedback,
    pr.created_at AS review_created_at,

    p.item_id,
    p.title,
    p.brand,
    p.product_type,
    p.color,
    p.category AS predicted_category,
    p.source,

    pp.confidence,
    pp.model_version,

    pi.image_path

FROM product_reviews pr

JOIN products p
    ON p.id = pr.product_id

LEFT JOIN product_predictions pp
    ON pp.product_id = p.id

LEFT JOIN product_images pi
    ON pi.product_id = p.id
    AND pi.is_primary = TRUE

ORDER BY
    pr.product_id,
    pr.created_at DESC
"""


conn = get_connection()

try:
    df = pd.read_sql_query(query, conn)
finally:
    conn.close()


# ---------------------------------------------------
# Determine Training Label
# ---------------------------------------------------

def get_training_label(row):
    if row["decision"] == "Approved":
        return row["predicted_category"]

    if (
        row["decision"] == "Rejected"
        and pd.notna(row["corrected_category"])
        and str(row["corrected_category"]).strip() != ""
    ):
        return row["corrected_category"]

    return None


df["training_label"] = df.apply(
    get_training_label,
    axis=1
)


# ---------------------------------------------------
# Keep Only Training-Eligible Feedback
# ---------------------------------------------------

training_df = df[
    df["training_label"].notna()
].copy()


# ---------------------------------------------------
# Add Feedback Metadata
# ---------------------------------------------------

training_df["feedback_source"] = training_df["decision"].apply(
    lambda decision:
        "human_confirmed"
        if decision == "Approved"
        else "human_corrected"
)

training_df["is_human_label"] = True

# Mark whether the reviewed product already existed in the original training split.
train_items = set(
    pd.read_parquet(
        PROJECT_ROOT / "data" / "splits" / "train.parquet"
    )["item_id"]
)

training_df["dataset_membership"] = training_df["item_id"].apply(
    lambda item_id:
        "original_train"
        if item_id in train_items
        else "new_reviewed"
)


# ---------------------------------------------------
# Save
# ---------------------------------------------------

training_df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------
# Summary
# ---------------------------------------------------

print()
print("----------------------------------------")
print("Active Learning Feedback Export")
print("----------------------------------------")

print("Total latest reviews :", len(df))
print("Eligible samples     :", len(training_df))
print("Excluded samples     :", len(df) - len(training_df))

print()
print("Decision distribution:")
print(training_df["decision"].value_counts())

print()
print("Training label distribution:")
print(training_df["training_label"].value_counts())

print()
print("Output:")
print(OUTPUT_FILE)