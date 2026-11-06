
"""
Export explicitly corrected human reviews for Active Learning.

Safety rules:
- Approved reviews are excluded until explicit category confirmation exists.
- Rejected reviews require a valid corrected_category.
- Only the latest review per product is considered.
- Original training items are excluded from new_reviewed.
- Missing image files and invalid labels are excluded.
- This script does not modify the active classifier or embeddings.
"""

from pathlib import Path
import argparse
import sys
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from services.database.postgres import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "active_learning"
OUTPUT_FILE = OUTPUT_DIR / "reviewed_training_data.parquet"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.parquet"

VALID_CATEGORIES = {
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen",
}

QUERY = """
SELECT DISTINCT ON (pr.product_id)
    pr.id AS review_id,
    pr.product_id,
    pr.prediction_id,
    pr.reviewer,
    pr.decision,
    pr.corrected_category,
    pr.review_reason,
    pr.feedback,
    pr.created_at AS review_created_at,

    p.item_id,
    p.title,
    p.brand,
    p.product_type,
    p.color,
    p.category AS predicted_category,
    p.source,

    pp.predicted_category AS linked_prediction_category,
    pp.confidence,
    pp.model_version,

    pi.image_path

FROM product_reviews pr

JOIN products p
    ON p.id = pr.product_id

LEFT JOIN product_predictions pp
    ON pp.id = pr.prediction_id

LEFT JOIN product_images pi
    ON pi.product_id = p.id
    AND pi.is_primary = TRUE

ORDER BY
    pr.product_id,
    pr.created_at DESC NULLS LAST,
    pr.id DESC
"""


def resolve_image_path(value):
    if pd.isna(value) or not str(value).strip():
        return None

    path = Path(str(value))

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Export reviewed feedback for Active Learning."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview eligible and excluded records without writing output.",
    )
    args = parser.parse_args()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Original training split not found: {TRAIN_FILE}"
        )

    conn = get_connection()
    try:
        df = pd.read_sql_query(QUERY, conn)
    finally:
        conn.close()

    if df.empty:
        print("No product reviews found. No export written.")
        return

    # Only explicitly confirmed AI-category corrections are accepted.
    # Legacy reviews with no structured reason are excluded.
    df["training_label"] = None

    corrected_mask = (
        df["decision"].eq("Rejected")
        & df["review_reason"].eq("incorrect_ai_category")
        & df["corrected_category"].notna()
    )

    df.loc[corrected_mask, "training_label"] = (
        df.loc[corrected_mask, "corrected_category"]
        .astype(str)
        .str.strip()
    )

    # Validate corrected labels.
    valid_label_mask = df["training_label"].isin(VALID_CATEGORIES)

    missing_label_mask = df["training_label"].isna() | (
        df["training_label"].astype(str).str.strip().eq("")
    )

    invalid_label_mask = (
        ~valid_label_mask
        & ~missing_label_mask
    )

    # Load original training IDs and compare as strings.
    train_df = pd.read_parquet(TRAIN_FILE)
    train_items = set(train_df["item_id"].dropna().astype(str))

    df["dataset_membership"] = df["item_id"].astype(str).apply(
        lambda item_id: (
            "original_train"
            if item_id in train_items
            else "new_reviewed"
        )
    )

    df["feedback_source"] = "human_corrected"
    df["is_human_label"] = valid_label_mask

    image_paths = df["image_path"].apply(resolve_image_path)
    image_exists_mask = image_paths.apply(
        lambda path: path is not None and path.is_file()
    )

    original_train_mask = df["dataset_membership"].eq("original_train")

    has_title_mask = (
        df["title"].notna()
        & df["title"].astype(str).str.strip().ne("")
    )

    eligible_mask = (
        valid_label_mask
        & ~original_train_mask
        & image_exists_mask
        & has_title_mask
    )

    # Assign one primary exclusion reason to each ineligible record.
    df["exclusion_reason"] = None

    df.loc[missing_label_mask, "exclusion_reason"] = "missing_or_unconfirmed_label"
    df.loc[invalid_label_mask, "exclusion_reason"] = "invalid_corrected_label"
    df.loc[original_train_mask, "exclusion_reason"] = "already_in_original_train"
    df.loc[~image_exists_mask, "exclusion_reason"] = "missing_image_file"
    df.loc[~has_title_mask, "exclusion_reason"] = "missing_title"

    training_df = df.loc[eligible_mask].copy()

    # Store paths in the form expected by the embedding generator.
    training_df["image_path"] = training_df["image_path"].astype(str)

    # Stable product-level deduplication.
    training_df = training_df.drop_duplicates(
        subset=["product_id"],
        keep="last",
    )

    # Keep the schema expected by the current embedding pipeline,
    # plus useful review audit fields.
    if args.dry_run:
        print("\nDRY RUN: No output file was written.")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        training_df.to_parquet(OUTPUT_FILE, index=False)

    print()
    print("-" * 50)
    print("Active Learning Feedback Export")
    print("-" * 50)
    print("Latest reviews                 :", len(df))
    print("Approved reviews excluded     :", int(df["decision"].eq("Approved").sum()))
    print(
        "Missing review reason          :",
        int(df["review_reason"].isna().sum()),
    )
    print(
        "Non-category rejection reasons :",
        int(
            (
                df["decision"].eq("Rejected")
                & df["review_reason"].notna()
                & ~df["review_reason"].eq("incorrect_ai_category")
            ).sum()
        ),
    )
    print("Invalid corrected labels      :", int(invalid_label_mask.sum()))
    print("Original training items       :", int(original_train_mask.sum()))
    print("Missing image files           :", int((~image_exists_mask).sum()))
    print("Eligible new corrected labels :", len(training_df))
    print("Excluded overall              :", len(df) - len(training_df))
    print()
    print("Eligible label distribution:")
    if args.dry_run and not training_df.empty:
        print("\nEligible product details:")
        print(
            training_df[
                [
                    "product_id",
                    "item_id",
                    "title",
                    "predicted_category",
                    "corrected_category",
                    "review_reason",
                    "image_path",
                ]
            ].to_string(index=False)
        )
    if training_df.empty:
        print("No eligible samples.")
    else:
        print(training_df["training_label"].value_counts().to_string())

    print()
    print("Exclusion reasons:")
    excluded_df = df.loc[~eligible_mask]

    if excluded_df.empty:
        print("No excluded records.")
    else:
        print(excluded_df["exclusion_reason"].value_counts().to_string())
    print()
    print("Output:", OUTPUT_FILE)
    print()
    print("Note: existing embedding arrays were not changed.")


if __name__ == "__main__":
    main()
